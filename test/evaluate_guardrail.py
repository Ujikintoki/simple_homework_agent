import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from input_guardrails import LegalCheckOutput, guardrail_agent  # noqa: E402
from agents import Runner  # noqa: E402


@dataclass
class EvalRow:
    idx: int
    prompt: str
    expected: str
    category: str
    leaf_domain: str
    domain_path: str


def progress(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def read_jsonl(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            expected = str(item.get("expected_label", "")).strip().lower()
            if expected not in {"allowed", "blocked"}:
                continue
            rows.append(
                EvalRow(
                    idx=idx,
                    prompt=str(item.get("prompt", "")).strip(),
                    expected=expected,
                    category=str(item.get("category", "")),
                    leaf_domain=str(item.get("leaf_domain", "")),
                    domain_path=str(item.get("domain_path", "")),
                )
            )
    return rows


def label_from_illegal(is_illegal: bool) -> str:
    return "blocked" if is_illegal else "allowed"


async def infer_one(row: EvalRow, semaphore: asyncio.Semaphore, retries: int = 2) -> dict[str, Any]:
    async with semaphore:
        for attempt in range(retries + 1):
            try:
                progress(f"LLM infer idx={row.idx} try={attempt + 1}")
                result = await Runner.run(guardrail_agent, row.prompt, context=None)
                parsed = result.final_output_as(LegalCheckOutput)
                pred = label_from_illegal(parsed.is_illegal)
                return {
                    "idx": row.idx,
                    "prompt": row.prompt,
                    "expected_label": row.expected,
                    "predicted_label": pred,
                    "category": row.category,
                    "leaf_domain": row.leaf_domain,
                    "domain_path": row.domain_path,
                    "reasoning": parsed.reasoning,
                    "rejection_message": parsed.rejection_message,
                    "error": "",
                }
            except Exception as exc:
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return {
                    "idx": row.idx,
                    "prompt": row.prompt,
                    "expected_label": row.expected,
                    "predicted_label": "error",
                    "category": row.category,
                    "leaf_domain": row.leaf_domain,
                    "domain_path": row.domain_path,
                    "reasoning": "",
                    "rejection_message": "",
                    "error": str(exc),
                }


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def compute_metrics(pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in pred_rows if r["predicted_label"] in {"allowed", "blocked"}]
    errors = [r for r in pred_rows if r["predicted_label"] == "error"]

    tp = sum(1 for r in usable if r["expected_label"] == "blocked" and r["predicted_label"] == "blocked")
    tn = sum(1 for r in usable if r["expected_label"] == "allowed" and r["predicted_label"] == "allowed")
    fp = sum(1 for r in usable if r["expected_label"] == "allowed" and r["predicted_label"] == "blocked")
    fn = sum(1 for r in usable if r["expected_label"] == "blocked" and r["predicted_label"] == "allowed")

    p_blocked = safe_div(tp, tp + fp)
    r_blocked = safe_div(tp, tp + fn)
    f1_blocked = safe_div(2 * p_blocked * r_blocked, p_blocked + r_blocked)

    tp_allowed = tn
    fp_allowed = fn
    fn_allowed = fp
    p_allowed = safe_div(tp_allowed, tp_allowed + fp_allowed)
    r_allowed = safe_div(tp_allowed, tp_allowed + fn_allowed)
    f1_allowed = safe_div(2 * p_allowed * r_allowed, p_allowed + r_allowed)

    macro_f1 = (f1_blocked + f1_allowed) / 2
    accuracy = safe_div(tp + tn, len(usable))

    return {
        "total_rows": len(pred_rows),
        "usable_rows": len(usable),
        "error_rows": len(errors),
        "confusion_matrix_blocked_positive": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "accuracy": accuracy,
            "blocked": {"precision": p_blocked, "recall": r_blocked, "f1": f1_blocked},
            "allowed": {"precision": p_allowed, "recall": r_allowed, "f1": f1_allowed},
            "macro_f1": macro_f1,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate guardrail with confusion matrix and F1.")
    parser.add_argument(
        "--dataset",
        default="test/generated_data/boundary_dataset_20260326_161511.jsonl",
        help="Path to generated jsonl dataset.",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="test/eval_results")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = read_jsonl(dataset_path)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No valid rows found in dataset.")

    progress(f"Loaded rows: {len(rows)}")
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [asyncio.create_task(infer_one(row, semaphore)) for row in rows]

    preds: list[dict[str, Any]] = []
    done = 0
    report_every = max(1, len(tasks) // 20)
    for task in asyncio.as_completed(tasks):
        preds.append(await task)
        done += 1
        if done == len(tasks) or done % report_every == 0:
            progress(f"Inference progress: {done}/{len(tasks)}")

    report = compute_metrics(preds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    pred_path = output_dir / f"predictions_{ts}.jsonl"
    report_path = output_dir / f"metrics_{ts}.json"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Confusion Matrix (positive='blocked') ===")
    cm = report["confusion_matrix_blocked_positive"]
    print(f"TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    m = report["metrics"]
    print("\n=== Metrics ===")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Blocked F1: {m['blocked']['f1']:.4f}")
    print(f"Allowed F1: {m['allowed']['f1']:.4f}")
    print(f"Macro F1: {m['macro_f1']:.4f}")
    print(f"\nSaved report: {report_path}")
    print(f"Saved predictions: {pred_path}")


if __name__ == "__main__":
    asyncio.run(main())
