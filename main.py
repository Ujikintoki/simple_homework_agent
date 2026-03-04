from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.

import sys
import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, \
    RunContextWrapper

# ——————————————————————————————————————————————————————————————————————————————
# 1. import agents
# ——————————————————————————————————————————————————————————————————————————————
from my_agents.math_agent.agent import math_agent
from my_agents.history_agent.agent import history_agent


# ——————————————————————————————————————————————————————————————————————————————
# 2. input guardrail
# ——————————————————————————————————————————————————————————————————————————————
class LegalCheckOutput(BaseModel):
    reasoning: str
    is_illegal: bool
    rejection_message: str | None = None  # 如果 is_illegal 为 True，则提供具体的拒绝回复


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""
    You are the strictly logical security and relevance guardrail for the 'Smart Tutor' homework agent. 
    The focus of this system is on reliability and guardrails.
    Your ONLY job is to analyze the user's input and determine if it is a valid homework question.

    RULES FOR ACCEPTANCE (is_illegal = False):
    - Accept standard math and history homework questions.
    - Accept requests to summarize the conversation.

    RULES FOR REJECTION (is_illegal = True):
    If the question falls into the following categories, set `is_illegal` to True and provide the exact `rejection_message` context:
    1. Travel routing or logistics (e.g., traveling from Hong Kong to London). 
       - Rejection message: "Sorry I cannot help you on that as it is not a homework question related to math or history."
    2. Local, non-academic trivia (e.g., first president of HKUST ).
       - Rejection message: "Sorry that is not likely a history home work question as it is about a local small university." 
    3. Dangerous, hypothetical, or everyday non-homework scenarios (e.g., throwing a firecracker on a busy street).
       - Rejection message: "Sorry that is not a homework question."

    Think step-by-step in `reasoning` before making your final boolean decision.
    """,
    output_type=LegalCheckOutput,
)


@input_guardrail
async def content_guardrail(context: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which uses an agent to check if the input
    violates our homework-only policy.
    """
    # 提取最新的用户输入文本进行护栏检测 (假设 input 是一个字符串或消息列表)
    user_text = input[-1]["content"] if isinstance(input, list) else input

    result = await Runner.run(guardrail_agent, user_text, context=context.context)
    final_output = result.final_output_as(LegalCheckOutput)

    # 如果 is_illegal 为 True，将触发异常
    return GuardrailFunctionOutput(
        output_info={"reason": final_output.rejection_message},  # 将拒绝话术传递给异常捕获块
        tripwire_triggered=final_output.is_illegal,
    )


# ——————————————————————————————————————————————————————————————————————————————
# 3. handout triage
# ——————————————————————————————————————————————————————————————————————————————
triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the request.",
    handoffs=[math_agent, history_agent],
    input_guardrails=[content_guardrail],
)

# ——————————————————————————————————————————————————————————————————————————————
# 4. Main Async Loop
# ——————————————————————————————————————————————————————————————————————————————
async def main():
    # print welcome
    print("AI: Welcome to Smart Tutor, your personal math and history homework tutor. What can I help you today?")

    # first handout mission
    current_agent = triage_agent
    inputs = []

    while True:
        try:
            # get input
            user_msg = input("User: ")
            if user_msg.lower() in ['quit', 'exit']:
                print("AI: Goodbye!")
                break

            # record input message
            inputs.append({"role": "user", "content": user_msg})

            # run current agent: triage math or history
            result = await Runner.run(current_agent, inputs)

            # print output message
            print(f"AI: {result.final_output}")

            # update message
            inputs = result.to_input_list()

            # 状态转移：如果 Triage Agent 决定交接，current_agent 会在这里自动变为 math_agent 或 history_agent
            current_agent = result.current_agent

        except InputGuardrailTripwireTriggered as e:
            # guardrail
            rejection_message = e.guardrail_result.output.output_info.get(
                "reason",
                "Sorry that is not a homework question."  # fallback
            )
            print(f"AI: {rejection_message}")

            # 将这次“拒绝”也作为 AI 的回答塞回上下文中
            # AI 就拥有了“刚刚拒绝过用户”的记忆，对话不会断裂
            inputs.append({"role": "assistant", "content": rejection_message})

        except Exception as e:
            # 捕获其他可能的网络或 API 异常，防止程序崩溃退出
            print(f"\n[System Error]: {str(e)}")
            inputs.pop()  # 移除刚才那条导致错误的 user 消息，允许用户重试


if __name__ == "__main__":
    asyncio.run(main())