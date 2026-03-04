# input_guardrails.py
from pydantic import BaseModel
from agents import (Agent, 
                    Runner, 
                    input_guardrail, 
                    GuardrailFunctionOutput, 
                    RunContextWrapper,
                )

class LegalCheckOutput(BaseModel):
    reasoning: str
    is_illegal: bool
    rejection_message: str | None = None

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
    2. Local, non-academic trivia (e.g., first president of HKUST).
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
    user_text = input[-1]["content"] if isinstance(input, list) else input

    result = await Runner.run(guardrail_agent, user_text, context=context.context)
    final_output = result.final_output_as(LegalCheckOutput)

    return GuardrailFunctionOutput(
        output_info={"reason": final_output.rejection_message}, 
        tripwire_triggered=final_output.is_illegal,
    )