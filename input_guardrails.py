import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncAzureOpenAI
from agents import (Agent, 
                    Runner, 
                    input_guardrail, 
                    GuardrailFunctionOutput, 
                    RunContextWrapper,
                    OpenAIChatCompletionsModel,
                    set_tracing_disabled,
                )

load_dotenv()

# ——————————————————————————————————————————————————————————————————————————————
# 1. api setting
# ——————————————————————————————————————————————————————————————————————————————
api_key = os.getenv("AZURE_OPENAI_API_KEY")

azure_client = AsyncAzureOpenAI(
    api_key=api_key,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    default_headers={"Ocp-Apim-Subscription-Key": api_key},
    timeout=60.0,
)

azure_model = OpenAIChatCompletionsModel(
    openai_client=azure_client,
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
)

# ——————————————————————————————————————————————————————————————————————————————
# 2. guardrail agent
# ——————————————————————————————————————————————————————————————————————————————
class LegalCheckOutput(BaseModel):
    reasoning: str
    is_illegal: bool
    rejection_message: str | None = None

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""
    You are the strictly logical security and relevance guardrail for the 'Smart Tutor' homework agent. 
    The focus of this system is on reliability and guardrails.
    Your ONLY job is to analyze the user's input and determine if it is a valid homework question or interaction.

    RULES FOR ACCEPTANCE (is_illegal = False):
    - Accept standard homework questions for the following supported subjects:
      * Math: arithmetic, algebra, geometry, calculus, probability, statistics, mathematical proofs, and computational word problems (e.g., computing distance or speed).
      * History: significant global or national historical events, figures, and timelines.
      * Chemistry: chemical reactions, periodic table, molecular structures, stoichiometry, etc.
      * Philosophy: ethics, logic, epistemology, and philosophical theories.
      * Economics: micro/macroeconomics, supply and demand, market structures, etc.
      * Finance: investment principles, corporate finance, financial mathematics, etc.
    - Accept requests to summarize or review the conversation so far.
    - Accept requests where the user states their academic level or asks the tutor to adjust difficulty.
    - Accept greetings, thank-you messages, and other polite conversational exchanges.

    RULES FOR REJECTION (is_illegal = True):
    If the question falls into the following categories, set `is_illegal` to True and provide the exact `rejection_message`:
    1. **Travel routing, logistics, or trip planning** — asking for the BEST WAY to travel, flight recommendations, 
       itinerary planning, etc. (e.g., "I need to travel to London from Hong Kong. What is the best way?").
       NOTE: This is DIFFERENT from asking how to mathematically compute a distance — that is a math question and should be ACCEPTED.
       - Rejection message: "Sorry I cannot help you on that as it is not a homework question related to homework:)."
    2. **Local, non-academic trivia** — questions about local institutions, small universities, or non-globally-significant organizations 
       (e.g., "Who was the first president of Hong Kong University of Science and Technology?").
       - Rejection message: "Sorry that is not likely a history home work question as it is about a local small university."
    3. **Dangerous, hypothetical, or everyday non-homework scenarios** — questions about harmful activities, 
       hypothetical dangerous situations, or general life advice unrelated to academics 
       (e.g., "What would happen if someone throws a firecracker on a busy street?").
       - Rejection message: "Sorry that is not a homework question."

    CRITICAL DISTINCTION:
    - "How to compute the distance between Hong Kong and Shenzhen?" → ACCEPTED (math computation)
    - "What is the best way to travel from Hong Kong to London?" → REJECTED (travel logistics)
    - "Explain supply and demand" → ACCEPTED (economics homework)

    Think step-by-step in `reasoning` before making your final boolean decision.
    """,
    model=azure_model,
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