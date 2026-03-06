import sys
import os
import asyncio
from dotenv import load_dotenv

# openai-agents
from openai import AsyncAzureOpenAI
from agents import Agent, Runner, InputGuardrailTripwireTriggered, OpenAIChatCompletionsModel, set_tracing_disabled

# add agents
from input_guardrails import content_guardrail
from my_agents.math_agent.agent import math_agent
from my_agents.history_agent.agent import history_agent

load_dotenv()

# ——————————————————————————————————————————————————————————————————————————————
# 1. API Setting & Model Setup
# ——————————————————————————————————————————————————————————————————————————————

api_key = os.getenv("AZURE_OPENAI_API_KEY")

default_headers = {
    "Ocp-Apim-Subscription-Key": api_key
}

# 实例化azure client
azure_client = AsyncAzureOpenAI(
    api_key=api_key,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/'),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    default_headers=default_headers
)

azure_model = OpenAIChatCompletionsModel(
    openai_client=azure_client,
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
)

math_agent.model = azure_model
history_agent.model = azure_model

# ——————————————————————————————————————————————————————————————————————————————
# 2. Handout Triage
# ——————————————————————————————————————————————————————————————————————————————

triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the request.",
    handoffs=[math_agent, history_agent],
    model=azure_model,
    input_guardrails=[content_guardrail],
)

# ——————————————————————————————————————————————————————————————————————————————
# 3. Main Async Loop
# ——————————————————————————————————————————————————————————————————————————————

async def main():
    # Disable framework telemetry to stop the missing API key warnings
    set_tracing_disabled(True)
    
    print("AI: Welcome to Smart Tutor, your personal math and history homework tutor. What can I help you today?")

    current_agent = triage_agent
    inputs = []

    while True:
        try:
            # get input
            user_msg = input("User: ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ['quit', 'exit']:
                print("AI: Goodbye!")
                break

            # record input message
            inputs.append({"role": "user", "content": user_msg})

            # run current agent
            result = await Runner.run(current_agent, inputs)

            # print output message
            print(f"AI: {result.final_output}")

            # update message history 
            if hasattr(result, 'to_input_list'):
                inputs = result.to_input_list()
            else:
                inputs.append({"role": "assistant", "content": result.final_output})

            # update current_agent
            if hasattr(result, 'current_agent') and result.current_agent:
                current_agent = result.current_agent
                
        except InputGuardrailTripwireTriggered as e:
            reason = "Sorry, that is not a homework question."
            r
            try:
                if hasattr(e, 'guardrail_result') and hasattr(e.guardrail_result, 'output'):
                    guardrail_func_output = e.guardrail_result.output
                    if hasattr(guardrail_func_output, 'output_info'):
                        info = guardrail_func_output.output_info
                        if isinstance(info, dict):
                            reason = info.get("reason", reason)
            except Exception:
                pass

            print(f"AI: {reason}")
            
            inputs.append({"role": "assistant", "content": reason})
            current_agent = triage_agent

        # except InputGuardrailTripwireTriggered as e:
        #     # guardrail
        #     reason = "Sorry, that is not a homework question."
            
        #     if hasattr(e, 'guardrail_result') and e.guardrail_result.output_info:
        #         info = e.guardrail_result.output_info
        #         if isinstance(info, dict):
        #             reason = info.get("reasoning", info.get("reason", reason))
        #         elif hasattr(info, 'reasoning'):
        #             reason = info.reasoning

        #     print(f"AI: {reason}")
        #     inputs.append({"role": "assistant", "content": reason})
        #     current_agent = triage_agent

        except Exception as e:
            print(f"\n[System Error]: {str(e)}")
            if inputs and inputs[-1]["role"] == "user":
                inputs.pop()  

if __name__ == "__main__":
    asyncio.run(main())