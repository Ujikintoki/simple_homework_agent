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
from my_agents.chemistry_agent.agent import chemistry_agent
from my_agents.philosophy_agent.agent import philosophy_agent
from my_agents.economics_agent.agent import economics_agent
from my_agents.finance_agent.agent import finance_agent

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
    default_headers=default_headers,
    timeout=60.0,
)

azure_model = OpenAIChatCompletionsModel(
    openai_client=azure_client,
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
)

math_agent.model = azure_model
history_agent.model = azure_model
chemistry_agent.model = azure_model
philosophy_agent.model = azure_model
economics_agent.model = azure_model
finance_agent.model = azure_model

# ——————————————————————————————————————————————————————————————————————————————
# 2. Handout Triage
# ——————————————————————————————————————————————————————————————————————————————

triage_agent = Agent(
    name="triage_agent",
    instructions="""
    You are the routing agent for 'Smart Tutor', a homework tutoring system.

    YOUR RESPONSIBILITIES:
    1. For math-related homework questions, handoff to math_agent.
    2. For history-related homework questions, handoff to history_agent.
    2. For chemistry-related homework questions, handoff to chemistry_agent.
    2. For economics-related homework questions, handoff to economics_agent.
    2. For philosophy-related homework questions, handoff to philosophy_agent.
    2. For finance-related homework questions, handoff to finance_agent.
    3. For requests to summarize or review the conversation so far, DO NOT handoff. 
       Instead, provide a clear, structured summary of the entire conversation yourself, 
       including all questions asked, answers given, and key topics discussed.
    4. For greetings, thank-you messages, or other polite exchanges, respond directly 
       with a friendly, brief reply. Do not handoff for these.
    5. If the user states their academic level (e.g., "I'm a year one student"), 
       acknowledge it and remember it for context, then wait for their next question.
    6.  If the user specifies an incorrect subject (e.g., asking a math question in a history context),
       you should ignore the incorrect framing, identify the true subject of the question,
       and route it to the appropriate agent.Do NOT reject a question solely because the subject context provided by the user is inappropriate.
    """,
    handoffs=[math_agent, history_agent, chemistry_agent, philosophy_agent, economics_agent, finance_agent],
    model=azure_model,
    input_guardrails=[content_guardrail],
)
    
async def ask_agent(user_msg, inputs, current_agent):
    try:
        inputs.append({"role": "user", "content": user_msg})

        result = await Runner.run(current_agent, inputs)

        response = result.final_output

        # 更新对话历史
        if hasattr(result, 'to_input_list'):
            inputs = result.to_input_list()
        else:
            inputs.append({"role": "assistant", "content": response})

        # 更新当前 agent
        if hasattr(result, 'current_agent') and result.current_agent:
            current_agent = result.current_agent

        return response, inputs, current_agent

    except InputGuardrailTripwireTriggered as e:
        reason = "Sorry, that is not a homework question."
        inputs.append({"role": "assistant", "content": reason})
        return reason, inputs, current_agent

# ——————————————————————————————————————————————————————————————————————————————
# 3. Main Async Loop
# ——————————————————————————————————————————————————————————————————————————————

async def main():
    # Disable framework telemetry to stop the missing API key warnings
    set_tracing_disabled(True)
    
    print("AI: Welcome to Smart Tutor, your personal homework tutor. What can I help you today?")

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



        except Exception as e:
            print(f"\n[System Error]: {str(e)}")
            if inputs and inputs[-1]["role"] == "user":
                inputs.pop()  

if __name__ == "__main__":
    asyncio.run(main())