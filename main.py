# test code
# from agents import Agent, Runner

# agent = Agent(name="Assistant", instructions="You are a helpful assistant")

# result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
# print(result.final_output)

# # Code within the code,
# # Functions calling themselves,
# # Infinite loop's dance.

import sys
import os
import asyncio
from agents import Agent, Runner, InputGuardrailTripwireTriggered
from dotenv import load_dotenv

load_dotenv()

# ——————————————————————————————————————————————————————————————————————————————
# 1. import agents
# ——————————————————————————————————————————————————————————————————————————————
from my_agents.math_agent.agent import math_agent
from my_agents.history_agent.agent import history_agent

load_dotenv()

# from agents import Agent, Runner, InputGuardrailTripwireTriggered
from openai import AsyncAzureOpenAI

# -------------------------------------------------------------------------
# 关键新增：Azure 客户端配置与注入
# -------------------------------------------------------------------------
try:
    from agents import set_default_openai_client, set_tracing_disabled
    
    # 关闭全局追踪，消除 "skipping trace export" 的警告
    set_tracing_disabled(disabled=True)
    
    # 实例化 Azure 异步客户端
    azure_client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # 将 Azure 客户端设为全局默认
    set_default_openai_client(azure_client)
    
except ImportError:
    # 备选方案：如果你的 agents 库版本不支持 set_default_openai_client
    # 我们可以通过直接设置 OS 环境变量来欺骗底层客户端
    os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
    os.environ["OPENAI_BASE_URL"] = f"{os.getenv('AZURE_OPENAI_ENDPOINT', '').rstrip('/')}/openai/deployments/{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', '')}"
    os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "")
# ——————————————————————————————————————————————————————————————————————————————
# 2. input guardrail
# ——————————————————————————————————————————————————————————————————————————————
from input_guardrails import content_guardrail

# ——————————————————————————————————————————————————————————————————————————————
# 3. handout triage
# ——————————————————————————————————————————————————————————————————————————————
triage_agent = Agent(
    name="triage_agent",
    instructions="Handoff to the appropriate agent based on the request.",
    handoffs=[math_agent, history_agent],
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
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