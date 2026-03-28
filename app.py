import streamlit as st
import time
import asyncio
from main import triage_agent, Runner, InputGuardrailTripwireTriggered

st.set_page_config(page_title="SmartTutor", page_icon="🤖")

st.title("🤖 SmartTutor")

# =============================
# 1. Session State Management
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Welcome to Smart Tutor, your personal homework tutor. I can handle your math, history, finance, economics, philosophy and chemistry homework questions. What can I help you today?",
        "is_warning": False
    }]

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

# 始终保持默认由 triage_agent 接管新请求
if "current_agent" not in st.session_state:
    st.session_state.current_agent = triage_agent

if "generating" not in st.session_state:
    st.session_state.generating = False
    
# =============================
# 1.5 Sidebar & Clear Chat Logic (新增防线1)
# =============================
with st.sidebar:
    if st.button("🗑️ (Clear Chat)"):
        # 强制重置所有状态，回到初始状态
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Welcome to Smart Tutor, your personal homework tutor. I can handle your math, history, finance, economics, philosophy and chemistry homework questions. What can I help you today?",
            "is_warning": False
        }]
        st.session_state.agent_history = []
        st.session_state.current_agent = triage_agent
        st.session_state.generating = False
        st.rerun()

# =============================
# 2. Self-healing logic after Stop button click
# =============================
if st.session_state.generating:
    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
        st.session_state.generating = False

# =============================
# 3. Chat History Rendering
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("is_warning"):
            st.markdown(f'''
                <div style="padding:8px 12px; border-radius:6px; background-color:#FFF9DB; border-left:4px solid #F4B400; font-size:0.92em;">
                <b>Notice:</b> {msg["content"]}
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# =============================
# 4. User Input
# =============================
if prompt := st.chat_input("Ask a homework question...", disabled=st.session_state.generating):
    st.session_state.generating = True
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.agent_history.append({"role": "user", "content": prompt})
    st.rerun()

# =============================
# 5. Handle Generation Logic
# =============================
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    st.session_state.current_agent = triage_agent
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                
                # ==========================================
                # 新增防线2：滑动窗口截断 (Sliding Window)
                # ==========================================
                MAX_MESSAGES = 12 # 保留最近的12条消息（包含当前用户输入，约6轮对话）
                
                # 如果历史记录超长，仅保留最近的 MAX_MESSAGES 条
                if len(st.session_state.agent_history) > MAX_MESSAGES:
                    # 确保截断后，列表的第一个元素是用户的提问 (role: user)
                    # 避免截断在 assistant 回复上导致对话逻辑错位
                    truncated_history = st.session_state.agent_history[-MAX_MESSAGES:]
                    if truncated_history[0]["role"] == "assistant":
                        truncated_history = truncated_history[1:]
                        
                    current_history = truncated_history
                else:
                    current_history = st.session_state.agent_history
                # ==========================================

                # Run backend logic (注意这里传入的是截断后的 current_history)
                result = asyncio.run(Runner.run(
                    st.session_state.current_agent,
                    current_history 
                ))

                response_text = result.final_output
                
                if not response_text or response_text.strip() == "":
                    response_text = "My core instructions are designed to maintain a safe academic environment. I cannot assist with this request."
                    is_warning = True
                else:
                    is_warning = False
                
                # 【核心修复】：移除锁定 current_agent 的逻辑，直接更新历史记录
                if hasattr(result, 'to_input_list'):
                    st.session_state.agent_history = result.to_input_list()
                else:
                    st.session_state.agent_history.append({"role": "assistant", "content": response_text})

            # Streaming print effect (仅针对正常回答)
            if not is_warning:
                placeholder = st.empty()
                full_res = ""
                for chunk in response_text.split(" "):
                    full_res += chunk + " "
                    placeholder.markdown(full_res + "▌")
                    time.sleep(0.03)
                placeholder.markdown(full_res)
            else:
                # 如果被内部逻辑判定为警告，直接显示
                st.markdown(f'''
                    <div style="padding:8px 12px; border-radius:6px; background-color:#FFF9DB; border-left:4px solid #F4B400; font-size:0.92em;">
                    <b>Notice:</b> {response_text}
                    </div>
                ''', unsafe_allow_html=True)

            # Save to message list
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "is_warning": is_warning 
            })

        except InputGuardrailTripwireTriggered as e:
            # 捕获护栏拦截
            reason = "My core instructions are designed to maintain a safe academic environment. I cannot bypass these protocols or assist with non-academic requests."
            try:
                if hasattr(e, 'guardrail_result'):
                    reason = e.guardrail_result.output.output_info.get("reason", reason)
            except: pass

            st.markdown(f'''
                <div style="padding:8px 12px; border-radius:6px; background-color:#FFF9DB; border-left:4px solid #F4B400; font-size:0.92em;">
                <b>Notice:</b> {reason}
                </div>
            ''', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": reason,
                "is_warning": True
            })
            st.session_state.agent_history.append({"role": "assistant", "content": reason})

        # except Exception as e:
        #     # 捕获其他系统错误
        #     error_msg = "My core instructions are designed to maintain a safe academic environment. I cannot fulfill this request."
            
        #     st.markdown(f'''
        #         <div style="padding:8px 12px; border-radius:6px; background-color:#FFF9DB; border-left:4px solid #F4B400; font-size:0.92em;">
        #         <b>Notice:</b> {error_msg}
        #         </div>
        #     ''', unsafe_allow_html=True)

        #     st.session_state.messages.append({
        #         "role": "assistant",
        #         "content": error_msg,
        #         "is_warning": True
        #     })
        #     st.session_state.agent_history.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            import traceback
            error_detail = str(e)
            error_msg = f"System Error: {error_detail}"
            
            print(f"[DEBUG ERROR] {error_detail}")
            print(traceback.format_exc())
            
            st.markdown(f'''
                <div style="padding:8px 12px; border-radius:6px; background-color:#FFEBEE; border-left:4px solid #D32F2F; font-size:0.92em; color:#B71C1C;">
                <b>Debug Info:</b> {error_msg}
                </div>
            ''', unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "is_warning": True
            })
            st.session_state.agent_history.append({"role": "assistant", "content": error_msg})
            
        finally:
            st.session_state.generating = False
            # 确保跑完这一轮后，状态归位，准备迎接下一轮的 triage
            st.session_state.current_agent = triage_agent
            st.rerun()