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
        "content": "Welcome to Smart Tutor, your personal homework tutor. What can I help you today?",
        "is_warning": False
    }]

if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

if "current_agent" not in st.session_state:
    st.session_state.current_agent = triage_agent

if "generating" not in st.session_state:
    st.session_state.generating = False

# =============================
# 2. Self-healing logic after Stop button click
# =============================
# If the page refreshes while in 'generating' state, but the last message is not from the user,
# it indicates the process was interrupted by 'Stop'. Force unlock.
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
    
    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.agent_history.append({"role": "user", "content": prompt})
    st.rerun()

# =============================
# 5. Handle Generation Logic
# =============================
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                # Run backend logic
                result = asyncio.run(Runner.run(
                    st.session_state.current_agent,
                    st.session_state.agent_history
                ))

                response_text = result.final_output
                
                # Update Agent and history (Internal logic)
                if hasattr(result, 'current_agent') and result.current_agent:
                    st.session_state.current_agent = result.current_agent
                
                if hasattr(result, 'to_input_list'):
                    st.session_state.agent_history = result.to_input_list()
                else:
                    st.session_state.agent_history.append({"role": "assistant", "content": response_text})

            # Streaming print effect
            placeholder = st.empty()
            full_res = ""
            for chunk in response_text.split(" "):
                full_res += chunk + " "
                placeholder.markdown(full_res + "▌")
                time.sleep(0.03)
            placeholder.markdown(full_res)

            # Save to message list
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "is_warning": False
            })

        except InputGuardrailTripwireTriggered as e:
            # Handle guardrail exceptions
            reason = "I'm sorry, that is outside my scope."
            try:
                if hasattr(e, 'guardrail_result'):
                    reason = e.guardrail_result.output.output_info.get("reason", reason)
            except: pass

            st.session_state.messages.append({
                "role": "assistant",
                "content": reason,
                "is_warning": True
            })
            # Reset to triage agent after guardrail is triggered
            st.session_state.current_agent = triage_agent
            st.session_state.agent_history.append({"role": "assistant", "content": reason})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.pop()
            
        finally:
            st.session_state.generating = False
            st.rerun()