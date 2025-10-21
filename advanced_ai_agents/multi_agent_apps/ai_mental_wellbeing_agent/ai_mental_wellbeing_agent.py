# ai_mental_wellbeing_agent.py
import os
import json
import uuid
from datetime import datetime

import streamlit as st
from autogen import (
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
    OpenAIWrapper,
    AFTER_WORK,
    UPDATE_SYSTEM_MESSAGE,
)

# =========================
#   APP / STORAGE SETUP
# =========================
os.environ["AUTOGEN_USE_DOCKER"] = "0"

APP_TITLE = "🧠 CerebraTech Mental Wellbeing Agent"
DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")

def _ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

def _read_history() -> list[dict]:
    _ensure_storage()
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_history(history: list[dict]):
    _ensure_storage()
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def _add_history_entry(entry: dict):
    history = _read_history()
    history.insert(0, entry)  # newest first
    _write_history(history)

def _delete_history_entry(entry_id: str):
    history = _read_history()
    history = [h for h in history if h.get("id") != entry_id]
    _write_history(history)

# Session defaults (form keys must exist to reload from history)
if "output" not in st.session_state:
    st.session_state.output = {"assessment": "", "action": "", "followup": ""}

defaults = {
    "provider": "-- Select Provider --",
    "api_key": "",
    "base_url": "",
    "model_name": "",
    "mental_state": "",
    "sleep_pattern": "7",
    "stress_level": 5,
    "support_system": [],
    "recent_changes": "",
    "current_symptoms": [],
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# =========================
#      SIDEBAR SETTINGS
# =========================
st.sidebar.title("⚙️ Settings")

st.sidebar.markdown(
    "<h3 style='margin-bottom:0.25rem'>🌐 "
    "<a href='https://cerebratech.xyz/' target='_blank' "
    "style='text-decoration:none;'>CerebraTech Website</a></h3>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

# Provider selector
st.session_state.provider = st.sidebar.selectbox(
    "Choose an LLM Provider",
    ["-- Select Provider --", "Ollama (no key required)", "OpenAI"],
    index=["-- Select Provider --", "Ollama (no key required)", "OpenAI"].index(st.session_state.provider),
    key="provider"
)

if st.session_state.provider == "OpenAI":
    st.session_state.api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key", type="password", key="api_key"
    )
    st.session_state.base_url = st.sidebar.text_input(
        "Optional: OpenAI Base URL",
        value=st.session_state.base_url or "",
        placeholder="leave blank for api.openai.com",
        key="base_url",
    )
    st.session_state.model_name = st.sidebar.text_input(
        "Model", value=st.session_state.model_name or "gpt-4o-mini", key="model_name"
    )

elif st.session_state.provider == "Ollama (no key required)":
    st.sidebar.info("Using **Ollama** — no API key required.")
    st.session_state.base_url = st.sidebar.text_input(
        "Ollama Base URL", value=st.session_state.base_url or "http://217.15.175.196:11434/v1", key="base_url"
    )
    st.session_state.model_name = st.sidebar.text_input(
        "Ollama Model", value=st.session_state.model_name or "llama3.2:1b", key="model_name"
    )
else:
    st.sidebar.warning("⚠️ Please select an LLM provider before proceeding.")

# Safety notice
st.sidebar.warning(
    "⚠️ Important Notice\n\n"
    "This application is a supportive tool and **not a replacement** for professional mental health care.\n\n"
    "If you are in crisis or having thoughts of self-harm:\n\n"
    "- Call **988** (Crisis Hotline)\n"
    "- Call **911** (Emergency Services)\n"
    "- Seek immediate professional help."
)

st.sidebar.divider()
st.sidebar.caption("🕘 History")

# History list (sidebar)
history = _read_history()
chosen_idx = None
if history:
    labels = [
        f"{i+1}. {h.get('created_at','')} — Stress {h.get('inputs',{}).get('stress_level','?')}/10 — {', '.join(h.get('inputs',{}).get('current_symptoms', [])[:2])}"
        for i, h in enumerate(history)
    ]
    chosen_idx = st.sidebar.selectbox("Saved entries", list(range(len(history))), format_func=lambda i: labels[i])
    col_h1, col_h2, col_h3, col_h4 = st.sidebar.columns([1,1,1,1])
    with col_h1:
        if st.button("View", use_container_width=True):
            st.session_state["view_history_id"] = history[chosen_idx]["id"]
    with col_h2:
        if st.button("Load to form", use_container_width=True):
            selected = history[chosen_idx]
            # Load inputs back to form keys
            for k, v in selected.get("inputs", {}).items():
                st.session_state[k] = v
            st.success("Fields reloaded from history.")
            st.experimental_rerun()
    with col_h3:
        if st.button("Delete", use_container_width=True):
            _delete_history_entry(history[chosen_idx]["id"])
            st.warning("Entry deleted.")
            st.experimental_rerun()
    with col_h4:
        # Export JSON for selected
        export_json = json.dumps(history[chosen_idx], ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Export", export_json, file_name="cerebratech_entry.json", use_container_width=True)

else:
    st.sidebar.info("No history yet. Generate a plan to create your first entry.")

# =========================
#        MAIN UI
# =========================
st.title(APP_TITLE)

st.info(
    "**Meet Your Mental Wellbeing Agent Team:**\n\n"
    "🧠 **Assessment Agent** – Evaluates your emotional and psychological state\n"
    "🎯 **Action Agent** – Builds an immediate action plan for support\n"
    "🔄 **Follow-up Agent** – Creates a long-term mental health strategy"
)

st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    st.session_state.mental_state = st.text_area(
        "How have you been feeling recently?",
        value=st.session_state.mental_state,
        placeholder="Describe your emotions, thoughts, or concerns...",
        key="mental_state",
    )
    st.session_state.sleep_pattern = st.select_slider(
        "Sleep Pattern (hours per night)",
        options=[f"{i}" for i in range(0, 13)],
        value=st.session_state.sleep_pattern,
        key="sleep_pattern",
    )

with col2:
    st.session_state.stress_level = st.slider(
        "Current Stress Level (1–10)",
        1, 10, int(st.session_state.stress_level),
        key="stress_level",
    )
    st.session_state.support_system = st.multiselect(
        "Current Support System",
        ["Family", "Friends", "Therapist", "Support Groups", "None"],
        default=st.session_state.support_system,
        key="support_system",
    )

st.session_state.recent_changes = st.text_area(
    "Any significant life changes or events recently?",
    value=st.session_state.recent_changes,
    placeholder="Job changes, relationships, loss, etc.",
    key="recent_changes",
)

st.session_state.current_symptoms = st.multiselect(
    "Current Symptoms",
    [
        "Anxiety",
        "Depression",
        "Insomnia",
        "Fatigue",
        "Loss of Interest",
        "Difficulty Concentrating",
        "Changes in Appetite",
        "Social Withdrawal",
        "Mood Swings",
        "Physical Discomfort",
    ],
    default=st.session_state.current_symptoms,
    key="current_symptoms",
)

# =========================
#      VALIDATION
# =========================
def validate_fields() -> list[str]:
    missing = []
    if st.session_state.provider == "-- Select Provider --":
        missing.append("Please select an LLM provider.")
    if st.session_state.provider == "OpenAI" and not st.session_state.api_key:
        missing.append("OpenAI API key is required for OpenAI mode.")
    if not st.session_state.mental_state.strip():
        missing.append("Please describe how you've been feeling.")
    if not st.session_state.recent_changes.strip():
        missing.append("Please describe any recent life changes.")
    if not st.session_state.current_symptoms:
        missing.append("Please select at least one symptom.")
    return missing

# =========================
#   GENERATE SUPPORT PLAN
# =========================
if st.button("Get Support Plan"):
    errors = validate_fields()
    if errors:
        for err in errors:
            st.error(err)
    else:
        with st.spinner("🤖 The agents are analyzing your input..."):
            try:
                task = f"""
Create a comprehensive and personalized mental health support plan based on the following details:

Emotional State: {st.session_state.mental_state}
Sleep: {st.session_state.sleep_pattern} hours per night
Stress Level: {st.session_state.stress_level}/10
Support System: {', '.join(st.session_state.support_system) if st.session_state.support_system else 'None reported'}
Recent Changes: {st.session_state.recent_changes}
Current Symptoms: {', '.join(st.session_state.current_symptoms)}
"""

                # Distinct prompts (so outputs don't parrot)
                system_messages = {
                    "assessment_agent": """
You are a compassionate mental health professional.
Acknowledge the user’s courage and analyze their emotional state with empathy and insight.
Focus on their mindset, tone, and underlying emotional needs.
Avoid suggesting actions yet — focus purely on understanding.
""",
                    "action_agent": """
You are a mental health action strategist.
Based on the assessment, design practical, evidence-based coping methods.
Include physical, social, and cognitive activities that fit their current stress level.
Ensure variety — do not repeat phrases used in assessment.
Use lists, bullet points, and short steps.
""",
                    "followup_agent": """
You are a recovery coach.
Develop a structured long-term plan with check-ins, progress tracking, and self-care milestones.
Focus on growth and prevention of relapse.
Maintain an optimistic and empowering tone.
Prefer concrete weekly or monthly milestones.
""",
                }

                # LLM config (works for both OpenAI / Ollama via OpenAIWrapper)
                llm_config = {
                    "api_key": st.session_state.api_key or "ollama",
                    "base_url": st.session_state.base_url or None,
                    "model": st.session_state.model_name,
                }

                # Shared context across agents
                context_variables = {"assessment": None, "action": None, "followup": None}

                def update_assessment(assessment_summary: str, ctx: dict) -> SwarmResult:
                    ctx["assessment"] = assessment_summary
                    st.sidebar.success("Assessment saved.")
                    return SwarmResult(agent="action_agent", context_variables=ctx)

                def update_action(action_summary: str, ctx: dict) -> SwarmResult:
                    ctx["action"] = action_summary
                    st.sidebar.success("Action Plan saved.")
                    return SwarmResult(agent="followup_agent", context_variables=ctx)

                def update_followup(followup_summary: str, ctx: dict) -> SwarmResult:
                    ctx["followup"] = followup_summary
                    st.sidebar.success("Long-term Strategy saved.")
                    return SwarmResult(agent="assessment_agent", context_variables=ctx)

                def update_system_message_func(agent: SwarmAgent, messages) -> str:
                    system_prompt = system_messages[agent.name]
                    agent.client = OpenAIWrapper(**agent.llm_config)
                    return system_prompt

                state_update = UPDATE_SYSTEM_MESSAGE(update_system_message_func)

                # Define agents
                assessment_agent = SwarmAgent(
                    "assessment_agent",
                    llm_config=llm_config,
                    functions=update_assessment,
                    update_agent_state_before_reply=[state_update],
                )
                action_agent = SwarmAgent(
                    "action_agent",
                    llm_config=llm_config,
                    functions=update_action,
                    update_agent_state_before_reply=[state_update],
                )
                followup_agent = SwarmAgent(
                    "followup_agent",
                    llm_config=llm_config,
                    functions=update_followup,
                    update_agent_state_before_reply=[state_update],
                )

                # Hand-offs
                assessment_agent.register_hand_off(AFTER_WORK(action_agent))
                action_agent.register_hand_off(AFTER_WORK(followup_agent))
                followup_agent.register_hand_off(AFTER_WORK(assessment_agent))

                # Run
                result, _, _ = initiate_swarm_chat(
                    initial_agent=assessment_agent,
                    agents=[assessment_agent, action_agent, followup_agent],
                    user_agent=None,
                    messages=task,
                    max_rounds=12,
                )

                st.session_state.output = {
                    "assessment": result.chat_history[-3]["content"],
                    "action": result.chat_history[-2]["content"],
                    "followup": result.chat_history[-1]["content"],
                }

                # Display
                with st.expander("🧾 Assessment Summary", expanded=False):
                    st.markdown(st.session_state.output["assessment"])
                with st.expander("🛠️ Action Plan", expanded=False):
                    st.markdown(st.session_state.output["action"])
                with st.expander("📅 Long-term Strategy", expanded=True):
                    st.markdown(st.session_state.output["followup"])

                # Save to history
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "provider": st.session_state.provider,
                    "model": st.session_state.model_name,
                    "inputs": {
                        "mental_state": st.session_state.mental_state,
                        "sleep_pattern": st.session_state.sleep_pattern,
                        "stress_level": st.session_state.stress_level,
                        "support_system": st.session_state.support_system,
                        "recent_changes": st.session_state.recent_changes,
                        "current_symptoms": st.session_state.current_symptoms,
                    },
                    "outputs": st.session_state.output,
                }
                _add_history_entry(entry)
                st.success("✅ Personalized mental health plan generated and saved to history!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# =========================
#   HISTORY VIEW (main)
# =========================
if "view_history_id" in st.session_state:
    view_id = st.session_state.pop("view_history_id")
    selected = next((h for h in _read_history() if h.get("id") == view_id), None)
    if selected:
        st.markdown("---")
        st.subheader("👀 Viewing Saved Entry")
        st.caption(f"Created at: {selected.get('created_at','')} • Provider: {selected.get('provider','')} • Model: {selected.get('model','')}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Emotional State**")
            st.write(selected["inputs"]["mental_state"])
            st.markdown("**Recent Changes**")
            st.write(selected["inputs"]["recent_changes"])
            st.markdown("**Support System**")
            st.write(", ".join(selected["inputs"]["support_system"]) or "None")
        with c2:
            st.markdown("**Stress Level**")
            st.write(f"{selected['inputs']['stress_level']}/10")
            st.markdown("**Sleep Pattern**")
            st.write(f"{selected['inputs']['sleep_pattern']} hours")
            st.markdown("**Symptoms**")
            st.write(", ".join(selected["inputs"]["current_symptoms"]))

        with st.expander("🧾 Assessment Summary", expanded=False):
            st.markdown(selected["outputs"]["assessment"])
        with st.expander("🛠️ Action Plan", expanded=False):
            st.markdown(selected["outputs"]["action"])
        with st.expander("📅 Long-term Strategy", expanded=False):
            st.markdown(selected["outputs"]["followup"])

        export_json = json.dumps(selected, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Export this entry (JSON)", export_json, file_name="cerebratech_entry.json")

# =========================
#       FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 15px;'>
        Powered by <b>CerebraTech</b> | Modified by <b>CerebraTech</b><br>
        <a href='https://cerebratech.xyz/' target='_blank' style='text-decoration: none; color: #1E88E5;'>
            🌐 Visit Website
        </a>
    </p>
    """,
    unsafe_allow_html=True,
)
