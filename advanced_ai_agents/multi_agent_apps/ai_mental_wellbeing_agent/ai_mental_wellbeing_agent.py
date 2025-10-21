import os
import streamlit as st
from autogen import (
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
    OpenAIWrapper,
    AFTER_WORK,
    UPDATE_SYSTEM_MESSAGE,
)

os.environ["AUTOGEN_USE_DOCKER"] = "0"

if "output" not in st.session_state:
    st.session_state.output = {"assessment": "", "action": "", "followup": ""}

# =========================
#      SIDEBAR SETTINGS
# =========================
st.sidebar.title("Model Provider")

provider = st.sidebar.selectbox(
    "Choose an LLM Provider",
    ["-- Select Provider --", "Ollama (no key required)", "OpenAI"],
    index=0,
)

api_key = None
base_url = None
model_name = None

if provider == "OpenAI":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    base_url = st.sidebar.text_input(
        "Optional: OpenAI Base URL", value="", placeholder="leave blank for api.openai.com"
    )
    model_name = st.sidebar.text_input("Model", value="gpt-4o-mini")

elif provider == "Ollama (no key required)":
    st.sidebar.info("Using **Ollama** — no API key required.")
    base_url = st.sidebar.text_input("Ollama Base URL", value="http://217.15.175.196:11434/v1")
    model_name = st.sidebar.text_input("Ollama Model", value="llama3.2:1b")

else:
    st.sidebar.warning("⚠️ Please select an LLM provider before proceeding.")

# -------------------------
#    Safety notice
# -------------------------
st.sidebar.warning(
    "⚠️ Important Notice\n\n"
    "This app is a supportive tool and **not** a substitute for professional mental health care.\n\n"
    "- If you are in crisis, call **988** (Crisis Hotline)\n"
    "- If in danger, call **911**\n"
    "- Seek professional help immediately."
)

# =========================
#        MAIN UI
# =========================
st.title("🧠 CerebraTech Mental Wellbeing Agent")

st.info(
    "**Meet Your Mental Wellbeing Agent Team:**\n\n"
    "🧠 **Assessment Agent** – Evaluates your emotional state\n"
    "🎯 **Action Agent** – Creates immediate action steps\n"
    "🔄 **Follow-up Agent** – Plans your long-term wellbeing strategy"
)

st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    mental_state = st.text_area(
        "How have you been feeling recently?",
        placeholder="Describe your emotions, thoughts, or concerns...",
    )
    sleep_pattern = st.select_slider(
        "Sleep Pattern (hours per night)",
        options=[f"{i}" for i in range(0, 13)],
        value="7",
    )

with col2:
    stress_level = st.slider("Current Stress Level (1–10)", 1, 10, 5)
    support_system = st.multiselect(
        "Current Support System",
        ["Family", "Friends", "Therapist", "Support Groups", "None"],
    )

recent_changes = st.text_area(
    "Any significant life changes or events recently?",
    placeholder="Job changes, relationships, loss, etc.",
)

current_symptoms = st.multiselect(
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
)

# =========================
#      VALIDATION
# =========================
def validate_fields():
    missing = []
    if provider == "-- Select Provider --":
        missing.append("Please select an LLM provider.")
    if provider == "OpenAI" and not api_key:
        missing.append("OpenAI API key is required for OpenAI mode.")
    if not mental_state.strip():
        missing.append("Please describe how you've been feeling.")
    if not recent_changes.strip():
        missing.append("Please describe any recent life changes.")
    if not current_symptoms:
        missing.append("Please select at least one symptom.")
    return missing


# =========================
#       MAIN LOGIC
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

Emotional State: {mental_state}
Sleep: {sleep_pattern} hours per night
Stress Level: {stress_level}/10
Support System: {', '.join(support_system) if support_system else 'None reported'}
Recent Changes: {recent_changes}
Current Symptoms: {', '.join(current_symptoms)}
"""

                # --- Ensure unique outputs by varying prompts slightly
                system_messages = {
                    "assessment_agent": """
You are an experienced mental health professional.
Start by acknowledging the user’s courage, then perform a detailed emotional assessment.
Focus on empathy, psychological insight, and understanding patterns in their behavior.
""",
                    "action_agent": """
You are a mental health action strategist.
Create a step-by-step plan with practical coping techniques, habits, and daily wellness routines.
Avoid repeating phrases used in the assessment stage.
""",
                    "followup_agent": """
You are a recovery and growth specialist.
Design a long-term wellness roadmap with realistic milestones, habit tracking, and follow-up structure.
Ensure the tone is positive and empowering, not repetitive.
""",
                }

                llm_config = {
                    "api_key": api_key or "ollama",
                    "base_url": base_url or None,
                    "model": model_name,
                }

                context_variables = {"assessment": None, "action": None, "followup": None}

                def update_assessment(assessment_summary, ctx):
                    ctx["assessment"] = assessment_summary
                    st.sidebar.success("Assessment: " + assessment_summary)
                    return SwarmResult(agent="action_agent", context_variables=ctx)

                def update_action(action_summary, ctx):
                    ctx["action"] = action_summary
                    st.sidebar.success("Action: " + action_summary)
                    return SwarmResult(agent="followup_agent", context_variables=ctx)

                def update_followup(followup_summary, ctx):
                    ctx["followup"] = followup_summary
                    st.sidebar.success("Follow-up: " + followup_summary)
                    return SwarmResult(agent="assessment_agent", context_variables=ctx)

                def update_system_message_func(agent: SwarmAgent, messages):
                    system_prompt = system_messages[agent.name]
                    agent.client = OpenAIWrapper(**agent.llm_config)
                    return system_prompt

                state_update = UPDATE_SYSTEM_MESSAGE(update_system_message_func)

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

                assessment_agent.register_hand_off(AFTER_WORK(action_agent))
                action_agent.register_hand_off(AFTER_WORK(followup_agent))
                followup_agent.register_hand_off(AFTER_WORK(assessment_agent))

                result, _, _ = initiate_swarm_chat(
                    initial_agent=assessment_agent,
                    agents=[assessment_agent, action_agent, followup_agent],
                    user_agent=None,
                    messages=task,
                    max_rounds=12,
                )

                # Show outputs
                st.session_state.output = {
                    "assessment": result.chat_history[-3]["content"],
                    "action": result.chat_history[-2]["content"],
                    "followup": result.chat_history[-1]["content"],
                }

                with st.expander("🧾 Assessment Summary"):
                    st.markdown(st.session_state.output["assessment"])
                with st.expander("🛠️ Action Plan"):
                    st.markdown(st.session_state.output["action"])
                with st.expander("📅 Long-term Strategy"):
                    st.markdown(st.session_state.output["followup"])

                st.success("✅ Personalized mental health plan generated successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
