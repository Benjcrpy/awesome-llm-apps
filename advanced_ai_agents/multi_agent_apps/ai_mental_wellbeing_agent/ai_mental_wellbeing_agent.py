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

# Disable docker mode for autogen in this app
os.environ["AUTOGEN_USE_DOCKER"] = "0"

# Session state for output
if "output" not in st.session_state:
    st.session_state.output = {"assessment": "", "action": "", "followup": ""}

# =========================
#  SIDEBAR (Provider Switch)
# =========================
st.sidebar.title("Model Provider")

provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ["Ollama (no key required)", "OpenAI"],
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
else:
    st.sidebar.info("Using **Ollama** — no API key required.")
    base_url = st.sidebar.text_input("Ollama Base URL", value="http://217.15.175.196:11434/v1")
    model_name = st.sidebar.text_input("Ollama Model", value="llama3.2:1b")

# Config used by OpenAIWrapper(**llm_config)
llm_config = {
    "api_key": api_key or "ollama",
    "base_url": base_url or None,
    "model": model_name,
}

# Safety notice
st.sidebar.warning(
    "⚠️ Important Notice\n\n"
    "This application is a supportive tool and **does not replace** professional mental health care. "
    "If you're in crisis or having thoughts of self-harm:\n\n"
    "- Call **988** (National Crisis Hotline)\n"
    "- Call **911** (Emergency Services)\n"
    "- Seek immediate help from a mental health professional"
)

# =========================
#        MAIN UI
# =========================
st.title("🧠 CerebraTech Mental Wellbeing Agent")

st.info(
    "**Meet Your Mental Wellbeing Agent Team:**\n\n"
    "🧠 **Assessment Agent** – Analyzes your situation and emotional needs\n"
    "🎯 **Action Agent** – Creates immediate action plans and connects you with resources\n"
    "🔄 **Follow-up Agent** – Designs your long-term support strategy"
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
#      ACTION BUTTON
# =========================
if st.button("Get Support Plan"):
    # Require key only when OpenAI is selected
    if provider == "OpenAI" and not (api_key and api_key.strip()):
        st.error("Please enter your OpenAI API key.")
    else:
        with st.spinner("🤖 AI Agents are analyzing your situation..."):
            try:
                # Task for the AI agents
                task = f"""
Create a comprehensive mental health support plan based on:

Emotional State: {mental_state}
Sleep: {sleep_pattern} hours per night
Stress Level: {stress_level}/10
Support System: {', '.join(support_system) if support_system else 'None reported'}
Recent Changes: {recent_changes}
Current Symptoms: {', '.join(current_symptoms) if current_symptoms else 'None reported'}
"""

                # System messages for each agent
                system_messages = {
                    "assessment_agent": """
You are an experienced mental health professional speaking directly to the user.
Your goals:
1) Acknowledge their courage for seeking help and create a safe space.
2) Analyze their emotional state with empathy and clinical accuracy.
3) Ask targeted questions to better understand their full context.
4) Identify patterns in their thoughts, behaviors, and relationships.
5) Assess risk levels using validated screening approaches.
6) Help them understand their situation in simple language.
7) Validate their experiences without minimizing or exaggerating.

Use “you” and “your” to address the user. Be warm but professional.
""",
                    "action_agent": """
You are a crisis intervention and resource specialist speaking directly to the user.
Your goals:
1) Provide immediate, evidence-based coping strategies tailored to their situation.
2) Prioritize interventions based on urgency and effectiveness.
3) Connect them with accessible mental health resources.
4) Create a concrete daily wellness plan (specific times and activities).
5) Suggest support communities and explain how to join them.
6) Balance crisis resources with empowerment techniques.
7) Teach simple self-regulation skills they can use immediately.

Focus on realistic, doable steps that fit their current capacity.
""",
                    "followup_agent": """
You are a mental health recovery planner speaking directly to the user.
Your goals:
1) Design a personalized long-term support strategy with milestones.
2) Create a progress-tracking plan that fits their habits and preferences.
3) Develop relapse prevention techniques based on their unique triggers.
4) Map their support network and identify existing resources.
5) Build a flexible self-care routine that evolves with recovery.
6) Prepare for setbacks using self-compassion techniques.
7) Set up a regular maintenance and check-in schedule.

Focus on sustainable progress, not perfection.
""",
                }

                # Context shared among agents
                context_variables = {"assessment": None, "action": None, "followup": None}

                # Callback functions (update context and hand off to next agent)
                def update_assessment_overview(assessment_summary: str, ctx: dict) -> SwarmResult:
                    ctx["assessment"] = assessment_summary
                    st.sidebar.success("Assessment: " + assessment_summary)
                    return SwarmResult(agent="action_agent", context_variables=ctx)

                def update_action_overview(action_summary: str, ctx: dict) -> SwarmResult:
                    ctx["action"] = action_summary
                    st.sidebar.success("Action Plan: " + action_summary)
                    return SwarmResult(agent="followup_agent", context_variables=ctx)

                def update_followup_overview(followup_summary: str, ctx: dict) -> SwarmResult:
                    ctx["followup"] = followup_summary
                    st.sidebar.success("Follow-up: " + followup_summary)
                    return SwarmResult(agent="assessment_agent", context_variables=ctx)

                # System message builder for each agent
                def update_system_message_func(agent: SwarmAgent, messages) -> str:
                    system_prompt = system_messages[agent.name]
                    current_gen = agent.name.split("_")[0]

                    if agent._context_variables.get(current_gen) is None:
                        system_prompt += (
                            f" Call the update function to first provide a short 2–3 sentence summary "
                            f"of your thoughts on {current_gen.upper()} based on the given context."
                        )
                        agent.llm_config["tool_choice"] = {
                            "type": "function",
                            "function": {"name": f"update_{current_gen}_overview"},
                        }
                    else:
                        agent.llm_config["tools"] = None
                        agent.llm_config["tool_choice"] = None
                        system_prompt += (
                            f"\n\nYour task: Write only the {current_gen.capitalize()} section of the report. "
                            f"Do not include other parts or use XML tags. "
                            f"Start your response with: '## {current_gen.capitalize()} Design'."
                        )
                        k = list(agent._oai_messages.keys())[-1]
                        agent._oai_messages[k] = agent._oai_messages[k][:1]

                    # Create client depending on provider (OpenAI or Ollama)
                    agent.client = OpenAIWrapper(**agent.llm_config)
                    return system_prompt

                state_update = UPDATE_SYSTEM_MESSAGE(update_system_message_func)

                # Define the three agents
                assessment_agent = SwarmAgent(
                    "assessment_agent",
                    llm_config=llm_config,
                    functions=update_assessment_overview,
                    update_agent_state_before_reply=[state_update],
                )

                action_agent = SwarmAgent(
                    "action_agent",
                    llm_config=llm_config,
                    functions=update_action_overview,
                    update_agent_state_before_reply=[state_update],
                )

                followup_agent = SwarmAgent(
                    "followup_agent",
                    llm_config=llm_config,
                    functions=update_followup_overview,
                    update_agent_state_before_reply=[state_update],
                )

                # Hand-offs between agents
                assessment_agent.register_hand_off(AFTER_WORK(action_agent))
                action_agent.register_hand_off(AFTER_WORK(followup_agent))
                followup_agent.register_hand_off(AFTER_WORK(assessment_agent))

                # Run the swarm chat
                result, _, _ = initiate_swarm_chat(
                    initial_agent=assessment_agent,
                    agents=[assessment_agent, action_agent, followup_agent],
                    user_agent=None,
                    messages=task,
                    max_rounds=13,
                )

                # Extract results
                st.session_state.output = {
                    "assessment": result.chat_history[-3]["content"],
                    "action": result.chat_history[-2]["content"],
                    "followup": result.chat_history[-1]["content"],
                }

                with st.expander("🧾 Situation Assessment"):
                    st.markdown(st.session_state.output["assessment"])

                with st.expander("🛠️ Action Plan & Resources"):
                    st.markdown(st.session_state.output["action"])

                with st.expander("📅 Long-term Support Strategy"):
                    st.markdown(st.session_state.output["followup"])

                st.success("✨ Mental health support plan generated successfully!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
