import os
import json
import uuid
from datetime import datetime
import streamlit as st
from autogen import OpenAIWrapper  # uses your compat wrapper (OpenAI or Ollama)

# -----------------------------
#   APP / SESSION SETUP
# -----------------------------
os.environ["AUTOGEN_USE_DOCKER"] = "0"

if "output" not in st.session_state:
    st.session_state.output = {"assessment": "", "action": "", "followup": ""}

# ---- History storage paths ----
DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

def _read_history() -> list:
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _write_history(entries: list) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

def add_history_entry(entry: dict) -> None:
    entries = _read_history()
    entries.insert(0, entry)  # latest first
    _write_history(entries)

# -----------------------------
#   SIDEBAR (Settings)
# -----------------------------
st.sidebar.title("⚙️ Settings")

# CerebraTech branding + link
st.sidebar.markdown(
    "<h3>🌐 <a href='https://cerebratech.xyz/' target='_blank' style='text-decoration:none;'>CerebraTech Website</a></h3>",
    unsafe_allow_html=True,
)
st.sidebar.divider()

provider = st.sidebar.selectbox(
    "Choose an LLM Provider",
    ["-- Select Provider --", "Ollama (no key required)", "OpenAI"],
    index=0,
    key="provider_select",
)

api_key = None
base_url = None
model_name = None

if provider == "OpenAI":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="openai_key")
    base_url = st.sidebar.text_input(
        "Optional: OpenAI Base URL",
        value="",
        placeholder="leave blank for api.openai.com",
        key="openai_base",
    ).strip() or None
    model_name = st.sidebar.text_input("Model", value="gpt-4o-mini", key="openai_model").strip()

elif provider == "Ollama (no key required)":
    st.sidebar.info("Using **Ollama** — no API key required.")
    base_url = st.sidebar.text_input(
        "Ollama Base URL", value="http://217.15.175.196:11434/v1", key="ollama_base"
    ).strip()
    model_name = st.sidebar.text_input("Ollama Model", value="llama3.2:1b", key="ollama_model").strip()

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

# -----------------------------
#   MAIN UI
# -----------------------------
st.title("🧠 CerebraTech Mental Wellbeing Agent")

st.info(
    "**Meet Your Mental Wellbeing Agent Team:**\n\n"
    "🧠 **Assessment Agent** – Evaluates your emotional and psychological state\n"
    "🎯 **Action Agent** – Builds an immediate action plan for support\n"
    "🔄 **Follow-up Agent** – Creates a long-term mental health strategy"
)

st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    mental_state = st.text_area(
        "How have you been feeling recently?",
        placeholder="Describe your emotions, thoughts, or concerns...",
        key="mental_state",
    )
    sleep_pattern = st.select_slider(
        "Sleep Pattern (hours per night)",
        options=[f"{i}" for i in range(0, 13)],
        value="7",
        key="sleep_pattern",
    )

with col2:
    stress_level = st.slider("Current Stress Level (1–10)", 1, 10, 5, key="stress_level")
    support_system = st.multiselect(
        "Current Support System",
        ["Family", "Friends", "Therapist", "Support Groups", "None"],
        key="support_system",
    )

recent_changes = st.text_area(
    "Any significant life changes or events recently?",
    placeholder="Job changes, relationships, loss, etc.",
    key="recent_changes",
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
    key="current_symptoms",
)

# -----------------------------
#   VALIDATION
# -----------------------------
def validate_fields() -> list[str]:
    missing = []
    if provider == "-- Select Provider --":
        missing.append("Please select an LLM provider.")
    if provider == "OpenAI" and not api_key:
        missing.append("OpenAI API key is required for OpenAI mode.")
    if provider == "Ollama (no key required)":
        if not base_url or not model_name:
            missing.append("Ollama Base URL and Model are required in Ollama mode.")
    if not mental_state or not mental_state.strip():
        missing.append("Please describe how you've been feeling.")
    if not recent_changes or not recent_changes.strip():
        missing.append("Please describe any recent life changes.")
    if not current_symptoms:
        missing.append("Please select at least one symptom.")
    return missing


# -----------------------------
#   LLM HELPERS (chain-of-3)
# -----------------------------
def build_llm_cfg() -> dict:
    """
    Uses your autogen.OpenAIWrapper. Works for both OpenAI and Ollama.
    """
    cfg = {"api_key": api_key or "ollama"}
    if base_url:
        cfg["base_url"] = base_url
    if model_name:
        cfg["model"] = model_name
    return cfg


def llm_chat(system_prompt: str, user_prompt: str, llm_cfg: dict) -> str:
    client = OpenAIWrapper(**llm_cfg)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return client.chat(msgs).strip()


def user_snapshot() -> str:
    return (
        "USER SNAPSHOT\n"
        f"- Emotional State: {mental_state.strip()}\n"
        f"- Sleep: {sleep_pattern} hours/night\n"
        f"- Stress Level: {stress_level}/10\n"
        f"- Support System: {', '.join(support_system) if support_system else 'None reported'}\n"
        f"- Recent Changes: {recent_changes.strip()}\n"
        f"- Current Symptoms: {', '.join(current_symptoms)}\n"
    )


ASSESSMENT_SYS = (
    "You are a licensed mental health professional. Analyze the USER SNAPSHOT.\n"
    "Deliver a brief, empathetic clinical assessment (180–280 words) that:\n"
    "• Identifies key themes and likely drivers (stressors, patterns, lifestyle factors)\n"
    "• Notes risk, but avoids diagnosis\n"
    "• Uses plain language with warmth and validation\n"
    "• Avoids repeating the inputs verbatim; synthesize them instead\n"
    "Format exactly:\n"
    "## Assessment Summary\n"
    "• Key patterns & concerns\n"
    "• Likely contributing factors\n"
    "• Immediate considerations (no crisis resources unless clearly needed)"
)

ACTION_SYS = (
    "You are a crisis-intervention and skills coach. Create a **practical** 1–3 day plan (220–320 words)\n"
    "based on the USER SNAPSHOT and the ASSESSMENT SUMMARY. Include:\n"
    "• 3–5 concrete coping strategies (step-by-step)\n"
    "• A simple day schedule with times (morning/afternoon/evening)\n"
    "• Social/support actions (what to say/how to ask)\n"
    "• ‘If overwhelmed’ mini-plan (60–120 seconds actions)\n"
    "Be specific, realistic, and non-parroting. Use bullets.\n"
    "Start with the heading: ## Action Plan"
)

FOLLOWUP_SYS = (
    "You are a mental health recovery planner. Create a **4–6 week** sustainable plan (200–300 words)\n"
    "using the USER SNAPSHOT, ASSESSMENT SUMMARY, and ACTION PLAN. Include:\n"
    "• Weekly rhythm (habits, exercise/sleep hygiene, reflection prompts)\n"
    "• Progress markers (how to know it's improving)\n"
    "• Relapse-prevention checklist\n"
    "• Gentle self-compassion framing\n"
    "No medical diagnosis. Avoid repeating inputs verbatim.\n"
    "Start with the heading: ## Long-term Strategy"
)

# -----------------------------
#   MAIN ACTION
# -----------------------------
if st.button("Get Support Plan"):
    errors = validate_fields()
    if errors:
        for err in errors:
            st.error(err)
    else:
        with st.spinner("🤖 Generating your personalized support plan..."):
            try:
                llm_cfg = build_llm_cfg()
                snapshot = user_snapshot()

                # 1) Assessment
                assessment_text = llm_chat(
                    ASSESSMENT_SYS,
                    snapshot,
                    llm_cfg,
                )

                # 2) Action Plan
                action_prompt = (
                    snapshot
                    + "\n\nASSESSMENT SUMMARY:\n"
                    + assessment_text
                    + "\n\nCreate the action plan now."
                )
                action_text = llm_chat(
                    ACTION_SYS,
                    action_prompt,
                    llm_cfg,
                )

                # 3) Long-term Strategy
                followup_prompt = (
                    snapshot
                    + "\n\nASSESSMENT SUMMARY:\n"
                    + assessment_text
                    + "\n\nACTION PLAN:\n"
                    + action_text
                    + "\n\nCreate the long-term strategy now."
                )
                followup_text = llm_chat(
                    FOLLOWUP_SYS,
                    followup_prompt,
                    llm_cfg,
                )

                # Save to session
                st.session_state.output = {
                    "assessment": assessment_text,
                    "action": action_text,
                    "followup": followup_text,
                }

                # -----------------------------
                #   SAVE TO HISTORY (auto)
                # -----------------------------
                add_history_entry(
                    {
                        "id": str(uuid.uuid4()),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "provider": provider,
                        "model": model_name,
                        "inputs": {
                            "mental_state": mental_state,
                            "sleep_pattern": sleep_pattern,
                            "stress_level": stress_level,
                            "support_system": support_system,
                            "recent_changes": recent_changes,
                            "current_symptoms": current_symptoms,
                        },
                        "outputs": st.session_state.output,
                    }
                )

                # Show results
                with st.expander("🧾 Assessment Summary", expanded=False):
                    st.markdown(st.session_state.output["assessment"])
                with st.expander("🛠️ Action Plan", expanded=False):
                    st.markdown(st.session_state.output["action"])
                with st.expander("📅 Long-term Strategy", expanded=False):
                    st.markdown(st.session_state.output["followup"])

                st.success("✅ Personalized mental health plan generated and saved to history!")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# -----------------------------
#   HISTORY VIEWER
# -----------------------------
st.markdown("## 📜 History")
history_entries = _read_history()

col_h1, col_h2 = st.columns([1, 1])
with col_h1:
    st.caption(f"Saved entries: **{len(history_entries)}**")
with col_h2:
    if st.button("🗑️ Clear History"):
        _write_history([])
        st.experimental_rerun()

if history_entries:
    # Limit to first 10 for display
    for idx, item in enumerate(history_entries[:10], start=1):
        title = f"[{item.get('created_at','')}] • {item.get('provider','?')} • {item.get('model','?')}"
        with st.expander(f"{idx}. {title}", expanded=False):
            st.markdown("**Inputs**")
            inp = item.get("inputs", {})
            st.write(
                {
                    "Emotional State": inp.get("mental_state", ""),
                    "Sleep (hrs)": inp.get("sleep_pattern", ""),
                    "Stress Level": inp.get("stress_level", ""),
                    "Support System": inp.get("support_system", []),
                    "Recent Changes": inp.get("recent_changes", ""),
                    "Symptoms": inp.get("current_symptoms", []),
                }
            )
            st.markdown("---")
            st.markdown("**Outputs**")
            outs = item.get("outputs", {})
            with st.expander("🧾 Assessment", expanded=False):
                st.markdown(outs.get("assessment", ""))
            with st.expander("🛠️ Action Plan", expanded=False):
                st.markdown(outs.get("action", ""))
            with st.expander("📅 Long-term Strategy", expanded=False):
                st.markdown(outs.get("followup", ""))

else:
    st.info("No history yet — generate a plan to see it here.")

# -----------------------------
#   FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 15px;'>
        Powered by <b>CerebraTech</b> | Modified by <b>CerebraTech</b><br>
        <a href='https://cerebratech.xyz/' target='_blank' style='text-decoration: none; color: #1E88E5;'>
            🌐 CerebraTech
        </a>
    </p>
    """,
    unsafe_allow_html=True,
)
