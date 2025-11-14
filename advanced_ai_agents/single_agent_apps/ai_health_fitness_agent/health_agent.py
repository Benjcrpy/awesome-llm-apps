import re
import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.google import Gemini
from agno.models.openai.like import OpenAILike

# ---------- PAGE CONFIG & BASIC STYLES ----------

st.set_page_config(
    page_title="AI Health & Fitness Planner",
    page_icon="️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple custom styling
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
.app-card {
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(250, 250, 250, 0.06);
    background: rgba(15, 15, 15, 0.85);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.stButton > button {
    border-radius: 999px;
    font-weight: 600;
    font-size: 1rem;
}
.qa-card {
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(250, 250, 250, 0.08);
    background: rgba(25, 25, 25, 0.9);
    margin-bottom: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- DISPLAY HELPERS ----------


def display_dietary_plan(plan_content):
    with st.expander("🍽️ Your Personalized Dietary Plan", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### ✅ Why this plan works")
            st.info(plan_content.get("why_this_plan_works", "Information not available"))

            st.markdown("### 🧾 Meal Plan")
            st.write(plan_content.get("meal_plan", "Plan not available"))

        with col2:
            st.markdown("### ⚠️ Important Considerations")
            considerations = plan_content.get("important_considerations", "").split("\n")
            for consideration in considerations:
                if consideration.strip():
                    st.warning(consideration)


def display_fitness_plan(plan_content):
    with st.expander("🏋️ Your Personalized Fitness Plan", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 🎯 Goals")
            st.success(plan_content.get("goals", "Goals not specified"))

            st.markdown("### 🏃‍♂️ Exercise Routine")
            st.write(plan_content.get("routine", "Routine not available"))

        with col2:
            st.markdown("### 💡 Pro Tips")
            tips = plan_content.get("tips", "").split("\n")
            for tip in tips:
                if tip.strip():
                    st.info(tip)


# ---------- MAIN APP ----------


def main():
    # ----- Session state -----
    if "dietary_plan" not in st.session_state:
        st.session_state.dietary_plan = {}
    if "fitness_plan" not in st.session_state:
        st.session_state.fitness_plan = {}
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "plans_generated" not in st.session_state:
        st.session_state.plans_generated = False

    # ----- Header -----
    st.markdown("## ♂️ AI Health & Fitness Planner")
    st.markdown(
        "Get personalized dietary and fitness plans tailored to your goals, "
        "body metrics, available equipment, and dietary restrictions."
    )
    st.markdown("")

    # =========================
    #           SIDEBAR
    # =========================
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("[🌐 CerebraTech Website](https://cerebratech.xyz)")
        st.markdown("---")

        provider = st.selectbox(
            "Choose an LLM Provider",
            options=["Ollama (no key required)", "Gemini"],
        )

        active_model = None

        if provider.startswith("Ollama"):
            st.info("Using Ollama — no API key required.")

            ollama_base_url = st.text_input(
                "Ollama Base URL",
                value="http://217.15.175.196:11434/v1",
            ).strip()

            ollama_model_name = st.text_input(
                "Ollama Model",
                value="llama3.2:1b",
            ).strip()

            if not ollama_base_url or not ollama_model_name:
                st.error("Please fill in both Ollama Base URL and Ollama Model.")
                return

            active_model = OpenAILike(
                id=ollama_model_name,
                base_url=ollama_base_url,
                api_key="ollama-no-key-required",
            )

        else:
            st.subheader("Gemini Settings")

            gemini_api_key = st.text_input(
                "Enter your Gemini API Key",
                type="password",
                help="This key is used to call Google Gemini models.",
            ).strip()

            st.text_input(
                "Optional: Gemini Base URL",
                placeholder="leave blank for default Gemini endpoint",
                help="Most users can leave this empty.",
            )

            if not gemini_api_key:
                st.warning(
                    "Please enter your Gemini API Key, or switch back to Ollama above."
                )
                return

            try:
                active_model = Gemini(
                    id="gemini-2.0-flash",
                    api_key=gemini_api_key,
                )
                st.success("Gemini configured successfully. ✅")
            except Exception as e:
                st.error(f"❌ Error initializing Gemini model: {e}")
                return

    # =========================
    #       MAIN FORM
    # =========================
    with st.form("profile_form"):
        # ---- Profile Card ----
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧍 Your Profile</div>', unsafe_allow_html=True)

        col_profile_left, col_profile_right = st.columns(2)

        # defaults to avoid reference before assignment
        height_cm = 0.0
        height_text = ""

        with col_profile_left:
            age = st.number_input(
                "Age *",
                min_value=10,
                max_value=100,
                step=1,
            )

            st.markdown("#### Height *")
            height_unit = st.radio(
                "Height Unit",
                options=["cm", "ft/in"],
                horizontal=True,
                key="height_unit_radio",
                label_visibility="collapsed",
            )

            if height_unit == "cm":
                height_cm = st.number_input(
                    "Height (cm)",
                    min_value=100.0,
                    max_value=250.0,
                    step=0.1,
                    value=170.0,
                )
                height_text = f"{height_cm:.1f} cm"
            else:
                height_raw = st.text_input(
                    "Height (ft/in)",
                    placeholder='e.g. 5\'6" or 5 ft 6 in',
                    help="You can type formats like 5'6, 5'6\", 5 ft 6 in, or 5 6.",
                )
                height_raw_stripped = height_raw.strip()
                if height_raw_stripped:
                    numbers = re.findall(r"\\d+\\.?\\d*", height_raw_stripped)
                    if numbers:
                        feet = float(numbers[0])
                        inches = float(numbers[1]) if len(numbers) >= 2 else 0.0
                        total_inches = feet * 12.0 + inches
                        height_cm = total_inches * 2.54
                        height_text = (
                            f"{feet:g} ft {inches:g} in (~{height_cm:.1f} cm)"
                        )
                    else:
                        height_cm = 0.0
                        height_text = height_raw_stripped
                else:
                    height_cm = 0.0
                    height_text = ""

            activity_level = st.selectbox(
                "Activity Level *",
                options=[
                    "Sedentary",
                    "Lightly Active",
                    "Moderately Active",
                    "Very Active",
                    "Extremely Active",
                ],
            )

        with col_profile_right:
            st.markdown("#### Weight *")
            weight_unit = st.radio(
                "Weight Unit",
                options=["kg", "lbs"],
                horizontal=True,
                key="weight_unit_radio",
                label_visibility="collapsed",
            )

            if weight_unit == "kg":
                weight_kg = st.number_input(
                    "Weight (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    step=0.1,
                    value=70.0,
                )
                weight_text = f"{weight_kg:.1f} kg"
            else:
                weight_lbs = st.number_input(
                    "Weight (lbs)",
                    min_value=44.0,
                    max_value=660.0,
                    step=0.5,
                    value=154.0,
                )
                weight_kg = weight_lbs * 0.45359237
                weight_text = f"{weight_lbs:.1f} lbs (~{weight_kg:.1f} kg)"

            sex = st.selectbox("Sex *", options=["Male", "Female", "Other"])

            fitness_goals = st.selectbox(
                "Fitness Goals *",
                options=[
                    "Lose Weight",
                    "Gain Muscle",
                    "Endurance",
                    "Stay Fit",
                    "Strength Training",
                ],
            )

        st.markdown("</div>", unsafe_allow_html=True)  # close card
        st.markdown("")

        # ---- Dietary + Equipment Card ----
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🥗 Dietary Preferences & Equipment</div>',
            unsafe_allow_html=True,
        )

        dietary_col, equipment_col = st.columns([3, 2])

        with dietary_col:
            st.markdown("**Dietary Restrictions / Preferences * **")
            st.caption("Select at least one option.")

            dietary_options = [
                "No Red Meat",
                "No Pork",
                "No Chicken",
                "No Seafood",
                "Dairy Free",
                "Gluten Free",
                "Nut Free",
                "Egg Free",
                "Vegetarian",
                "Vegan",
                "Pescatarian",
                "Halal",
                "Kosher",
                "Low Carb",
                "Low Fat",
            ]

            col_a, col_b = st.columns(2)
            dietary_restrictions = []

            for i, opt in enumerate(dietary_options):
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    if st.checkbox(opt, key=f"diet_{opt.replace(' ', '_').lower()}"):
                        dietary_restrictions.append(opt)

        with equipment_col:
            equipment = st.text_area(
                "Available Equipment / Tools *",
                placeholder=(
                    "Example: bodyweight only, pull-up bar, adjustable dumbbells, "
                    "resistance bands, kettlebells, treadmill, air fryer, oven, etc."
                ),
                help="The workout and meal plan will adapt to the equipment you list here.",
                height=140,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # close card
        st.markdown("")
        st.markdown("---")

        submitted = st.form_submit_button(
            "✨ Generate My Personalized Plan", use_container_width=True
        )

    # ----- Validation & Plan Generation -----
    if submitted:
        errors = []

        if age is None:
            errors.append("Please enter your age.")
        if height_cm <= 0:
            errors.append(
                "Please enter a valid height (cm or a valid feet/inches format like 5'6\")."
            )
        if weight_kg <= 0:
            errors.append("Please enter a valid weight.")
        if not activity_level:
            errors.append("Please select your activity level.")
        if not sex:
            errors.append("Please select your sex.")
        if not fitness_goals:
            errors.append("Please select your fitness goal.")
        if not dietary_restrictions:
            errors.append("Please select at least one dietary restriction / preference.")
        if not equipment or not equipment.strip():
            errors.append("Please list at least one item in Available Equipment / Tools.")

        if errors:
            for msg in errors:
                st.error(msg)
            st.stop()

        dietary_restrictions_text = ", ".join(dietary_restrictions)

        with st.spinner("Creating your personalized health and fitness routine..."):
            try:
                dietary_agent = Agent(
                    name="Dietary Expert",
                    role="Provides personalized dietary recommendations.",
                    model=active_model,
                    instructions=[
                        "Consider the user's full profile, including age, sex, weight, "
                        "height, activity level, fitness goals, dietary restrictions, "
                        "and available equipment/tools.",
                        "Respect ALL dietary restrictions and preferences listed by the user. "
                        "Do not include any ingredient that violates those restrictions.",
                        "Take into account the user's available kitchen tools and equipment "
                        "when suggesting meals and preparation methods.",
                        "Suggest a detailed meal plan for the day, including breakfast, "
                        "lunch, dinner, and snacks.",
                        "Provide a clear explanation of why the plan is suited to the user's goals.",
                        "Focus on clarity, coherence, and practicality.",
                    ],
                )

                fitness_agent = Agent(
                    name="Fitness Expert",
                    role="Provides personalized fitness recommendations.",
                    model=active_model,
                    instructions=[
                        "Provide exercises tailored to the user's goals and fitness level.",
                        "ONLY prescribe exercises that can realistically be performed with the "
                        "user's available equipment and bodyweight. Avoid suggesting machines or "
                        "tools they did not list.",
                        "If the user has no equipment specified, focus on bodyweight-friendly "
                        "exercises only.",
                        "Include warm-up, main workout, and cool-down exercises.",
                        "Explain the benefits of each recommended exercise.",
                        "Ensure the plan is actionable, detailed, and safe for the given profile.",
                    ],
                )

                user_profile = f"""
User Profile:
- Age: {age}
- Sex: {sex}
- Weight: {weight_text} (≈ {weight_kg:.1f} kg)
- Height: {height_text}
- Activity Level: {activity_level}
- Fitness Goals: {fitness_goals}
- Dietary Restrictions / Preferences: {dietary_restrictions_text}
- Available Equipment / Tools: {equipment.strip()}
"""

                dietary_plan_response: RunOutput = dietary_agent.run(user_profile)
                dietary_plan = {
                    "why_this_plan_works": (
                        "This plan is tailored to the user's goals, metrics, dietary "
                        "restrictions, and available tools."
                    ),
                    "meal_plan": dietary_plan_response.content,
                    "important_considerations": """
- Hydration: Drink plenty of water throughout the day.
- Electrolytes: Monitor sodium, potassium, and magnesium levels.
- Fiber: Ensure adequate intake through vegetables and fruits (within restrictions).
- Listen to your body: Adjust portion sizes and food choices as needed.
""",
                }

                fitness_plan_response: RunOutput = fitness_agent.run(user_profile)
                fitness_plan = {
                    "goals": (
                        "Build strength, improve endurance, and maintain overall fitness "
                        "using the user's available equipment."
                    ),
                    "routine": fitness_plan_response.content,
                    "tips": """
- Track your progress regularly.
- Allow proper rest between workouts.
- Focus on proper form, especially with weights.
- Stay consistent with your routine.
""",
                }

                st.session_state.dietary_plan = dietary_plan
                st.session_state.fitness_plan = fitness_plan
                st.session_state.plans_generated = True
                st.session_state.qa_pairs = []

                display_dietary_plan(dietary_plan)
                display_fitness_plan(fitness_plan)

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.stop()

    # =========================
    #            Q&A
    # =========================
    if st.session_state.plans_generated:
        st.markdown("### ❓ Questions about your plan?")
        question_input = st.text_input("Ask a question about your plan (required to submit):")

        if st.button("💬 Get Answer"):
            if not question_input or not question_input.strip():
                st.error("Please type a question before requesting an answer.")
            else:
                with st.spinner("Generating the best answer for you..."):
                    dietary_plan = st.session_state.dietary_plan
                    fitness_plan = st.session_state.fitness_plan

                    context = (
                        f"Dietary Plan: {dietary_plan.get('meal_plan', '')}\n\n"
                        f"Fitness Plan: {fitness_plan.get('routine', '')}"
                    )
                    full_context = f"{context}\n\nUser Question: {question_input.strip()}"

                    try:
                        qa_agent = Agent(
                            model=OpenAILike(
                                id="llama3.2:1b",
                                base_url="http://217.15.175.196:11434/v1",
                                api_key="ollama-no-key-required",
                            )
                            if provider.startswith("Ollama")
                            else active_model,
                            debug_mode=True,
                            markdown=True,
                        )
                        run_response: RunOutput = qa_agent.run(full_context)

                        if hasattr(run_response, "content"):
                            answer = run_response.content
                        else:
                            answer = "Sorry, I couldn't generate a response at this time."

                        st.session_state.qa_pairs.append((question_input, answer))
                    except Exception as e:
                        st.error(f"❌ An error occurred while getting the answer: {e}")

    if st.session_state.qa_pairs:
        st.markdown("### 📚 Q&A History")
        for question, answer in st.session_state.qa_pairs:
            st.markdown(
                f'<div class="qa-card"><b>Q:</b> {question}<br/><b>A:</b> {answer}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
