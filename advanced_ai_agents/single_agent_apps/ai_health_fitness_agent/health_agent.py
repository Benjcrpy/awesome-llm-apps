import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.google import Gemini
from agno.models.openai.like import OpenAILike


st.set_page_config(
    page_title="AI Health & Fitness Planner",
    page_icon="️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def main():
    # Session state initialization
    if "dietary_plan" not in st.session_state:
        st.session_state.dietary_plan = {}
    if "fitness_plan" not in st.session_state:
        st.session_state.fitness_plan = {}
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "plans_generated" not in st.session_state:
        st.session_state.plans_generated = False

    st.title("️‍♂️ AI Health & Fitness Planner")
    st.markdown(
        """
Get personalized dietary and fitness plans tailored to your goals, body metrics,
available equipment, and dietary restrictions.
""",
        unsafe_allow_html=True,
    )

    # =========================
    #       SETTINGS / LLM
    # =========================
    with st.sidebar:
        st.header("⚙️ Settings")

        # (Optional) link to your site or docs
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
            )

            ollama_model_name = st.text_input(
                "Ollama Model",
                value="llama3.2:1b",
            )

            # OpenAI-compatible client pointing to Ollama
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
            )

            gemini_base_url = st.text_input(
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
                # We ignore gemini_base_url because the Gemini model
                # in agno handles its own endpoint configuration.
                active_model = Gemini(
                    id="gemini-2.0-flash",
                    api_key=gemini_api_key,
                )
                st.success("Gemini configured successfully. ✅")
            except Exception as e:
                st.error(f"❌ Error initializing Gemini model: {e}")
                return

    # =========================
    #       USER PROFILE
    # =========================
    st.header("🧍‍♂️ Your Profile")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            step=1,
            help="Enter your age.",
        )

        # HEIGHT with unit toggle
        st.markdown("#### Height")
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
            height_ft = st.number_input(
                "Height (ft)",
                min_value=3,
                max_value=8,
                value=5,
            )
            height_in = st.number_input(
                "Additional inches",
                min_value=0.0,
                max_value=11.9,
                step=0.1,
                value=7.0,
            )
            total_inches = height_ft * 12 + height_in
            height_cm = total_inches * 2.54
            height_text = f"{height_ft} ft {height_in:.1f} in (~{height_cm:.1f} cm)"

        activity_level = st.selectbox(
            "Activity Level",
            options=[
                "Sedentary",
                "Lightly Active",
                "Moderately Active",
                "Very Active",
                "Extremely Active",
            ],
            help="Choose your typical activity level.",
        )

        # DIETARY RESTRICTIONS as checkboxes
        st.markdown("#### Dietary Restrictions / Preferences")
        st.caption("You can select one or multiple options.")
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
        dietary_restrictions = []
        for opt in dietary_options:
            if st.checkbox(opt, key=f"diet_{opt.replace(' ', '_').lower()}"):
                dietary_restrictions.append(opt)

        # Equipment / tools
        equipment = st.text_area(
            "Available Equipment / Tools",
            placeholder=(
                "Example: bodyweight only, adjustable dumbbells, barbell, resistance bands, "
                "treadmill, stationary bike, pull-up bar, kettlebells, air fryer, oven, "
                "microwave, blender, meal-prep containers, etc."
            ),
            help=(
                "List all workout and kitchen tools you have. "
                "The plan will be adapted to the tools you actually own."
            ),
        )

    with col2:
        # WEIGHT with unit toggle
        st.markdown("#### Weight")
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

        sex = st.selectbox("Sex", options=["Male", "Female", "Other"])

        fitness_goals = st.selectbox(
            "Fitness Goals",
            options=[
                "Lose Weight",
                "Gain Muscle",
                "Endurance",
                "Stay Fit",
                "Strength Training",
            ],
            help="What do you want to achieve?",
        )

    dietary_restrictions_text = (
        ", ".join(dietary_restrictions) if dietary_restrictions else "None specified"
    )

    if st.button("✨ Generate My Personalized Plan", use_container_width=True):
        with st.spinner("Creating your personalized health and fitness routine..."):
            try:
                # Dietary agent
                dietary_agent = Agent(
                    name="Dietary Expert",
                    role="Provides personalized dietary recommendations.",
                    model=active_model,
                    instructions=[
                        "Consider the user's full profile, including age, sex, weight, "
                        "height, activity level, fitness goals, dietary restrictions, "
                        "and available equipment/tools.",
                        "Respect ALL dietary restrictions and preferences listed by the user. "
                        "Do not include any ingredient that violates those restrictions "
                        "(e.g., no red meat if the user selected 'No Red Meat').",
                        "Take into account the user's available kitchen tools and equipment "
                        "(e.g., air fryer, oven, microwave, blender, meal-prep containers) "
                        "when suggesting meals and preparation methods.",
                        "Suggest a detailed meal plan for the day, including breakfast, "
                        "lunch, dinner, and snacks.",
                        "Provide a clear explanation of why the plan is suited to the user's goals.",
                        "Focus on clarity, coherence, and practicality.",
                    ],
                )

                # Fitness agent
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

                # User profile passed to agents
                user_profile = f"""
User Profile:
- Age: {age}
- Sex: {sex}
- Weight: {weight_text} (≈ {weight_kg:.1f} kg)
- Height: {height_text}
- Activity Level: {activity_level}
- Fitness Goals: {fitness_goals}
- Dietary Restrictions / Preferences: {dietary_restrictions_text}
- Available Equipment / Tools: {equipment or "User did not specify any equipment (bodyweight only)."}
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

    # Q&A section
    if st.session_state.plans_generated:
        st.header("❓ Questions about your plan?")

        question_input = st.text_input("Ask a question about your plan:")

        if st.button("💬 Get Answer"):
            if question_input:
                with st.spinner("Generating the best answer for you..."):
                    dietary_plan = st.session_state.dietary_plan
                    fitness_plan = st.session_state.fitness_plan

                    context = (
                        f"Dietary Plan: {dietary_plan.get('meal_plan', '')}\n\n"
                        f"Fitness Plan: {fitness_plan.get('routine', '')}"
                    )
                    full_context = f"{context}\n\nUser Question: {question_input}"

                    try:
                        qa_agent = Agent(
                            model=active_model,
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
                        st.error(
                            f"❌ An error occurred while getting the answer: {e}"
                        )

    # Q&A history
    if st.session_state.qa_pairs:
        st.header("📚 Q&A History")
        for question, answer in st.session_state.qa_pairs:
            st.markdown(f"**Q:** {question}")
            st.markdown(f"**A:** {answer}")


if __name__ == "__main__":
    main()
