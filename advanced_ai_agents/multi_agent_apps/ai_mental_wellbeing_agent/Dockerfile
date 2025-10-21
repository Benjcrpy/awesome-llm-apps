# --- Dockerfile for AI Mental Wellbeing Agent ---
FROM python:3.11-slim

# System deps (wheels/tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from repo
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ❗ Install AutoGen packages required by ai_mental_wellbeing_agent.py
# The code imports from `autogen` (AgentChat v0.2+ APIs like SwarmAgent, etc.)
RUN pip install --no-cache-dir \
    "autogen-agentchat>=0.2.32" \
    "autogen-ext>=0.2.32" \
    "openai>=1.40.0"

# Copy app code
COPY . /app

# Runtime env
ENV AUTOGEN_USE_DOCKER=0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=5590
EXPOSE 5590

# Start Streamlit on fixed port 5590
CMD ["bash","-lc","streamlit run ai_mental_wellbeing_agent.py --server.address ${STREAMLIT_SERVER_ADDRESS} --server.port ${STREAMLIT_SERVER_PORT}"]
