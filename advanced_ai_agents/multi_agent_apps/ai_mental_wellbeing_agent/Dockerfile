# --- Dockerfile for AI Mental Wellbeing Agent ---
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bust cache to force fresh pip install on redeploys
ARG BUILD_NOCACHE=4

# Install deps from requirements.txt (these already include autogen-agentchat, autogen-ext, pyautogen)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code (kasama ang autogen.py shim)
COPY . /app

# Make sure Python can import local modules first (no $ expansion to avoid warning)
ENV PYTHONPATH=/app
ENV AUTOGEN_USE_DOCKER=0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=5590

EXPOSE 5590

# Start Streamlit on fixed port 5590
CMD ["bash","-lc","streamlit run ai_mental_wellbeing_agent.py --server.address ${STREAMLIT_SERVER_ADDRESS} --server.port ${STREAMLIT_SERVER_PORT}"]
