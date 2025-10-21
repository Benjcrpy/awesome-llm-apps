# --- Dockerfile for AI Mental Wellbeing Agent ---
FROM python:3.11-slim

# System deps (optional pero useful kung may wheels na kelangan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Env & port
ENV AUTOGEN_USE_DOCKER=0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=5590
EXPOSE 5590

# Start
CMD ["bash","-lc","streamlit run ai_mental_wellbeing_agent.py --server.address ${STREAMLIT_SERVER_ADDRESS} --server.port ${STREAMLIT_SERVER_PORT}"]
