# --- Dockerfile for AI Mental Wellbeing Agent ---
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bust cache para siguradong re-run ang install
ARG BUILD_NOCACHE=3

# Install deps from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# (Optional) keep these; ok lang kung nasa requirements na
RUN pip install --no-cache-dir --upgrade \
    pyautogen \
    autogen-agentchat==0.2.36 \
    autogen-ext==0.2.36 \
    "openai>=1.40.0"

# Copy app code (kasama ang autogen.py shim)
COPY . /app

# 👉 SIGURADUHIN na mauuna mahanap ang /app kapag nag-iimport ng modules
ENV PYTHONPATH=/app:${PYTHONPATH}

ENV AUTOGEN_USE_DOCKER=0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=5590
EXPOSE 5590

CMD ["bash","-lc","streamlit run ai_mental_wellbeing_agent.py --server.address ${STREAMLIT_SERVER_ADDRESS} --server.port ${STREAMLIT_SERVER_PORT}"]
