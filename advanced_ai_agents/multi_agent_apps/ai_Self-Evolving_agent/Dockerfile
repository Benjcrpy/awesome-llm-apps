FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Lean system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Project deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# EvoAgentX lib
RUN python -m pip install --no-cache-dir \
      https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# App source
COPY . /app

# Ollama config (tip: set secrets in Coolify env, not baked here)
ENV OLLAMA_BASE_URL="http://217.15.175.196:11434/v1" \
    OLLAMA_MODEL="llama3.2:1b"

# Clean caches
RUN rm -rf /root/.cache

# Worker app (no GUI, no X needed)
CMD ["python", "ai_Self-Evolving_agent.py"]
