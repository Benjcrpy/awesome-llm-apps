FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt /app/requirements.txt

# Install deps
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt && \
    python -m pip install --no-cache-dir \
      https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# Copy source last
COPY . /app

# --- OLLAMA CONFIG (override in Coolify as env if needed) ---
ENV OLLAMA_BASE_URL="http://217.15.175.196:11434/v1" \
    OLLAMA_MODEL="llama3.2:1b" \
    OLLAMA_API_KEY="ollama"

# No port needed (worker)
CMD ["python", "ai_Self-Evolving_agent.py"]
