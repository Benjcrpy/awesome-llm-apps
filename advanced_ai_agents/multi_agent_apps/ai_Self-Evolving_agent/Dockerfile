FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System deps incl. Tk (for tkinter) + xvfb + xauth (for headless GUI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    python3-tk tk tcl xvfb xauth \
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

# Ollama config (better: set in Coolify env screen)
ENV OLLAMA_BASE_URL="http://217.15.175.196:11434/v1" \
    OLLAMA_MODEL="llama3.2:1b" \
    OLLAMA_API_KEY="ollama"

# Clean caches
RUN rm -rf /root/.cache

# Headless run (Tk needs X)
CMD ["bash", "-lc", "xvfb-run -a -s '-screen 0 1024x768x24' python ai_Self-Evolving_agent.py"]
