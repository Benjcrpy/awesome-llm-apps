FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python deps (lean)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Install EvoAgentX from zip WITHOUT deps to avoid huge downloads (CUDA/llama-index, etc.)
RUN python -m pip install --no-cache-dir --no-deps \
    https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# Copy app source
COPY . /app

# Ollama config (set in Coolify env ideally)
ENV OLLAMA_BASE_URL="http://217.15.175.196:11434/v1" \
    OLLAMA_MODEL="llama3.2:1b"

# Clean caches
RUN rm -rf /root/.cache

# Run headless (tkinter is stubbed in your .py)
CMD ["python", "ai_Self-Evolving_agent.py"]
