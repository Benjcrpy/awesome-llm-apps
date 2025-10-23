FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Lean system deps; alisin agad pagkatapos
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- 1) Install CPU-only Torch FIRST (para hindi mag-pull ng CUDA/nvidia-* during deps) ---
# Official CPU wheels come from this index URL.
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu

# --- 2) Install your requirements (now sentence-transformers will reuse CPU torch) ---
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# --- 3) EvoAgentX (zip) ---
RUN python -m pip install --no-cache-dir \
      https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# --- 4) Copy source LAST ---
COPY . /app

# --- 5) Env for Ollama (set real values in Coolify env screen; safe defaults here) ---
ENV OLLAMA_BASE_URL="http://217.15.175.196:11434/v1" \
    OLLAMA_MODEL="llama3.2:1b" \
    OLLAMA_API_KEY="ollama"

# Clean up any stray caches just in case
RUN rm -rf /root/.cache

# Worker app (no exposed port)
CMD ["python", "ai_Self-Evolving_agent.py"]
