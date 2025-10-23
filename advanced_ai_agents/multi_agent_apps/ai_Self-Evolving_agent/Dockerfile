FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Lean system deps + Tk + Xvfb for headless GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    python3-tk tk tcl xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

# 1) CPU-only Torch (avoid CUDA blobs)
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu

# 2) Project deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# 3) EvoAgentX lib
RUN python -m pip install --no-cache-dir \
      https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# 4) Copy source
COPY . /app

# Ollama config (set real values in Coolify env if you prefer)
ENV OLLAMA_BASE_URL="http://217.15.175.196:11434/v1" \
    OLLAMA_MODEL="llama3.2:1b" \
    OLLAMA_API_KEY="ollama"

# Clean caches
RUN rm -rf /root/.cache

# Run under Xvfb so Tkinter won't crash in headless
CMD ["bash", "-lc", "xvfb-run -a python ai_Self-Evolving_agent.py"]
