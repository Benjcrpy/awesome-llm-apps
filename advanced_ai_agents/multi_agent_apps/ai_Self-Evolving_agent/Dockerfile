FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal OS deps (keep it lean para iwas "No space left on device")
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install only the lean runtime deps (huwag 'yung mabibigat)
COPY requirements.txt /app/
RUN python -m pip install -r requirements.txt

# EvoAgentX (exactly as you used, pero iisa lang na extra line)
RUN python -m pip install "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx"

# Copy source last for better caching
COPY . /app

# Runtime check + launch
RUN cat > /app/start.sh <<'BASH'
#!/usr/bin/env bash
set -e

echo "RUNTIME which python: $(which python)"
python - <<'PY'
import sys, streamlit
print("RUNTIME PY:", sys.executable)
print("RUNTIME STREAMLIT:", streamlit.__version__)
PY

# IMPORTANT:
# Streamlit port/address; Streamlit UI only (wala nang OpenAI client dito)
exec python -m streamlit run ai_Self-Evolving_agent.py \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-5523} \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
BASH

RUN chmod +x /app/start.sh

ENV PORT=5523
EXPOSE 5523

ENTRYPOINT ["/app/start.sh"]
