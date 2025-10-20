FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    tk tcl \
    libx11-6 libxext6 libxrender1 libxft2 libfontconfig1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir streamlit==1.36.0

# Build-time check (dapat makita mo sa logs: BUILD STREAMLIT 1.36.0)
RUN python - <<'PY'
import sys
import streamlit
print("BUILD PY:", sys.executable)
print("BUILD STREAMLIT:", streamlit.__version__)
PY

RUN python -m pip install --no-cache-dir \
    "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx" \
    ddgs duckduckgo-search litellm telethon psycopg2-binary feedparser tiktoken

RUN python -m pip uninstall -y llama-index || true \
 && python -m pip install --no-cache-dir \
    "llama-index-core==0.10.63" \
    "llama-index-embeddings-azure-openai>=0.1.0,<0.2.0"

COPY . /app

# Runtime check + launch (fixed version)
RUN cat > /app/start.sh <<'BASH'
#!/usr/bin/env bash
set -e

echo "RUNTIME which python: $(which python)"
python - <<'PY'
import sys, streamlit
print("RUNTIME PY:", sys.executable)
print("RUNTIME STREAMLIT:", streamlit.__version__)
PY

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
