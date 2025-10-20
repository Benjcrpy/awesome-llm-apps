FROM python:3.11-slim

WORKDIR /app

# Base tools (git for pip+git, compilers for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (less resolver issues)
RUN python -m pip install --no-cache-dir --upgrade pip

# 1) Core deps + STREAMLIT (pin para stable)
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir streamlit==1.36.0

# Sanity check: streamlit must be importable NOW
RUN python - <<'PY'
import streamlit, sys
print("OK streamlit", streamlit.__version__)
PY

# 2) EvoAgentX + common extras the app expects
RUN python -m pip install --no-cache-dir \
    "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx" \
    ddgs duckduckgo-search litellm telethon psycopg2-binary feedparser tiktoken

# 3) LlamaIndex: remove monolith (if any) then install modular packages
RUN python -m pip uninstall -y llama-index || true \
 && python -m pip install --no-cache-dir \
    "llama-index-core==0.10.63" \
    "llama-index-embeddings-azure-openai>=0.1.0,<0.2.0"

# App code
COPY . /app

ENV PORT=5523
EXPOSE 5523

# Run via python -m to avoid PATH issues; proxy-friendly flags
CMD ["python","-m","streamlit","run","ai_Self-Evolving_agent.py","--server.address=0.0.0.0","--server.port=5523","--server.enableCORS=false","--server.enableXsrfProtection=false"]
