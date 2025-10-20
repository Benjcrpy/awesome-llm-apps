FROM python:3.11-slim

WORKDIR /app

# Build tools + git for pip VCS installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Make sure pip is recent
RUN python -m pip install --no-cache-dir --upgrade pip

# ----- PHASE 1: Base deps + STREAMLIT -----
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir streamlit==1.36.0

# Sanity @ build-time: fail fast if streamlit missing
RUN python - <<'PY'
import sys, platform
print("BUILD PY:", sys.executable)
print("BUILD PYVER:", sys.version)
import streamlit
print("BUILD STREAMLIT:", streamlit.__version__)
PY

# ----- PHASE 2: EvoAgentX + extras -----
RUN python -m pip install --no-cache-dir \
    "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx" \
    ddgs duckduckgo-search litellm telethon psycopg2-binary feedparser tiktoken
# LlamaIndex modular packages (remove monolith if any)
RUN python -m pip uninstall -y llama-index || true \
 && python -m pip install --no-cache-dir \
    "llama-index-core==0.10.63" \
    "llama-index-embeddings-azure-openai>=0.1.0,<0.2.0"

# App code last
COPY . /app

# Start script with RUNTIME diagnostics (makikita mo sa container logs)
RUN printf '#!/usr/bin/env bash\nset -e\n\
echo \"RUNTIME which python: $(which python)\"; \
python -c \"import sys,platform; print(\\\"RUNTIME PY:\\\", sys.executable); print(\\\"RUNTIME PYVER:\\\", sys.version)\"; \
python -m pip show streamlit || true; \
python - <<PY\nimport streamlit; print(\\\"RUNTIME STREAMLIT:\\\", streamlit.__version__)\nPY\n\
exec python -m streamlit run ai_Self-Evolving_agent.py --server.address=0.0.0.0 --server.port=${PORT:-5523} --server.enableCORS=false --server.enableXsrfProtection=false\n' > /app/start.sh \
 && chmod +x /app/start.sh

ENV PORT=5523
EXPOSE 5523

# IMPORTANT: use our script so UI cannot override
ENTRYPOINT ["/app/start.sh"]
