FROM python:3.11-slim

WORKDIR /app

# Tools for building wheels + pip from git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
 && rm -rf /var/lib/apt/lists/*

# Always upgrade pip first
RUN python -m pip install --no-cache-dir --upgrade pip

# --- Install deps (PHASE 1) ---
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir streamlit==1.36.0

# Sanity check: MUST print a version here (build will fail if not)
RUN python - <<'PY'
import streamlit
print("OK streamlit", streamlit.__version__)
PY

# --- Install EvoAgentX & extras (PHASE 2) ---
RUN python -m pip install --no-cache-dir \
    "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx" \
    ddgs duckduckgo-search litellm telethon psycopg2-binary feedparser tiktoken

# LlamaIndex modules (remove monolith if present, install modular)
RUN python -m pip uninstall -y llama-index || true \
 && python -m pip install --no-cache-dir \
    "llama-index-core==0.10.63" \
    "llama-index-embeddings-azure-openai>=0.1.0,<0.2.0"

# App code after deps
COPY . /app

# Startup script to guarantee Streamlit is importable at runtime
RUN printf '#!/usr/bin/env bash\nset -e\npython - <<PY\nimport sys,streamlit; print(\"RUNTIME streamlit\", streamlit.__version__)\nPY\nexec python -m streamlit run ai_Self-Evolving_agent.py --server.address=0.0.0.0 --server.port=${PORT:-5523} --server.enableCORS=false --server.enableXsrfProtection=false\n' > /app/start.sh \
 && chmod +x /app/start.sh

ENV PORT=5523
EXPOSE 5523

# Use our start script (avoids PATH surprises)
CMD ["/app/start.sh"]
