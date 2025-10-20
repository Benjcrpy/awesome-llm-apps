FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
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

# Runtime check + launch
RUN printf '#!/usr/bin/env bash\nset -e\n'\
'echo "RUNTIME which python: $(which python)"; '\
'python -c "import sys; print(\"RUNTIME PY:\", sys.executable)"; '\
'python -m pip show streamlit || true; '\
'python - <<PY\nimport streamlit; print(\"RUNTIME STREAMLIT:\", streamlit.__version__)\nPY\n'\
'exec python -m streamlit run ai_Self-Evolving_agent.py --server.address=0.0.0.0 --server.port=${PORT:-5523} --server.enableCORS=false --server.enableXsrfProtection=false\n' \
> /app/start.sh && chmod +x /app/start.sh

ENV PORT=5523
EXPOSE 5523

ENTRYPOINT ["/app/start.sh"]
