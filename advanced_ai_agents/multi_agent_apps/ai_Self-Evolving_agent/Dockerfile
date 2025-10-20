FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       streamlit \
       "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx" \
       ddgs duckduckgo-search litellm telethon psycopg2-binary feedparser tiktoken \
    && pip uninstall -y llama-index || true \
    && pip install --no-cache-dir \
       "llama-index-core==0.10.63" \
       "llama-index-embeddings-azure-openai>=0.1.0,<0.2.0"

COPY . /app

ENV PORT=5523
EXPOSE 5523

CMD ["python","-m","streamlit","run","ai_Self-Evolving_agent.py","--server.address=0.0.0.0","--server.port=5523","--server.enableCORS=false","--server.enableXsrfProtection=false"]
