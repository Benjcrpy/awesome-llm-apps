FROM python:3.11-slim

WORKDIR /app

# tools needed to pip install from git & build wheels
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
# install project deps + make sure streamlit and evoagentx exist
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit \
    && pip install --no-cache-dir "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx"

COPY . /app

ENV PORT=5523
EXPOSE 5523

CMD ["python","-m","streamlit","run","ai_Self-Evolving_agent.py","--server.address=0.0.0.0","--server.port=5523","--server.enableCORS=false","--server.enableXsrfProtection=false"]
