FROM python:3.11-slim

WORKDIR /app

# optional build tools (safe)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
# Install deps + make sure streamlit & EvoAgentX exist
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit && \
    pip install --no-cache-dir "git+https://github.com/EvoAgentX/EvoAgentX.git@main#egg=evoagentx"

COPY . /app

ENV PORT=5523
EXPOSE 5523

CMD ["python","-m","streamlit","run","ai_Self-Evolving_agent.py",
     "--server.address","0.0.0.0","--server.port","5523",
     "--server.enableCORS","false","--server.enableXsrfProtection","false"]
