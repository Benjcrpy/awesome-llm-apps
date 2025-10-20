FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

ENV PORT=5523
EXPOSE 5523

CMD ["streamlit", "run", "ai_Self-Evolving_agent.py", "--server.address", "0.0.0.0", "--server.port", "5523"]
