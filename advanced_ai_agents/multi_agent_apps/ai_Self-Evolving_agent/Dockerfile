FROM python:3.11-slim

WORKDIR /app

# (optional) system deps kung may mag-build na wheel; safe na minimal
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
# Install deps + make sure streamlit exists
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit

COPY . /app

ENV PORT=5523
EXPOSE 5523

# Use python -m para siguradong nasa tamang interpreter PATH
CMD ["python", "-m", "streamlit", "run", "ai_Self-Evolving_agent.py", "--server.address", "0.0.0.0", "--server.port", "5523"]
