FROM python:3.11-slim

# Keep image small / no pip cache
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install ALL deps in a single layer (from requirements.txt)
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip cache purge || true

# Copy source last for better caching
COPY . /app

# Start script (Streamlit)
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'echo "RUNTIME which python: $(which python)"' \
'python - <<PY' \
'import sys, streamlit' \
'print("RUNTIME PY:", sys.executable)' \
'print("RUNTIME STREAMLIT:", streamlit.__version__)' \
'PY' \
'exec python -m streamlit run ai_Self-Evolving_agent.py --server.address=0.0.0.0 --server.port=${PORT:-5523} --server.enableCORS=false --server.enableXsrfProtection=false' \
> /app/start.sh && chmod +x /app/start.sh

ENV PORT=5523
EXPOSE 5523

ENTRYPOINT ["/app/start.sh"]
