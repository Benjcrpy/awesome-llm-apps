FROM python:3.11-slim

# Keep image small / no pip cache
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# We only need 'git' so pip can install evoagentx from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python deps
COPY requirements.txt /app/
RUN python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir streamlit==1.36.0 \
 && python - <<'PY'\nimport streamlit; print("STREAMLIT OK:", streamlit.__version__)\nPY \
 || (echo "Streamlit install failed" && exit 1)

# Copy source last for better caching
COPY . /app

ENV PORT=5523
EXPOSE 5523

# Run Streamlit directly (no loops)
CMD python -m streamlit run ai_Self-Evolving_agent.py \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-5523} \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
