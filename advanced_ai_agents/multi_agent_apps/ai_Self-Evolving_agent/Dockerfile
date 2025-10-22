FROM python:3.11-slim

# Keep image small / no pip cache
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install your EXACT requirements.txt (as-is)
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# 2) Install Streamlit explicitly (fixes ModuleNotFoundError)
RUN python -m pip install --no-cache-dir streamlit==1.36.0 && \
    python - <<'PY'\nimport streamlit; print("STREAMLIT OK:", streamlit.__version__)\nPY

# 3) Install EvoAgentX from ZIP (NO git, NO apt-get)
#    Avoids dpkg/apt "No space left on device"
RUN python -m pip install --no-cache-dir \
    https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# 4) Copy app source last
COPY . /app

# 5) Run Streamlit directly
ENV PORT=5523
EXPOSE 5523
CMD python -m streamlit run ai_Self-Evolving_agent.py \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-5523} \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
