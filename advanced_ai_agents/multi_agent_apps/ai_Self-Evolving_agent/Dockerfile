FROM python:3.11-slim

# keep image small / no pip cache
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) install your EXACT requirements.txt (as-is)
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# 2) install streamlit explicitly + verify (no heredoc)
RUN python -m pip install --no-cache-dir streamlit==1.36.0 && \
    python -c "import streamlit; print('STREAMLIT OK:', streamlit.__version__)"

# 3) install EvoAgentX via ZIP (no git, no apt-get → avoids dpkg/no-space errors)
RUN python -m pip install --no-cache-dir \
    https://github.com/EvoAgentX/EvoAgentX/archive/refs/heads/main.zip

# 4) copy app source last (better caching)
COPY . /app

# 5) run streamlit directly
ENV PORT=5523
EXPOSE 5523
CMD ["python", "-m", "streamlit", "run", "ai_Self-Evolving_agent.py", "--server.address=0.0.0.0", "--server.port=5523", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
