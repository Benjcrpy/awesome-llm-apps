# --- Dockerfile for AI Mental Wellbeing Agent ---
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from repo
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Ensure the 'autogen' namespace is available
# (Some environments require explicit install of the meta/namespace package.)
RUN pip install --no-cache-dir \
    "autogen-agentchat>=0.2.36" \
    "autogen-ext>=0.2.36" \
    "openai>=1.40.0"

# Sanity check during build (fails early if not present)
RUN python -c "import autogen; print('AutoGen OK:', getattr(autogen, '__version__', 'unknown'))"

# Copy app code
COPY . /app

# Runtime env
ENV AUTOGEN_USE_DOCKER=0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=5590
EXPOSE 5590

# Start Streamlit on fixed port 5590
CMD ["bash","-lc","streamlit run ai_mental_wellbeing_agent.py --server.address ${STREAMLIT_SERVER_ADDRESS} --server.port ${STREAMLIT_SERVER_PORT}"]
