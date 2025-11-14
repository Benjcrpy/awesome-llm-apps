
FROM python:3.11-slim

# Working directory sa loob ng container
WORKDIR /app

# Install system deps kung kailangan mag-build ng wheels later
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements ng AI Health & Fitness agent
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy lahat ng project files
COPY . .

# Env basics
ENV PYTHONUNBUFFERED=1

# Streamlit will listen on port 3000 para sakto sa Coolify default
EXPOSE 3000

# Start command
CMD ["sh", "-c", "streamlit run health_agent.py --server.port=${PORT:-3000} --server.address=0.0.0.0"]
