FROM python:3.12-slim

WORKDIR /app

# Install system dependencies. iptables: the bot (privileged, host network) inserts the sandbox
# network's egress filter into the host's firewall at startup (secure_container/sessions.py).
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    ffmpeg \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the bot
CMD ["python", "main_bot.py"] 