FROM python:3.12-slim

WORKDIR /workspace

# Add debug prints before each step
RUN echo "===> STARTING INITIAL APT UPDATE" && \
    apt-get update && apt-get install -y apt-transport-https && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Add more detailed debugging output for troubleshooting
RUN echo "===> CHECKING BASE IMAGE TOOLS" && \
    echo "Python version:" && python --version && \
    echo "Checking for basic tools:" && \
    which grep || echo "grep not found" && \
    which sed || echo "sed not found" && \
    which awk || echo "awk not found"

# Add retry logic for apt-get update
RUN echo "===> STARTING APT UPDATE WITH RETRY" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    for i in $(seq 1 3); do apt-get update -y && break || sleep 5; done

# Install python
# Note: python3 and pip are already installed in python:3.12-slim image
RUN echo "===> CHECKING PYTHON INSTALLATION" && \
    python3 --version && \
    pip --version

# Install system dependencies in smaller groups to identify problematic packages
RUN echo "===> INSTALLING GIT, CURL, WGET" && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN echo "===> INSTALLING BUILD ESSENTIALS" && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "===> INSTALLING FFMPEG" && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN echo "===> INSTALLING NETWORK TOOLS" && \
    apt-get update && apt-get install -y --no-install-recommends \
    busybox \
    openssh-client \
    netcat-openbsd \
    iputils-ping \
    dnsutils \
    net-tools \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Split problematic packages into smaller groups for better error handling
RUN echo "===> INSTALLING TEXT EDITORS" && \
    apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    less \
    && rm -rf /var/lib/apt/lists/*

# Skip installation of text processing tools as they are likely already in the base image
# Instead, check if they exist and only install if missing
RUN echo "===> CHECKING TEXT PROCESSING TOOLS" && \
    (which grep && which sed && which awk && echo "Text processing tools already installed") || \
    (echo "Installing missing text processing tools" && \
     apt-get update && apt-get install -y --no-install-recommends \
     grep \
     sed \
     awk \
     && rm -rf /var/lib/apt/lists/*)

RUN echo "===> INSTALLING ARCHIVE TOOLS" && \
    apt-get update && apt-get install -y --no-install-recommends \
    tar \
    gzip \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN echo "===> INSTALLING JQ AND HTOP" && \
    apt-get update && apt-get install -y --no-install-recommends \
    jq \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN echo "===> INSTALLING SYSTEM TOOLS" && \
    apt-get update && apt-get install -y --no-install-recommends \
    lsof \
    strace \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
RUN echo "===> PREPARING FOR PIP INSTALL"
COPY requirements.txt .
RUN echo "===> INSTALLING PYTHON REQUIREMENTS" && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Python packages installed:"  && \
    pip list

# Create a non-root user to run commands
RUN echo "===> CREATING NON-ROOT USER" && \
    useradd -m -s /bin/bash runner && \
    echo "User created:" && id runner
RUN echo "===> SETTING UP USER WORKSPACE" && \
    mkdir -p /home/runner/workspace && chown -R runner:runner /home/runner && \
    echo "Workspace permissions:" && ls -la /home/runner

# Allow runner to use sudo without password for specific commands
RUN echo "===> CONFIGURING SUDO PERMISSIONS" && \
    echo "runner ALL=(ALL) NOPASSWD: /usr/bin/apt-get, /usr/bin/apt, /usr/bin/apt-cache" > /etc/sudoers.d/runner && \
    echo "Sudo config created:" && cat /etc/sudoers.d/runner

# Set up the entrypoint script
RUN echo "===> PREPARING ENTRYPOINT"
COPY entrypoint.sh /entrypoint.sh
RUN echo "===> SETTING ENTRYPOINT PERMISSIONS" && \
    chmod +x /entrypoint.sh && \
    echo "Entrypoint permissions:" && ls -la /entrypoint.sh

# Switch to non-root user only after all root operations are complete
USER runner
WORKDIR /home/runner/workspace

ENTRYPOINT ["/entrypoint.sh"] 