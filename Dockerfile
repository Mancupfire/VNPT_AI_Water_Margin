# VNPT AI Water Margin - Docker Submission
# Track 2: The Builder

# Use CUDA 12.2 base image (required by competition)
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements first (for layer caching)
COPY requirements.txt /code/

# Install Python dependencies (with progress output)
RUN pip3 install -v -r requirements.txt

# Copy application code
COPY . /code/

# Copy credentials (if available)
# Note: For security, you should inject these at runtime via volume mount
# For submission, you can include them in the image (organizers will use their own)
COPY .secret /code/.secret

# Copy environment config
COPY .env /code/.env 

# Download and initialize resources during build
# This pre-builds the vector database if documents are available
RUN python3 process_data.py || echo "Vector DB initialization skipped (will run during inference if needed)"

# Make inference script executable
RUN chmod +x /code/inference.sh

# Set entrypoint
CMD ["bash", "inference.sh"]

# Healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Metadata
LABEL maintainer="VNPT AI Water Margin Team"
LABEL description="Multiple-choice QA system with RAG and domain-based routing"
LABEL version="1.0"
