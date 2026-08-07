FROM python:3.9-slim

WORKDIR /app

# Install build dependencies needed for compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with timeout and retries
RUN pip install --no-cache-dir \
    --default-timeout=1000 \
    -r requirements.txt

# Copy application code
COPY . .

ENV LGL_OUTPUT_DIR=/docker
EXPOSE 5006
CMD ["python", "vae_lgl_analysis.py", "--host", "0.0.0.0", "--port", "5006"]
