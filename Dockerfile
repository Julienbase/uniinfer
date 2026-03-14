# UniInfer — CPU variant
# Usage:
#   docker build -t uniinfer .
#   docker run -p 8000:8000 uniinfer --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"

FROM python:3.12-slim AS base

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install UniInfer (CPU-only llama-cpp-python)
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir prometheus-client

# Clean up build tools
RUN apt-get purge -y build-essential cmake && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create cache directory
RUN mkdir -p /root/.uniinfer/cache

EXPOSE 8000

ENTRYPOINT ["uniinfer", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
