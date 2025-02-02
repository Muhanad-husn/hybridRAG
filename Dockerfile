# Build stage with model pre-download
FROM python:3.12.8-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies and model download tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev curl && \
    pip install --user -r requirements.txt && \
    huggingface-cli download Helsinki-NLP/opus-mt-ar-en --local-dir /root/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-ar-en && \
    apt-get purge -y --auto-remove gcc python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Runtime stage with DNS fixes
FROM python:3.12.8-slim

WORKDIR /app

# Install CA certificates and configure DNS
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy dependencies and model cache
COPY --from=builder /root/.local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache
COPY . .

# Create non-root user and set up cache directory
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser && \
    chmod -R 755 /home/appuser

# Ensure scripts are executable
RUN chmod +x entrypoint.sh

ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
# Prefer local cached models
ENV HF_HUB_OFFLINE=1

EXPOSE 5000

USER appuser

# Entrypoint with retry capability
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "app.py"]