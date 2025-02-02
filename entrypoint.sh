#!/bin/bash
set -e

# Configure reliable DNS
echo "nameserver 8.8.8.8" > /etc/resolv.conf
echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# Set up Hugging Face cache
export TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface
export HF_HOME=/home/appuser/.cache/huggingface
export HF_DATASETS_CACHE=/home/appuser/.cache/huggingface
mkdir -p /home/appuser/.cache/huggingface
chown -R appuser:appuser /home/appuser/.cache

# Function to check if a variable is set
check_env_var() {
    if [ -z "${!1}" ]; then
        echo "Error: $1 is not set" >&2
        exit 1
    fi
}

# Check required environment variables
check_env_var OPENROUTER_API_KEY
check_env_var LLM_SHERPA_URL

# Generate .env file
echo "Generating .env file from environment variables"
printf "OPENROUTER_API_KEY=%s\n" "$OPENROUTER_API_KEY" > .env

# Generate config.yaml
if [ ! -f config/config.yaml ]; then
    echo "Generating config.yaml from environment variables"
    mkdir -p config || { echo "Error: Failed to create config directory" >&2; exit 1; }
    
    cat << EOF > config/config.yaml
llm_sherpa:
  api_url: ${LLM_SHERPA_URL}/api/parseDocument?renderFormat=all
openrouter:
  api_key: "${OPENROUTER_API_KEY}"
  site_url: ${HYBRIDRAG_SITE_URL:-http://127.0.0.1:3000}
  site_name: HybridRAG
EOF

    if [ $? -ne 0 ]; then
        echo "Error: Failed to create config.yaml" >&2
        exit 1
    fi
fi

# Execute with retry logic
max_retries=5
retry_delay=10
attempt=1

while [ $attempt -le $max_retries ]; do
    if [ $# -eq 0 ]; then
        echo "Attempt $attempt/$max_retries: Running app.py"
        python app.py && exit 0 || echo "Attempt $attempt failed"
    else
        echo "Attempt $attempt/$max_retries: Executing command"
        "$@" && exit 0 || echo "Attempt $attempt failed"
    fi
    
    attempt=$((attempt + 1))
    if [ $attempt -le $max_retries ]; then
        echo "Waiting 5 seconds before next attempt..."
        sleep 5
    fi
done

echo "All $max_retries attempts failed!"
exit 1