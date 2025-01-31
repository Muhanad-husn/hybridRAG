#!/bin/bash

# Generate .env from mounted volume
if [ ! -f .env ]; then
  echo "Generating .env file from environment variables"
  printf "OPENROUTER_API_KEY=%s\n" "$OPENROUTER_API_KEY" > .env
fi

# Convert config.yaml environment variables
if [ ! -f config/config.yaml ]; then
  echo "Generating config.yaml from environment variables"
  mkdir -p config
  awk -v api_url="$LLM_SHERPA_URL/api/parseDocument?renderFormat=all" '
  BEGIN {
    print "llm_sherpa:"
    print "  api_url: " api_url
    print "openrouter:"
    print "  api_key: \"${OPENROUTER_API_KEY}\""
    print "  site_url: http://localhost:3000" 
    print "  site_name: HybridRAG"
  }
  ' > config/config.yaml
fi

exec "$@"