import os
import time
import random
import requests
from typing import Dict, Any, Optional
from config.config_loader import config
from tools.logger import get_logger

logger = get_logger(__name__)

class OpenRouterClient:
    """Client for making requests to OpenRouter API"""
    
    def __init__(self, component_type: str = 'extract_model'):
        """Initialize OpenRouter client with API key and configuration"""
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        self.component_type = component_type
        
        # Load config
        self.config = config.get(component_type, {})
        self.api_base_url = self.config.get('api_base_url', 'https://openrouter.ai/api/v1/chat/completions')
        self.model = self.config.get('model', 'anthropic/claude-2.1')
        self.max_retries = self.config.get('max_retries', 5)
        self.base_delay = self.config.get('base_delay', 1)
        self.timeout = self.config.get('timeout', 30)

    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Get completion from OpenRouter API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 for more deterministic outputs)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            
        Returns:
            Dict containing response content and metadata
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/network_analysis_tool",
            "X-Title": "Graph4All Network Analysis Tool"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        # Remove None messages
        payload["messages"] = [msg for msg in payload["messages"] if msg is not None]
        
        for attempt in range(self.max_retries):
            try:
                # Validate payload before sending
                if not isinstance(payload.get("messages"), list) or not payload["messages"]:
                    raise ValueError("Invalid messages format in payload")

                # Make the API request
                response = requests.post(
                    url=self.api_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                # Handle non-200 responses
                if response.status_code != 200:
                    error_msg = f"API error {response.status_code}"
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            if "error" in error_data:
                                error_msg = f"{error_msg}: {error_data['error']}"
                            elif "message" in error_data:
                                error_msg = f"{error_msg}: {error_data['message']}"
                    except:
                        error_msg = f"{error_msg}: {response.text}"
                    
                    logger.error(f"OpenRouter API error: {error_msg}")
                    logger.error(f"Request payload: {payload}")
                    
                    # If model error, try fallback model
                    if "not a valid model ID" in error_msg:
                        fallback_model = "anthropic/claude-2.1"
                        logger.info(f"Trying fallback model: {fallback_model}")
                        payload["model"] = fallback_model
                        continue
                        
                    return {
                        "content": "",
                        "model": self.model,
                        "usage": {},
                        "error": error_msg
                    }

                # Parse successful response
                response_data = response.json()
            
                # Validate response structure
                if not response_data or "choices" not in response_data or not response_data["choices"]:
                    error_msg = "Invalid response structure from API"
                    logger.error(error_msg)
                    return {
                        "content": "",
                        "model": self.model,
                        "usage": {},
                        "error": error_msg
                    }

                # Extract content safely
                try:
                    content = response_data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    error_msg = f"Missing expected fields in API response: {str(e)}"
                    logger.error(error_msg)
                    return {
                        "content": "",
                        "model": self.model,
                        "usage": {},
                        "error": error_msg
                    }

                return {
                    "content": content,
                    "model": response_data.get("model", self.model),
                    "usage": response_data.get("usage", {}),
                    "error": None
                }
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = (self.base_delay * 2 ** attempt) + (random.randint(0, 1000) / 1000)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    error_msg = f"Failed after {self.max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    return {
                        "content": "",
                        "model": self.model,
                        "usage": {},
                        "error": error_msg
                    }

