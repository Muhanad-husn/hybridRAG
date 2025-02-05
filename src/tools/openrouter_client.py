import os
import time
import random
import requests
import yaml
from typing import Dict, Any, Optional, ClassVar
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient(BaseModel):
    """Client for making requests to OpenRouter API"""
    
    # Model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Class constants
    API_BASE_URL: ClassVar[str] = 'https://openrouter.ai/api/v1/chat/completions'
    DEFAULT_MODEL: ClassVar[str] = 'anthropic/claude-3.5-haiku-20241022'
    DEFAULT_MAX_RETRIES: ClassVar[int] = 5
    DEFAULT_BASE_DELAY: ClassVar[int] = 1
    DEFAULT_TIMEOUT: ClassVar[int] = 30
    
    # Instance fields with proper type annotations
    api_key: Optional[str] = Field(None, description="OpenRouter API key")
    component_type: str = Field(default='extract_model', description="Component type")
    model: str = Field(default=DEFAULT_MODEL, description="Model identifier")
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES, description="Maximum retry attempts")
    base_delay: int = Field(default=DEFAULT_BASE_DELAY, description="Base delay between retries")
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Request timeout in seconds")
    max_tokens: int = Field(description="Maximum tokens in response")
    temperature: float = Field(description="Sampling temperature")
    
    def __init__(self, **data):
        """Initialize OpenRouter client with API key and configuration"""
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Set max_tokens and temperature from config
        data['max_tokens'] = config['llm']['max_tokens']
        data['temperature'] = config['llm']['temperature']
        
        super().__init__(**data)
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

    def get_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Get completion from OpenRouter API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides instance value if provided)
            max_tokens: Maximum tokens in response (overrides instance value if provided)
            top_p: Nucleus sampling parameter
            
        Returns:
            Dict containing response content and metadata
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "HybridRAG"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
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
                    url=self.API_BASE_URL,
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
                #logger.info(f"Raw API response: {response_data}")
            
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