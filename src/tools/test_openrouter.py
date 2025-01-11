import os
import logging
from typing import Optional, List, Mapping, Any, Dict, ClassVar
from pydantic import Field, ConfigDict
from langchain.llms.base import LLM
from openrouter_client import OpenRouterClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenRouterLLM(LLM):
    """Custom LLM that uses OpenRouterClient"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        validate_assignment=True,
        protected_namespaces=(),
        str_strip_whitespace=True,
        str_max_length=2000
    )
    
    client: Any = Field(default=None, description="OpenRouter client instance")
    system_prompt: str = Field(default="", description="System prompt for the LLM")
    api_url: str = Field(default='https://openrouter.ai/api/v1/chat/completions', description="API URL")
    
    def __init__(self, **kwargs):
        logger.debug("Initializing OpenRouterLLM with kwargs: %s", kwargs)
        super().__init__(**kwargs)
        logger.debug("OpenRouterLLM initialized successfully")
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the OpenRouter API and return the response."""
        logger.debug("Calling OpenRouter API with prompt: %s", prompt)
        response = self.client.get_completion(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.0
        )
        if response.get("error"):
            raise ValueError(f"OpenRouter API error: {response['error']}")
        return response["content"]
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.client.model}

def main():
    """Test the OpenRouterClient and OpenRouterLLM classes."""
    try:
        logger.info("Starting OpenRouter test")
        
        # Test OpenRouterClient
        logger.info("Testing OpenRouterClient initialization")
        client = OpenRouterClient(component_type='test')
        logger.info("OpenRouterClient initialized successfully")
        
        # Test OpenRouterLLM
        logger.info("Testing OpenRouterLLM initialization")
        llm = OpenRouterLLM()
        llm.client = client
        logger.info("OpenRouterLLM initialized successfully")
        
        # Test a simple completion
        test_prompt = "What is 2+2?"
        logger.info("Testing completion with prompt: %s", test_prompt)
        result = llm(test_prompt)
        logger.info("Received response: %s", result)
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error("Test failed with error: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()