import os
import sys
import logging

# Add src directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_layer.graph_constructor import GraphConstructor

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test the GraphConstructor class."""
    try:
        logger.info("Starting GraphConstructor test")
        
        # Initialize GraphConstructor
        logger.info("Testing GraphConstructor initialization")
        constructor = GraphConstructor()
        logger.info("GraphConstructor initialized successfully")
        
        # Test LLM initialization
        logger.info("Testing LLM access")
        llm = constructor.llm
        logger.info("LLM accessed successfully: %s", llm._llm_type if llm else "No LLM")
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error("Test failed with error: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()