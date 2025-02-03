import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigHandler:
    """Centralized configuration management for the application."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one config instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the config handler."""
        if self._config is None:
            self._config = {}
    
    @staticmethod
    def load(config_path: str = "config/config.yaml") -> Dict[str, Any]:
        """
        Load configuration from yaml file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {str(e)}")
            raise
    
    @classmethod
    def get_instance(cls) -> 'ConfigHandler':
        """Get the singleton instance of ConfigHandler."""
        if cls._instance is None:
            cls._instance = ConfigHandler()
        return cls._instance
    
    def load_config(self, config_path: str = "config/config.yaml") -> None:
        """Load configuration into the singleton instance."""
        self._config = self.load(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot notation supported, e.g., 'llm.model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self._config
            for part in key.split('.'):
                value = value.get(part, {})
            return value if value != {} else default
        except Exception as e:
            logger.error(f"Error retrieving config key '{key}': {str(e)}")
            return default
    
    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """
        Get an entire configuration section.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Dict containing section configuration or None if not found
        """
        return self._config.get(section)
    
    def validate_required_keys(self, required_keys: list) -> bool:
        """
        Validate that all required configuration keys exist.
        
        Args:
            required_keys: List of required configuration keys
            
        Returns:
            True if all required keys exist, False otherwise
        """
        try:
            for key in required_keys:
                value = self.get(key)
                if value is None:
                    logger.error(f"Missing required configuration key: {key}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating configuration: {str(e)}")
            return False

# Global instance
config = ConfigHandler.get_instance()