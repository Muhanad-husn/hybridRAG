import logging
from logging.handlers import RotatingFileHandler
import os
import yaml
from typing import Optional

def setup_logger(config_path: str = "config/config.yaml") -> None:
    """
    Setup and configure the application logger.
    
    Args:
        config_path: Path to the configuration file
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join("logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        log_config = config.get('logging', {})
        log_level = getattr(
            logging,
            log_config.get('log_level', 'INFO').upper()
        )
        max_size = log_config.get('max_log_size', 5 * 1024 * 1024)  # 5 MB default
        backup_count = log_config.get('backup_count', 3)
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove any existing handlers
        root_logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler (with rotation)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        logging.info("Logger initialized successfully")
        
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        # Set up a basic console logger as fallback
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.error("Failed to setup logger with config, using basic configuration")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name if name else '')