import logging
from logging.handlers import RotatingFileHandler
import os
import yaml
from typing import Optional
from datetime import datetime

class DuplicateFilter(logging.Filter):
    """Filter to prevent repeated log messages within a short time window."""
    def __init__(self, timeout=1.0):
        super().__init__()
        self.timeout = timeout
        self.last_log = {}
        
    def filter(self, record):
        # Create a key from the log record's essential attributes
        key = (record.module, record.levelno, record.msg)
        current_time = record.created
        
        # Check if we've seen this message recently
        if key in self.last_log:
            if current_time - self.last_log[key] < self.timeout:
                return False
            
        self.last_log[key] = current_time
        return True

class RequestFormatter(logging.Formatter):
    """Custom formatter that includes request details when available."""
    
    def format(self, record):
        # Get the original format
        fmt = self._fmt
        
        # If there's request info, add it to the format
        if hasattr(record, 'request'):
            req = record.request
            record.url = f"{req.method} {req.scheme}://{req.host}{req.path}"
            if req.query_string:
                record.url += f"?{req.query_string.decode('utf-8')}"
            record.remote_addr = req.remote_addr
            fmt = '%(asctime)s - %(levelname)s - [%(url)s] - %(remote_addr)s - %(message)s'
        elif hasattr(record, 'url'):
            fmt = '%(asctime)s - %(levelname)s - [%(url)s] - %(message)s'
            
        # Set the format and delegate to parent
        self._fmt = fmt
        result = super().format(record)
        self._fmt = self._style._fmt
        return result

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
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        root_logger.handlers = []
        
        # Create formatters
        file_formatter = RequestFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = RequestFormatter(
            '%(levelname)s - %(message)s'
        )
        
        # Setup file handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=log_config.get('max_log_size', 5 * 1024 * 1024),  # 5 MB default
            backupCount=log_config.get('backup_count', 3),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(DuplicateFilter(1.0))  # 1 second timeout for duplicates
        root_logger.addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(DuplicateFilter(1.0))
        root_logger.addHandler(console_handler)
        
        # Configure werkzeug logger to reduce HTTP request logs
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.WARNING)
        
        # Configure Flask logger
        flask_logger = logging.getLogger('flask')
        flask_logger.setLevel(logging.INFO)
        flask_logger.propagate = False
        
        # Configure other loggers
        for logger_name, logger_config in log_config.get('loggers', {}).items():
            if logger_name != 'root':
                logger = logging.getLogger(logger_name)
                logger.setLevel(getattr(logging, logger_config.get('level', 'INFO')))
                logger.propagate = logger_config.get('propagate', False)
        
        logging.info("Logger initialized successfully")
        
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        # Set up a basic console logger as fallback
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
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