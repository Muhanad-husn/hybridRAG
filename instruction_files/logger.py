import logging
from logging.handlers import RotatingFileHandler
import os
import streamlit as st
from termcolor import colored
from config.config_loader import config

class StreamlitHandler(logging.Handler):
    """Custom handler to output logs to Streamlit"""
    def emit(self, record):
        try:
            # Only emit warnings in Streamlit context
            if hasattr(st, 'runtime') and record.levelno >= logging.WARNING:
                msg = self.format(record)
                st.warning(msg)
        except Exception:
            self.handleError(record)

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    COLORS = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    def format(self, record):
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname, 'white'))

def setup_logger():
    """Setup and configure the application logger"""
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Get configuration
    log_level = getattr(logging, config.get('logging', {}).get('log_level', 'INFO').upper())
    max_size = config.get('logging', {}).get('max_log_size', 5 * 1024 * 1024)  # 5 MB
    backup_count = config.get('logging', {}).get('backup_count', 3)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    root_logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    streamlit_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
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

    # Only add Streamlit handler if we're in a Streamlit context
    if hasattr(st, 'runtime'):
        streamlit_handler = StreamlitHandler()
        streamlit_handler.setLevel(logging.INFO)
        streamlit_handler.setFormatter(streamlit_formatter)
        root_logger.addHandler(streamlit_handler)

# Initialize logger
setup_logger()

def get_logger(name=None):
    """Get a logger instance"""
    return logging.getLogger(name if name else '')
