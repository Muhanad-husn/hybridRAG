from typing import Any
import logging
from functools import wraps
from typing import Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

class DocumentProcessingError(Exception):
    """Base exception class for document processing errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}

class CacheError(DocumentProcessingError):
    """Exception raised for cache-related errors."""
    pass

class InvalidFormatError(DocumentProcessingError):
    """Exception raised when document format is invalid or unsupported."""
    pass

class ConfigurationError(DocumentProcessingError):
    """Exception raised for configuration-related errors."""
    pass

class ProcessingError(DocumentProcessingError):
    """Exception raised for errors during document processing."""
    pass

class ValidationError(DocumentProcessingError):
    """Exception raised for document validation errors."""
    pass

def log_errors(logger: logging.Logger) -> Callable[[F], F]:
    """Decorator to handle errors and log them consistently."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except DocumentProcessingError as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", 
                           extra={"error_details": e.details})
                raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise DocumentProcessingError(f"Unexpected error: {str(e)}")
        return wrapper
    return decorator