from typing import Any
import logging
from functools import wraps
from typing import Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

def log_errors(logger: logging.Logger) -> Callable[[F], F]:
    """Decorator to handle errors and log them consistently."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator