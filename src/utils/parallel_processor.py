from typing import TypeVar, List, Callable, Optional, Type
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor, as_completed
import multiprocessing
from functools import wraps
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')  # Type of items to process
R = TypeVar('R')  # Type of result

def parallel_batch_process(
    batch_size: int = 1000,
    executor_type: Type[Executor] = ThreadPoolExecutor,
    max_workers: Optional[int] = None
):
    """
    Decorator for parallel batch processing.
    
    Args:
        batch_size: Size of batches to process
        executor_type: Type of executor to use (ThreadPoolExecutor or ProcessPoolExecutor)
        max_workers: Maximum number of workers (defaults to CPU count)
        
    Returns:
        Decorated function that processes items in parallel batches
    """
    def decorator(func: Callable[[List[T]], List[R]]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> List[R]:
            items = args[0] if args else kwargs.get('items', [])
            if not items:
                return []
                
            workers = max_workers or min(32, (multiprocessing.cpu_count() * 2))
            results = []
            
            try:
                with executor_type(max_workers=workers) as executor:
                    futures = []
                    
                    # Process items in batches
                    for i in range(0, len(items), batch_size):
                        batch = items[i:i + batch_size]
                        future = executor.submit(func, batch, *args[1:], **kwargs)
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in as_completed(futures):
                        try:
                            batch_result = future.result()
                            if isinstance(batch_result, list):
                                results.extend(batch_result)
                            else:
                                results.append(batch_result)
                        except Exception as e:
                            logger.error(f"Error in parallel batch processing: {str(e)}")
                            
                return results
                
            except Exception as e:
                logger.error(f"Error setting up parallel processing: {str(e)}")
                return []
                
        return wrapper
    return decorator

def get_optimal_executor(cpu_bound: bool = False) -> Type[Executor]:
    """
    Get the optimal executor type based on the workload.
    
    Args:
        cpu_bound: Whether the workload is CPU-bound
        
    Returns:
        Executor type to use
    """
    if cpu_bound:
        return ProcessPoolExecutor
    return ThreadPoolExecutor

def get_optimal_batch_size(total_items: int) -> int:
    """
    Calculate optimal batch size based on total items.
    
    Args:
        total_items: Total number of items to process
        
    Returns:
        Optimal batch size
    """
    cpu_count = multiprocessing.cpu_count()
    min_batch = 100
    max_batch = 5000
    
    # Aim for at least 2 items per CPU core, but no more than max_batch
    optimal = max(min_batch, total_items // (cpu_count * 2))
    return min(optimal, max_batch)
