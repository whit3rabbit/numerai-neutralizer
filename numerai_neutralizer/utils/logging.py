import logging
import json
from typing import Optional, Dict
from functools import wraps
from time import time

def setup_logger(
    name: str = "numerai_neutralizer",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configure logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_with_data(logger: logging.Logger, level: int, msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    """Helper function to log messages with structured data."""
    if structured_data:
        try:
            msg = f"{msg} | Data: {json.dumps(structured_data)}"
        except (TypeError, ValueError):
            msg = f"{msg} | Data: {str(structured_data)}"
    logger.log(level, msg, *args, **kwargs)

# Performance monitoring decorator
def log_performance(func):
    """Decorator to log function performance metrics."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        try:
            result = func(*args, **kwargs)
            execution_time = time() - start_time
            
            # Log performance metrics
            log_with_data(
                logger,
                logging.DEBUG,
                f"Function {func.__name__} completed",
                {
                    'function': func.__name__,
                    'execution_time_seconds': execution_time,
                    'success': True
                }
            )
            return result
        except Exception as e:
            execution_time = time() - start_time
            log_with_data(
                logger,
                logging.ERROR,
                f"Error in function {func.__name__}",
                {
                    'function': func.__name__,
                    'execution_time_seconds': execution_time,
                    'error': str(e),
                    'success': False
                }
            )
            raise
    return wrapper

# Initialize default logger
logger = setup_logger()

# Convenience methods for structured logging
def debug(msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    log_with_data(logger, logging.DEBUG, msg, structured_data, *args, **kwargs)

def info(msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    log_with_data(logger, logging.INFO, msg, structured_data, *args, **kwargs)

def warning(msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    log_with_data(logger, logging.WARNING, msg, structured_data, *args, **kwargs)

def error(msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    log_with_data(logger, logging.ERROR, msg, structured_data, *args, **kwargs)

def critical(msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    log_with_data(logger, logging.CRITICAL, msg, structured_data, *args, **kwargs)

def exception(msg: str, structured_data: Optional[Dict] = None, *args, **kwargs):
    if structured_data:
        try:
            msg = f"{msg} | Data: {json.dumps(structured_data)}"
        except (TypeError, ValueError):
            msg = f"{msg} | Data: {str(structured_data)}"
    logger.exception(msg, *args, **kwargs)