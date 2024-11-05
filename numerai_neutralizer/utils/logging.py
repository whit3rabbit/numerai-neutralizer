import logging
from typing import Optional

def setup_logger(
    name: str = "numerai_neutralizer",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Configure logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
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

# Initialize default logger
logger = setup_logger()