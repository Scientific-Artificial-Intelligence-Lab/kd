import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str = None, log_level: str = 'DEBUG', max_log_size: int = 10 * 1024 * 1024, backup_count: int = 5):
    """
    Setup a logger with specified configurations.

    Args:
        name (str): The name of the logger (e.g., 'my_app').
        log_file (str, optional): The path to the log file. If None, logs will not be saved to a file.
        log_level (str): The logging level. E.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        max_log_size (int): Maximum size of the log file before it gets rotated (in bytes).
        backup_count (int): Number of backup files to keep when rotating logs.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    # Create a log format
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File handler for writing logs to a file, with rotation
    if log_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


