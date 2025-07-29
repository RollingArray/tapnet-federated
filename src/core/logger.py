"""
===============================================================================
logger.py — Configurable Logging Utility for TapNet Federated Learning
===============================================================================

Description:
    This module provides a flexible logging setup for both server and client 
    scripts used in the TapNet Federated Learning framework.

    It enables:
    - Separate log files for server and client processes
    - Simultaneous logging to file and console
    - Consistent timestamped formatting
    - Prevention of duplicate handlers

Configuration:
    - Log paths are configured in `config.py` as SERVER_LOG_FILE and CLIENT_LOG_FILE.
    - Supported log types: 'server' or 'client'
    - Logging level: INFO (can be changed as needed)

Usage:
    from logger import get_logger

    log = get_logger("server", "server")
    log.info("Server started")

Author: Ranjoy Sen
===============================================================================
"""

import logging
from config import SERVER_LOG_FILE, CLIENT_LOG_FILE, MODEL_COMPARE_LOG_FILE

def get_logger(name: str, log_type: str = "server") -> logging.Logger:
    """
    Get a configured logger instance for server or client.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    if log_type == "server":
        log_file = SERVER_LOG_FILE
    elif log_type == "client":
        log_file = CLIENT_LOG_FILE
    elif log_type == "compare":
        log_file = MODEL_COMPARE_LOG_FILE
    
    else:
        raise ValueError("log_type must be 'server' or 'client'")

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
