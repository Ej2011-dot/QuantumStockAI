# utils/logger.py
# Structured logging setup for QuantumTrader.

import logging
import os
from datetime import datetime


def setup_logger(name: str = "quantum_trader", log_dir: str = "results", level: str = "INFO") -> logging.Logger:
    """
    Set up a logger that writes to both stdout and a timestamped file.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logger initialised. Log file: %s", log_file)
    return logger
