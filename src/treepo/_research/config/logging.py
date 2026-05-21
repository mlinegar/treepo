"""
Centralized logging configuration.

Provides consistent logging format across all modules and scripts.

Usage:
    from treepo._research.config.logging import setup_logging

    # Basic usage (returns root logger)
    logger = setup_logging()

    # With custom level
    logger = setup_logging(level="DEBUG")
    logger = setup_logging(level=logging.DEBUG)

    # With verbose flag
    logger = setup_logging(verbose=True)  # DEBUG level

    # Named logger
    logger = setup_logging(name="my_module")
"""

import logging
from typing import Optional, Union


def setup_logging(
    level: Optional[Union[str, int]] = None,
    verbose: bool = False,
    name: Optional[str] = None,
    format_style: str = "default",
) -> logging.Logger:
    """
    Configure logging with a consistent format.

    Args:
        level: Logging level as string ("DEBUG", "INFO", etc.) or int.
               Defaults to INFO unless verbose=True.
        verbose: If True, sets level to DEBUG. Overridden by explicit level.
        name: Logger name. Defaults to root logger.
        format_style: Format style - "default" or "compact".

    Returns:
        Configured logger instance.
    """
    # Determine level
    if level is not None:
        if isinstance(level, str):
            log_level = getattr(logging, level.upper(), logging.INFO)
        else:
            log_level = level
    else:
        log_level = logging.DEBUG if verbose else logging.INFO

    # Select format
    if format_style == "compact":
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
        datefmt = "%H:%M:%S"
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%H:%M:%S"

    # Configure root logger (affects all loggers)
    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=datefmt,
        force=True,  # Override any existing config
    )

    # Return the requested logger
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    This is a convenience wrapper around logging.getLogger.
    Call setup_logging() first to configure the format.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Named logger instance.
    """
    return logging.getLogger(name)
