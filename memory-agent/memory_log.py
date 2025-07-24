"""
This module sets up a logger that sends logs to a Loki instance.
Info: https://github.com/xente/loki-logger-handler
"""

import logging
import os
import json
from loki_logger_handler.loki_logger_handler import LokiLoggerHandler
from typing import Any


def get_logger(**kwargs) -> logging.Logger:
    """
    Returns the logger instance.
    Args:
        **kwargs: Additional parameters for configuring the logger, such as:
            - name (str): The name of the logger.
            - loki_url (str): The URL of the Loki server.
            - level (str): The logging level (DEBUG, INFO, WARNING,
              ERROR, CRITICAL).
            - service (str): The name of the service.
            - version (str): The version of the application.
    Raises:
        ValueError: If the log level is invalid.

    Returns:
        logging.Logger: The logger instance configured for Loki.
    """

    name = kwargs.get("name", os.getenv("APP_NAME", "logger"))
    loki_url: str | None = kwargs.get("loki_url", os.getenv("LOKI_URL"))
    level = kwargs.get("level", os.getenv("LOG_LEVEL", "DEBUG"))
    service = kwargs.get("service", os.getenv("APP_SERVICE", "logger_service"))
    version = kwargs.get("version", os.getenv("APP_VERSION", "1.0.0"))

    logger = logging.getLogger(name)

    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError(
            f"Invalid log level: {level}. Must be one of DEBUG, INFO, "
            "WARNING, ERROR, CRITICAL."
        )

    if level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    elif level == "INFO":
        logger.setLevel(logging.INFO)
    elif level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)

    labels: dict[str, Any] = {
        "application": name,
        "environment": os.getenv("APP_ENV")
    }
    metadata_default = {
        "service": service,
        "version": version
    }

    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(funcName)s:%(lineno)d - '
        '%(levelname)s - %(message)s'
    )

    if os.getenv("APP_ENV") != "production" or loki_url is None:
        # Add a console handler for local development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        # Create an instance of the custom handler
        agent_logger_handler = LokiLoggerHandler(
            url=loki_url,
            labels=labels,
            label_keys={},
            timeout=10,
            enable_structured_loki_metadata=True,
            loki_metadata=metadata_default,
            loki_metadata_keys=["thread_id"]
        )
        agent_logger_handler.setFormatter(formatter)
        agent_logger_handler.setLevel(level)
        logger.addHandler(agent_logger_handler)

    return logger


def get_metadata(thread_id: str, metadata: dict | None = None) -> dict:
    """
    Prepares metadata for logging.

    Args:
        metadata (dict, optional): Additional metadata to include in the log.

    Returns:
        dict: A dictionary containing the thread ID and
        any additional metadata.
    """
    extra_metadata = {"thread_id": thread_id}
    if metadata:
        extra_metadata["loki_metadata"] = json.dumps(metadata)
    return extra_metadata
