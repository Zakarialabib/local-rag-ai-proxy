"""
Centralized logging configuration with file rotation and dated logging.
Modern Python logging using TimedRotatingFileHandler with structured logging via structlog.
"""

import logging
import logging.handlers
import structlog
from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class JSONRenderer:
    """Render logs as JSON for structured parsing."""

    def __call__(self, logger, method_name, event_dict):
        """Render event_dict as JSON with ISO timestamp."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        event_dict["timestamp"] = timestamp
        event_dict["level"] = method_name.upper()
        return json.dumps(event_dict, default=str)


def setup_logger(
    name: str,
    log_file: str = "proxy-api.log",
    level: int = logging.DEBUG,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Setup a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: Name of the log file (stored in logs/ directory)
        level: Logging level
        console: Add console handler
        file: Add rotating file handler

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation (rotate at midnight)
    if file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.handlers.TimedRotatingFileHandler(
            str(log_path),
            when="midnight",
            interval=1,
            backupCount=14,  # Keep 14 days of logs
            utc=False,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y%m%d"
        logger.addHandler(file_handler)

    return logger


def setup_structlog(name: str = "proxy-api") -> structlog.BoundLogger:
    """
    Setup structlog with dev console rendering for structured logging.

    Args:
        name: Logger name

    Returns:
        Configured structlog logger
    """
    # Setup the file logger first
    file_handler = logging.handlers.TimedRotatingFileHandler(
        str(LOGS_DIR / f"{name}-structured.log"),
        when="midnight",
        interval=1,
        backupCount=14,
        utc=False,
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    
    # Configure structlog with console and file rendering
    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),  # Pretty console output
        ],
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)
    
    print(f"[OK] Structured logging configured: {LOGS_DIR}")

    return structlog.get_logger(name)


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the centralized configuration."""
    if not logging.getLogger(name).handlers:
        setup_logger(name)
    return logging.getLogger(name)


def error_to_dict(exc: Exception) -> Dict[str, Any]:
    """Convert exception to dictionary for logging."""
    import traceback

    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


# Module-level shortcuts for common loggers
PROXY_LOGGER = setup_logger("proxy", log_file="proxy.log")
WEBAPP_LOGGER = setup_logger("webapp", log_file="webapp.log")
ERROR_LOGGER = setup_logger("errors", log_file="errors.log", level=logging.ERROR)


def log_api_error(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    exc: Exception,
    details: Dict[str, Any] = None,
) -> None:
    """
    Log API errors in a structured format.

    Args:
        logger: Logger instance
        endpoint: API endpoint
        method: HTTP method
        exc: Exception object
        details: Additional context dictionary
    """
    error_data = error_to_dict(exc)
    error_data.update(
        {
            "endpoint": endpoint,
            "method": method,
            **(details or {}),
        }
    )
    logger.error(f"API Error: {endpoint}", extra=error_data)


def log_request(
    logger: logging.Logger,
    endpoint: str,
    method: str,
    request_data: Dict[str, Any] = None,
    response_status: int = None,
    duration_ms: float = None,
) -> None:
    """
    Log API requests in a structured format.

    Args:
        logger: Logger instance
        endpoint: API endpoint
        method: HTTP method
        request_data: Request body/parameters
        response_status: HTTP response status code
        duration_ms: Request duration in milliseconds
    """
    log_data = {
        "endpoint": endpoint,
        "method": method,
    }
    if request_data:
        log_data["request"] = request_data
    if response_status is not None:
        log_data["status"] = response_status
    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    logger.info(f"{method} {endpoint}", extra=log_data)


if __name__ == "__main__":
    # Test logging setup
    logger = get_logger("test")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")

    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_api_error(logger, "/api/test", "GET", e, {"test_key": "test_value"})

    print(f"Logs directory: {LOGS_DIR}")
    print(f"Log files can be found in: {LOGS_DIR}")
