"""
Logging utilities for the LLM Pipeline API.
"""

import logging
import sys
import json
import time
from typing import Optional, Any
from datetime import datetime
from functools import wraps
from contextlib import contextmanager

from fastapi import Request


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class RequestContextFilter(logging.Filter):
    """Filter that adds request context to log records."""

    def __init__(self):
        super().__init__()
        self.request_id: Optional[str] = None
        self.user_id: Optional[str] = None

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id
        record.user_id = self.user_id
        return True


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON formatting
        log_file: Optional file path for logging
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class RequestLogger:
    """Logger for HTTP requests with metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: dict[str, Any] = {}

    def log_request_start(
        self,
        request_id: str,
        method: str,
        path: str,
        client_ip: Optional[str] = None
    ) -> None:
        """Log the start of a request."""
        self.metrics[request_id] = {
            "start_time": time.perf_counter(),
            "method": method,
            "path": path
        }

        self.logger.info(
            f"Request started: {method} {path}",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "client_ip": client_ip
                }
            }
        )

    def log_request_end(
        self,
        request_id: str,
        status_code: int,
        tokens_used: Optional[int] = None
    ) -> None:
        """Log the end of a request."""
        metrics = self.metrics.pop(request_id, {})
        start_time = metrics.get("start_time", time.perf_counter())
        duration_ms = (time.perf_counter() - start_time) * 1000

        log_data = {
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "method": metrics.get("method"),
            "path": metrics.get("path")
        }

        if tokens_used:
            log_data["tokens_used"] = tokens_used

        level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(
            level,
            f"Request completed: {status_code} ({duration_ms:.2f}ms)",
            extra={"extra_data": log_data}
        )

    def log_inference(
        self,
        request_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float
    ) -> None:
        """Log inference metrics."""
        tokens_per_second = (completion_tokens / duration_ms) * 1000 if duration_ms > 0 else 0

        self.logger.info(
            f"Inference completed: {model}",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "duration_ms": round(duration_ms, 2),
                    "tokens_per_second": round(tokens_per_second, 2)
                }
            }
        )


# Global request logger
request_logger = RequestLogger(get_logger("llm_pipeline.requests"))


async def log_request(request: Request, call_next):
    """FastAPI middleware for request logging."""
    import uuid

    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    client_ip = request.client.host if request.client else None

    request_logger.log_request_start(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=client_ip
    )

    response = await call_next(request)

    request_logger.log_request_end(
        request_id=request_id,
        status_code=response.status_code
    )

    response.headers["X-Request-ID"] = request_id
    return response


@contextmanager
def log_duration(logger: logging.Logger, operation: str):
    """Context manager to log operation duration."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start) * 1000
        logger.info(f"{operation} completed in {duration:.2f}ms")
