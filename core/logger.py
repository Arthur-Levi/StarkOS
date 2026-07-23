"""
core/logger.py
===============

Centralized logging configuration for StarkOS.

Responsibilities
----------------
- Single source of truth for the `starkos.<module>` logger naming
  convention used by every module in the codebase.
- Configure console and/or rotating-file output, plain-text or
  structured JSON, from one place.
- Correlation ID support: tag every log line emitted while handling a
  given request/command/session with the same id, via a contextvar --
  no need to thread an id through every function signature.

This closes out the "Logger profissional" item from the original
StarkOS roadmap (JSON logs, rotation, correlation id) as a small,
dependency-free module built entirely on the standard library.

Usage
-----
    from core.logger import get_logger, configure_logging, LoggingConfig, correlation_scope

    logger = get_logger("kernel")  # -> logging.Logger named "starkos.kernel"

    configure_logging(LoggingConfig(level="INFO", structured=False))

    with correlation_scope():  # or correlation_scope("request-123")
        logger.info("Handling one request end to end.")
        # every log line emitted anywhere in this scope carries the same
        # correlation id, including from other modules/threads that
        # don't know about the scope explicitly.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# =============================================================================
# Exceptions
# =============================================================================

class LoggingConfigurationError(Exception):
    """Raised when a logging configuration value is invalid (e.g. an
    unknown level name)."""

# =============================================================================
# Naming convention
# =============================================================================

_NAMESPACE = "starkos"

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger under the `starkos.` namespace -- the single place
    that encodes StarkOS's logger-naming convention, so every module
    gets it right without repeating the prefix by hand. Idempotent:
    passing an already-prefixed name (or "starkos" itself) is left as-is.

        get_logger("kernel")          -> logger "starkos.kernel"
        get_logger("cli.console")     -> logger "starkos.cli.console"
        get_logger("starkos.kernel")  -> logger "starkos.kernel" (unchanged)
    """
    if name == _NAMESPACE or name.startswith(f"{_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_NAMESPACE}.{name}")

# =============================================================================
# Correlation ID (contextvar-based -- no signature threading required)
# =============================================================================

_correlation_id: ContextVar[Optional[str]] = ContextVar("starkos_correlation_id", default=None)

def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]

def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()

def set_correlation_id(correlation_id: Optional[str]) -> None:
    _correlation_id.set(correlation_id)

@contextmanager
def correlation_scope(correlation_id: Optional[str] = None) -> Iterator[str]:
    """Tag every log line emitted within this `with` block (in this
    task/thread) with the same correlation id, restoring the previous
    value on exit. Generates a fresh id if none is supplied."""
    token = _correlation_id.set(correlation_id or new_correlation_id())
    try:
        yield _correlation_id.get()  # type: ignore[return-value]
    finally:
        _correlation_id.reset(token)

class _CorrelationIdFilter(logging.Filter):
    """Injects the current correlation id (or "-") into every record
    passing through a handler that has this filter attached."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get() or "-"
        return True

# =============================================================================
# Structured (JSON) formatting
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Renders each log record as one JSON object per line -- easy to ship
    to a log aggregator. Any `extra={...}` fields a module attaches are
    included verbatim (falling back to `str()` for anything that isn't
    JSON-serializable, so a formatting bug never breaks logging itself).
    """

    _RESERVED = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime", "correlation_id"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None) or "-",
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key in self._RESERVED or key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        return json.dumps(payload, ensure_ascii=False)

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    structured: bool = False  # JSON output (via JSONFormatter) when True
    console: bool = True
    log_file: Optional[Path] = None
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3
    include_correlation_id: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

def _resolve_level(level: str) -> int:
    resolved = getattr(logging, level.upper(), None)
    if not isinstance(resolved, int):
        raise LoggingConfigurationError(f"Unknown log level '{level}'.")
    return resolved

def _build_formatter(config: LoggingConfig) -> logging.Formatter:
    if config.structured:
        return JSONFormatter()
    text = "%(asctime)s | %(levelname)-8s | %(name)-28s"
    if config.include_correlation_id:
        text += " | cid=%(correlation_id)s"
    text += " | %(message)s"
    return logging.Formatter(text)

def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure the root logger for the whole process: console and/or a
    rotating log file, plain-text or JSON, with correlation ids attached
    if enabled. Safe to call more than once (replaces prior handlers
    rather than stacking them).
    """
    config = config or LoggingConfig()
    level = _resolve_level(config.level)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = _build_formatter(config)
    correlation_filter = _CorrelationIdFilter() if config.include_correlation_id else None

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        if correlation_filter is not None:
            console_handler.addFilter(correlation_filter)
        root.addHandler(console_handler)

    if config.log_file is not None:
        try:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file, maxBytes=config.max_bytes, backupCount=config.backup_count, encoding="utf-8"
            )
        except OSError as exc:
            raise LoggingConfigurationError(f"Unable to open log file '{config.log_file}'.") from exc
        file_handler.setFormatter(formatter)
        if correlation_filter is not None:
            file_handler.addFilter(correlation_filter)
        root.addHandler(file_handler)

    get_logger("logger").info(
        "Logging configured.",
        extra={"log_level": config.level, "structured": config.structured, "log_file": str(config.log_file or "")},
    )