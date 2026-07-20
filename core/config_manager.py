"""
core/config_manager.py
======================

Centralized configuration provider for StarkOS.

Responsibilities
----------------
- Load configuration
- Validate configuration
- Cache values
- Provide immutable snapshots
- Support runtime environments
- Future hot-reload support
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

from core.event_bus import Event, EventBus
from core.service_container import ServiceContainer

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIG_FILE: Final[Path] = Path("Config.Yaml")
DEFAULT_ENVIRONMENT: Final[str] = "development"
CONFIG_VERSION: Final[str] = "0.4"
CACHE_SIZE_LIMIT: Final[int] = 512

# =============================================================================
# Exceptions
# =============================================================================

class ConfigError(RuntimeError):
    """Base configuration exception."""

class ConfigLoadError(ConfigError):
    """Raised when configuration cannot be loaded."""

class ConfigValidationError(ConfigError):
    """Raised when configuration is invalid."""

class MissingConfigurationError(ConfigError):
    """Raised when a required configuration key does not exist."""

# =============================================================================
# Environment
# =============================================================================

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

# =============================================================================
# Metadata
# =============================================================================

@dataclass(slots=True, frozen=True)
class ConfigMetadata:
    version: str
    environment: Environment
    source: Path
    loaded_at: datetime

# =============================================================================
# Immutable Snapshot
# =============================================================================

@dataclass(slots=True, frozen=True)
class ConfigSnapshot:
    metadata: ConfigMetadata
    data: dict[str, Any]

# =============================================================================
# Config Manager
# =============================================================================

class ConfigManager:
    """
    Centralized configuration provider.

    The ConfigManager is the single source of truth
    for runtime configuration.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config_path: Path = DEFAULT_CONFIG_FILE,
        environment: Environment = Environment.DEVELOPMENT,
    ) -> None:
        self._services = services
        self._config_path = config_path
        self._environment = environment
        self._config: dict[str, Any] = {}
        self._cache: dict[str, Any] = {}
        self._metadata: ConfigMetadata | None = None
        self._last_modified = 0.0
        self._last_fingerprint = ""
        self._event_bus: EventBus | None = None

        logger.info("ConfigManager created (env=%s).", environment.value)

    @property
    def environment(self) -> Environment:
        return self._environment

    @property
    def metadata(self) -> ConfigMetadata:
        if self._metadata is None:
            raise ConfigLoadError("Configuration not loaded.")
        return self._metadata

    @property
    def config_path(self) -> Path:
        return self._config_path

    def _default_config(self) -> dict[str, Any]:
        return {
            "kernel": {
                "name": "StarkOS",
                "version": "0.4",
                "auto_start": True,
            },
            "logging": {
                "level": "INFO",
                "structured": True,
            },
            "services": {},
            "modules": {},
            "identity": {
                "persona": "STARK",
            },
            "environment": {
                "name": self.environment.value,
            },
        }

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _load_yaml(self) -> dict[str, Any]:
        logger.info("Loading configuration: %s", self._config_path)
        if not self._config_path.exists():
            raise ConfigLoadError(f"Configuration file '{self._config_path}' does not exist.")
        try:
            with self._config_path.open("r", encoding="utf8") as stream:
                data = yaml.safe_load(stream)
        except Exception as exc:
            logger.exception("Unable to load YAML.")
            raise ConfigLoadError("YAML loading failed.") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigValidationError("Configuration root must be a mapping.")
        return data

    def _validate(self, config: dict[str, Any]) -> None:
        logger.debug("Validating configuration.")
        required = ("kernel", "logging", "services")
        for key in required:
            if key not in config:
                raise ConfigValidationError(f"Missing section '{key}'.")
            if not isinstance(config[key], dict):
                raise ConfigValidationError(f"Section '{key}' must be a mapping.")

    def _apply_environment(self, config: dict[str, Any]) -> dict[str, Any]:
        environments = config.get("environments", {})
        if not isinstance(environments, dict):
            return config
        override = environments.get(self.environment.value, {})
        if not isinstance(override, dict):
            return config
        logger.info("Applying environment: %s", self.environment.value)
        return self._deep_merge(config, override)

    def load(self) -> None:
        logger.info("Loading runtime configuration.")
        yaml_config = self._load_yaml()
        merged = self._deep_merge(self._default_config(), yaml_config)
        merged = self._apply_environment(merged)
        self._validate(merged)
        self._config = merged
        self._invalidate_cache()
        self._update_metadata()
        logger.info("Configuration loaded.")

    def reload(self) -> None:
        logger.info("Reloading configuration.")
        self.load()

    def _invalidate_cache(self) -> None:
        self._cache.clear()
        logger.debug("Configuration cache cleared.")

    def _update_metadata(self) -> None:
        self._metadata = ConfigMetadata(
            version=CONFIG_VERSION,
            environment=self._environment,
            source=self._config_path,
            loaded_at=datetime.utcnow(),
        )

    def get(self, path: str, default: Any = None) -> Any:
        if path in self._cache:
            return self._cache[path]
        try:
            value = self._lookup(path)
        except MissingConfigurationError:
            if default is not None:
                return default
            raise
        if len(self._cache) < CACHE_SIZE_LIMIT:
            self._cache[path] = value
        return value

    def _lookup(self, path: str) -> Any:
        node: Any = self._config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise MissingConfigurationError(f"Configuration key '{path}' not found.")
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        section = self.get(name)
        if not isinstance(section, dict):
            raise ConfigValidationError(f"'{name}' is not a section.")
        return copy.deepcopy(section)

    def kernel_config(self) -> dict[str, Any]:
        return self.section("kernel")

    def logging_config(self) -> dict[str, Any]:
        return self.section("logging")

    def services_config(self) -> dict[str, Any]:
        return self.section("services")

    def modules_config(self) -> dict[str, Any]:
        return self.section("modules")

    def identity_config(self) -> dict[str, Any]:
        return self.section("identity")

    def environment_config(self) -> dict[str, Any]:
        return self.section("environment")

    def snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            metadata=self.metadata,
            data=copy.deepcopy(self._config),
        )

    def fingerprint(self) -> str:
        serialized = json.dumps(
            self._config,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf8")).hexdigest()

    def remember_state(self) -> None:
        self._last_modified = self.config_path.stat().st_mtime if self.config_path.exists() else 0.0
        self._last_fingerprint = self.fingerprint()

    def check_updates(self) -> bool:
        if not self.config_path.exists():
            return False
        modified = self.config_path.stat().st_mtime
        return modified != getattr(self, "_last_modified", 0.0)

    async def reload_if_changed(self) -> bool:
        if not self.check_updates():
            return False
        previous = self.fingerprint()
        self.reload()
        current = self.fingerprint()
        self.remember_state()
        if current != previous:
            logger.info("Configuration updated.")
            await self._publish_event("config.changed", fingerprint=current, environment=self.environment.value)
            return True
        return False

    async def watch(self, interval: float = 2.0) -> None:
        import asyncio
        logger.info("Configuration watcher started.")
        while True:
            try:
                await self.reload_if_changed()
            except Exception:
                logger.exception("Configuration watch failure.")
            await asyncio.sleep(interval)

    def bind_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        logger.info("EventBus attached to ConfigManager.")

    async def _publish_event(self, event_name: str, **payload: Any) -> None:
        # NOTE: EventBus.publish() takes a single Event object, not
        # (topic, payload) positional arguments -- build the Event here.
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(Event(topic=event_name, source="ConfigManager", payload=payload))
        except Exception:
            logger.exception("Unable to publish configuration event.")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "environment": self.environment.value,
            "cache_entries": len(self._cache),
            "config_file": str(self.config_path),
            "fingerprint": self.fingerprint(),
            "metadata": self.metadata.__dict__ if self._metadata else None,
        }

    @property
    def loaded(self) -> bool:
        return bool(self._config)