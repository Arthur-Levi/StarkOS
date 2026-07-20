"""
core/kernel.py
==============

Central orchestrator of StarkOS.

The Kernel is intentionally lightweight.

Responsibilities
----------------
- Compose infrastructure components.
- Coordinate lifecycle.
- Provide a unified public API.
- Publish high-level system events.
- Serve as the entry point for StarkOS.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import Any, Optional

from core.event_bus import Event, EventBus
from core.identity import Identity
from core.lifecycle import LifecycleManager
from core.module_registry import Module, ModuleRegistry
from core.service_container import ServiceContainer

logger = logging.getLogger("starkos.kernel")

# =============================================================================
# Exceptions
# =============================================================================

class KernelError(Exception):
    """Base Kernel exception."""

class KernelInitializationError(KernelError):
    """Raised when Kernel initialization fails."""

class KernelStartupError(KernelError):
    """Raised when startup fails."""

class KernelShutdownError(KernelError):
    """Raised when shutdown fails."""

class KernelStateError(KernelError):
    """Raised when an operation is invalid for the current state."""

# =============================================================================
# Kernel State
# =============================================================================

class KernelState(Enum):
    CREATED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class KernelConfig:
    name: str = "StarkOS"
    version: str = "0.4"
    startup_timeout: float = 30.0
    shutdown_timeout: float = 30.0
    enable_demo: bool = True
    publish_events: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Runtime Context
# =============================================================================

@dataclass(slots=True)
class KernelContext:
    state: KernelState = KernelState.CREATED
    boot_timestamp: float = 0.0
    started_at: float | None = None
    restart_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Kernel
# =============================================================================

class Kernel:
    """
    StarkOS central orchestrator.

    The Kernel coordinates the infrastructure
    without owning the implementation of any
    subsystem.
    """

    def __init__(
        self,
        *,
        config: KernelConfig | None = None,
        services: ServiceContainer | None = None,
        registry: ModuleRegistry | None = None,
        event_bus: EventBus | None = None,
        lifecycle: LifecycleManager | None = None,
        health_monitor: Any | None = None,
    ) -> None:
        self._config = config or KernelConfig()
        self._context = KernelContext()
        self._lock = RLock()

        self._services = services if services is not None else ServiceContainer()
        self._registry = registry if registry is not None else ModuleRegistry()
        self._event_bus = event_bus if event_bus is not None else EventBus()
        self._lifecycle = lifecycle if lifecycle is not None else LifecycleManager(
            registry=self._registry,
            services=self._services,
            event_publisher=self._event_bus,
        )
        self._health_monitor = health_monitor

        # Wiring
        self._registry.add_change_listener(self._lifecycle.notify_registry_changed)

        # Register core services (only if not already registered by the
        # composition root -- avoids DuplicateServiceError when the caller
        # passes in an already-populated ServiceContainer).
        # NOTE: register_instance(instance) takes the instance ONLY -- its
        # type is inferred via type(instance). Calling it as
        # register_instance(SomeType, instance) -- as the original code did
        # -- passes two positional args into a one-arg method and raises
        # TypeError at every boot.
        if not self._services.contains(ServiceContainer):
            self._services.register_instance(self._services)
        if not self._services.contains(ModuleRegistry):
            self._services.register_instance(self._registry)
        if not self._services.contains(EventBus):
            self._services.register_instance(self._event_bus)
        if not self._services.contains(LifecycleManager):
            self._services.register_instance(self._lifecycle)

        if self._health_monitor is not None and not self._services.contains(type(self._health_monitor)):
            self._services.register_instance(self._health_monitor)

        logger.info(
            "Kernel created.",
            extra={"version": self._config.version, "kernel_name": self._config.name},
        )

    def _set_state(self, state: KernelState) -> None:
        self._context.state = state
        logger.debug("Kernel state changed.", extra={"state": state.name})

    async def _publish(self, topic: str, **payload: Any) -> None:
        if not self._config.publish_events:
            return
        await self._event_bus.publish(
            Event(
                topic=topic,
                source="Kernel",
                payload=payload,
            )
        )

    @property
    def state(self) -> KernelState:
        return self._context.state

    @property
    def registry(self) -> ModuleRegistry:
        return self._registry

    @property
    def services(self) -> ServiceContainer:
        return self._services

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def lifecycle(self) -> LifecycleManager:
        return self._lifecycle

    # ============================================================================
    # Module Registration
    # ============================================================================

    def register_module(
        self,
        module: Module,
        *,
        name: str | None = None,
        priority: int = 100,
        dependencies: set[str] | None = None,
    ) -> None:
        self._registry.register_module(
            module=module,
            name=name,
            priority=priority,
            dependencies=dependencies or set(),
        )

        logger.info("Module registered.", extra={"module_name": name or module.__class__.__name__})

    # ============================================================================
    # Lifecycle
    # ============================================================================

    async def initialize(self) -> None:
        self._ensure_state(KernelState.CREATED, KernelState.STOPPED)
        self._set_state(KernelState.INITIALIZING)
        self._context.boot_timestamp = time.time()

        logger.info("Kernel initialization started.")

        try:
            await self._publish("kernel.initializing", version=self._config.version)
            await self._lifecycle.initialize_modules()
            self._set_state(KernelState.INITIALIZED)
            await self._publish("kernel.initialized", modules=len(self._registry.list_modules()))
            logger.info("Kernel initialized successfully.")
        except Exception as exc:
            self._set_state(KernelState.FAILED)
            logger.exception("Kernel initialization failed.")
            await self._publish("kernel.failed")
            raise KernelInitializationError("Unable to initialize Kernel.") from exc

    async def start(self) -> None:
        self._ensure_state(KernelState.INITIALIZED, KernelState.STOPPED)
        self._set_state(KernelState.STARTING)

        logger.info("Kernel startup started.")

        try:
            await self._publish("kernel.starting", version=self._config.version)
            await self._lifecycle.start_modules()
            self._context.started_at = time.time()
            self._set_state(KernelState.RUNNING)
            await self._publish("kernel.running")
            logger.info("Kernel started successfully.")
        except Exception as exc:
            self._set_state(KernelState.FAILED)
            logger.exception("Kernel startup failed.")
            await self._publish("kernel.failed")
            raise KernelStartupError("Kernel startup failed.") from exc

    async def stop(self) -> None:
        self._ensure_state(KernelState.RUNNING, KernelState.INITIALIZED)
        self._set_state(KernelState.STOPPING)

        logger.info("Kernel shutdown started.")

        try:
            await self._publish("kernel.stopping")
            await asyncio.wait_for(self._lifecycle.shutdown_modules(), timeout=self._config.shutdown_timeout)
            await self._services.shutdown()
            self._set_state(KernelState.STOPPED)
            await self._publish("kernel.stopped")
            logger.info("Kernel stopped successfully.")
        except asyncio.TimeoutError as exc:
            logger.error("Kernel shutdown timeout.")
            self._set_state(KernelState.FAILED)
            raise KernelShutdownError("Shutdown timeout.") from exc
        except Exception as exc:
            logger.exception("Kernel shutdown failed.")
            self._set_state(KernelState.FAILED)
            raise KernelShutdownError("Shutdown failed.") from exc

    async def restart(self) -> None:
        logger.info("Kernel restart requested.")
        self._context.restart_count += 1

        try:
            await self._publish("kernel.restarting", restart_count=self._context.restart_count)
            await self._lifecycle.restart()
            self._context.started_at = time.time()
            self._set_state(KernelState.RUNNING)
            await self._publish("kernel.restarted", restart_count=self._context.restart_count)
            logger.info("Kernel restarted successfully.")
        except Exception as exc:
            self._set_state(KernelState.FAILED)
            logger.exception("Kernel restart failed.")
            raise KernelError("Unable to restart Kernel.") from exc

    # ============================================================================
    # Public API
    # ============================================================================

    def resolve_module(self, name: str) -> Module:
        try:
            return self._registry.resolve_module(name)
        except Exception as exc:
            logger.exception("Unable to resolve module.", extra={"module_name": name})
            raise KernelError(f"Module '{name}' not found.") from exc

    def list_modules(self) -> tuple[str, ...]:
        return self._registry.list_modules()

    def list_services(self) -> tuple[str, ...]:
        return self._services.list_services()

    async def publish_event(self, topic: str, payload: dict[str, Any] | None = None) -> None:
        await self._publish(topic, **(payload or {}))

    async def health(self) -> dict[str, Any]:
        report = {
            "kernel": {
                "name": self._config.name,
                "version": self._config.version,
                "state": self.state.name,
                "uptime_seconds": (time.time() - self._context.started_at if self._context.started_at else 0.0),
                "restart_count": self._context.restart_count,
            },
            "registry": {
                "registered_modules": len(self._registry.list_modules()),
            },
            "services": {
                "registered_services": len(self._services.list_services()),
            },
        }

        if self._health_monitor is not None:
            try:
                # HealthMonitor.health()/diagnostics() are blocking (psutil
                # sampling) -- run off the event loop thread.
                report["monitor"] = await asyncio.to_thread(self._health_monitor.diagnostics)
            except Exception:
                logger.exception("HealthMonitor failed.")
                report["monitor"] = {"status": "FAILED"}

        return report

    def resolve_service(self, service_type: type[Any]) -> Any:
        return self._services.resolve(service_type)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "kernel_state": self._context.state.name,
            "registered_modules": self._registry.list_modules(),
            "registered_services": self._services.list_services(),
            "boot_timestamp": self._context.boot_timestamp,
            "restart_count": self._context.restart_count,
        }

    async def demo(self) -> dict[str, Any]:
        """
        Official StarkOS demonstration: exercises Identity, HealthMonitor,
        module/service introspection and the EventBus in one pass, and
        returns a structured report the CLI (or main.py's --demo flag)
        can render.
        """
        logger.info("Running Kernel demonstration.")
        await self._publish("kernel.demo.started")

        identity_report: dict[str, Any] = {"available": False}
        identity = self._services.resolve_optional(Identity)
        if identity is not None:
            response = identity.greet()
            identity_report = {
                "available": True,
                "message": response.text,
                "suggestions": list(response.suggestions),
            }

        if self._health_monitor is not None:
            health_report = await asyncio.to_thread(self._health_monitor.diagnostics)
        else:
            health_report = {"available": False}

        report = {
            "kernel": {
                "name": self._config.name,
                "version": self._config.version,
                "state": self.state.name,
            },
            "modules": self.list_modules(),
            "services": self.list_services(),
            "identity": identity_report,
            "health": health_report,
            "diagnostics": self.diagnostics(),
        }

        await self._publish("kernel.demo.completed")
        logger.info("Kernel demonstration completed.")
        return report

    def _ensure_state(self, *allowed: KernelState) -> None:
        if self._context.state not in allowed:
            raise KernelStateError(
                f"Kernel state is {self._context.state.name}, expected one of {', '.join(s.name for s in allowed)}."
            )

    @property
    def is_running(self) -> bool:
        return self._context.state == KernelState.RUNNING