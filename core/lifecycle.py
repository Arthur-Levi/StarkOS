"""
core/lifecycle.py
=================

Lifecycle orchestration for StarkOS.

Responsibilities
----------------
- Initialize registered modules.
- Coordinate startup and shutdown.
- Respect module priorities and dependencies.
- Provide deterministic lifecycle transitions.
- Remain independent from business logic.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from core.event_bus import Event, EventBus
from core.module_registry import Module, ModuleRegistry, ModuleState
from core.service_container import ServiceContainer, Disposable, AsyncDisposable

logger = logging.getLogger("starkos.lifecycle")

# =============================================================================
# Exceptions
# =============================================================================

class LifecycleError(Exception):
    """Base lifecycle exception."""

class InitializationError(LifecycleError):
    """Raised when module initialization fails."""

class StartupError(LifecycleError):
    """Raised when startup fails."""

class ShutdownError(LifecycleError):
    """Raised when shutdown fails."""

class DependencyResolutionError(LifecycleError):
    """Raised when module dependencies cannot be satisfied."""

# =============================================================================
# Lifecycle State
# =============================================================================

class LifecycleState(Enum):
    CREATED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()

class LifecyclePhase(Enum):
    INITIALIZE = auto()
    START = auto()
    STOP = auto()
    RESTART = auto()

# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class LifecycleAware(Protocol):
    async def before_initialize(self) -> None:
        ...

    async def after_initialize(self) -> None:
        ...

    async def before_shutdown(self) -> None:
        ...

    async def after_shutdown(self) -> None:
        ...

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class LifecycleConfig:
    parallel_startup: bool = False
    startup_timeout: float = 30.0
    shutdown_timeout: float = 30.0
    continue_on_error: bool = False
    emit_events: bool = True

# =============================================================================
# Context
# =============================================================================

@dataclass(slots=True)
class LifecycleContext:
    state: LifecycleState = LifecycleState.CREATED
    initialized_modules: list[str] = field(default_factory=list)
    started_modules: list[str] = field(default_factory=list)
    failed_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Execution Plan
# =============================================================================

@dataclass(slots=True)
class ExecutionPlan:
    startup_order: tuple[str, ...]
    shutdown_order: tuple[str, ...]
    parallel_batches: tuple[tuple[str, ...], ...]
    graph_hash: str

# =============================================================================
# Lifecycle Manager
# =============================================================================

class LifecycleManager:
    """
    Coordinates the lifecycle of all registered modules.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        services: ServiceContainer,
        *,
        event_publisher: EventBus | None = None,
        config: LifecycleConfig | None = None,
    ) -> None:
        self._registry = registry
        self._services = services
        self._event_bus = event_publisher
        self._config = config or LifecycleConfig()
        self._context = LifecycleContext()
        self._lock = RLock()
        self._plan: ExecutionPlan | None = None

        logger.info(
            "LifecycleManager initialized.",
            extra={
                "parallel_startup": self._config.parallel_startup,
                "startup_timeout": self._config.startup_timeout,
                "shutdown_timeout": self._config.shutdown_timeout,
            },
        )

    def _set_state(self, state: LifecycleState) -> None:
        self._context.state = state
        logger.debug("Lifecycle state changed.", extra={"state": state.name})

    def _registered_modules(self) -> Iterable[Module]:
        for module_name in self._registry.list_modules():
            yield self._registry.resolve_module(module_name)

    async def _invoke_hook(self, module: Module, hook: str) -> None:
        if not isinstance(module, LifecycleAware):
            return
        callback = getattr(module, hook, None)
        if callback is None:
            return
        await callback()

    async def _emit(self, topic: str, **payload: Any) -> None:
        """Publish a lifecycle event if an EventBus was supplied."""
        if self._event_bus is None or not self._config.emit_events:
            return
        try:
            await self._event_bus.publish(Event(topic=topic, source="LifecycleManager", payload=payload))
        except Exception:
            logger.exception("Failed to publish lifecycle event.", extra={"topic": topic})

    # ============================================================================
    # Dependency Graph
    # ============================================================================

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        # NOTE: dependencies live on ModuleDescriptor.dependencies, not inside
        # the free-form `metadata` dict -- read them from the real field via
        # the registry's descriptor accessor.
        graph: dict[str, set[str]] = {}
        for descriptor in self._registry.list_descriptors():
            graph[descriptor.name] = set(descriptor.dependencies)
        return graph

    def _priority_of(self, name: str) -> int:
        return self._registry.get_descriptor(name).priority

    def _topological_sort(self) -> list[str]:
        graph = self._build_dependency_graph()
        resolved: list[str] = []
        graph = {key: set(value) for key, value in graph.items()}

        while graph:
            ready = sorted(
                (name for name, deps in graph.items() if not deps),
                key=lambda name: self._priority_of(name),
            )
            if not ready:
                raise DependencyResolutionError("Circular dependency detected.")

            resolved.extend(ready)
            for name in ready:
                graph.pop(name)
            for deps in graph.values():
                deps.difference_update(ready)

        return resolved

    def _parallel_batches(self) -> list[list[str]]:
        graph = self._build_dependency_graph()
        batches: list[list[str]] = []
        graph = {key: set(value) for key, value in graph.items()}

        while graph:
            ready = sorted(
                (name for name, deps in graph.items() if not deps),
                key=lambda name: self._priority_of(name),
            )
            if not ready:
                raise DependencyResolutionError("Circular dependency detected.")

            batches.append(ready)
            for module in ready:
                graph.pop(module)
            for deps in graph.values():
                deps.difference_update(ready)

        return batches

    def _execution_plan(self) -> ExecutionPlan:
        if self._plan is not None:
            return self._plan

        startup = tuple(self._topological_sort())
        shutdown = tuple(reversed(startup))
        batches = tuple(tuple(batch) for batch in self._parallel_batches())

        self._plan = ExecutionPlan(
            startup_order=startup,
            shutdown_order=shutdown,
            parallel_batches=batches,
            graph_hash="placeholder",  # Replace with actual hash in production
        )

        logger.debug("Execution plan rebuilt.", extra={"modules": len(startup)})
        return self._plan

    def invalidate_execution_plan(self) -> None:
        self._plan = None
        logger.debug("Execution plan invalidated.")

    # ============================================================================
    # Initialization
    # ============================================================================

    async def initialize_modules(self) -> None:
        with self._lock:
            self._set_state(LifecycleState.INITIALIZING)

        logger.info("Module initialization started.")

        try:
            if not self._config.parallel_startup:
                order = self._topological_sort()
                for module_name in order:
                    descriptor = self._registry.get_descriptor(module_name)
                    module = descriptor.module
                    try:
                        await self._invoke_hook(module, "before_initialize")
                        await asyncio.wait_for(module.initialize(), timeout=self._config.startup_timeout)
                        self._registry.set_module_state(module_name, ModuleState.INITIALIZED)
                        self._context.initialized_modules.append(module_name)
                        await self._invoke_hook(module, "after_initialize")
                        logger.info("Module initialized.", extra={"module_name": module_name})
                    except Exception:
                        self._registry.set_module_state(module_name, ModuleState.FAILED)
                        self._context.failed_modules.append(module_name)
                        logger.exception("Initialization failed.", extra={"module_name": module_name})
                        if not self._config.continue_on_error:
                            raise
            else:
                for batch in self._parallel_batches():
                    tasks = []
                    descriptors = []
                    for module_name in batch:
                        descriptor = self._registry.get_descriptor(module_name)
                        descriptors.append(descriptor)
                        tasks.append(asyncio.wait_for(descriptor.module.initialize(), timeout=self._config.startup_timeout))

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for descriptor, result in zip(descriptors, results):
                        if isinstance(result, Exception):
                            self._registry.set_module_state(descriptor.name, ModuleState.FAILED)
                            self._context.failed_modules.append(descriptor.name)
                            logger.exception("Initialization failed.", extra={"module_name": descriptor.name})
                            if not self._config.continue_on_error:
                                raise result
                        else:
                            self._registry.set_module_state(descriptor.name, ModuleState.INITIALIZED)
                            self._context.initialized_modules.append(descriptor.name)
                            logger.info("Module initialized.", extra={"module_name": descriptor.name})

            self._set_state(LifecycleState.INITIALIZED)
            logger.info("Initialization completed.", extra={"modules": len(self._context.initialized_modules)})

        except Exception as exc:
            self._set_state(LifecycleState.FAILED)
            raise InitializationError("Lifecycle initialization failed.") from exc

    # ============================================================================
    # Startup
    # ============================================================================

    async def start_modules(self) -> None:
        """
        Start every INITIALIZED module, in dependency order, transitioning
        it to RUNNING. This is a distinct phase from ``initialize_modules()``
        because the Module protocol separates `initialize()` (wire up state)
        from `start()` (begin doing work).
        """
        with self._lock:
            self._set_state(LifecycleState.STARTING)

        logger.info("Module startup started.")

        try:
            order = self._topological_sort()
            for module_name in order:
                descriptor = self._registry.get_descriptor(module_name)
                if descriptor.state is not ModuleState.INITIALIZED:
                    # Skip modules that failed initialization or are already running.
                    continue

                module = descriptor.module
                try:
                    await asyncio.wait_for(module.start(), timeout=self._config.startup_timeout)
                    self._registry.set_module_state(module_name, ModuleState.RUNNING)
                    self._context.started_modules.append(module_name)
                    await self._emit("module.started", module=module_name)
                    logger.info("Module started.", extra={"module_name": module_name})
                except Exception:
                    self._registry.set_module_state(module_name, ModuleState.FAILED)
                    self._context.failed_modules.append(module_name)
                    logger.exception("Module startup failed.", extra={"module_name": module_name})
                    if not self._config.continue_on_error:
                        raise

            self._set_state(LifecycleState.RUNNING)
            logger.info("Module startup completed.", extra={"modules": len(self._context.started_modules)})

        except Exception as exc:
            self._set_state(LifecycleState.FAILED)
            raise StartupError("Lifecycle startup failed.") from exc

    # ============================================================================
    # Shutdown
    # ============================================================================

    async def shutdown_modules(self) -> None:
        logger.info("Lifecycle shutdown started.")

        with self._lock:
            self._set_state(LifecycleState.STOPPING)

        execution_order = list(reversed(self._topological_sort()))
        failures: list[str] = []

        for module_name in execution_order:
            descriptor = self._registry.get_descriptor(module_name)
            # A module can be disposed whether it only reached INITIALIZED
            # (start() was never called) or made it all the way to RUNNING.
            if descriptor.state not in (ModuleState.INITIALIZED, ModuleState.RUNNING):
                continue

            try:
                await self._shutdown_module(descriptor)
                self._registry.set_module_state(module_name, ModuleState.STOPPED)
                logger.info("Module stopped.", extra={"module_name": module_name})
            except Exception:
                failures.append(module_name)
                self._registry.set_module_state(module_name, ModuleState.FAILED)
                logger.exception("Module shutdown failed.", extra={"module_name": module_name})
                if not self._config.continue_on_error:
                    raise

        self._context.started_modules.clear()
        self._context.initialized_modules.clear()
        self._set_state(LifecycleState.STOPPED)
        logger.info("Lifecycle shutdown completed.", extra={"failures": len(failures)})

    async def _shutdown_module(self, descriptor) -> None:
        module = descriptor.module
        await self._invoke_hook(module, "before_shutdown")

        # The Module protocol exposes `stop()`, not `shutdown()` -- calling
        # the wrong name silently no-ops modules on shutdown.
        if hasattr(module, "stop"):
            await asyncio.wait_for(module.stop(), timeout=self._config.shutdown_timeout)

        await self._dispose_module(module)
        await self._invoke_hook(module, "after_shutdown")
        await self._emit("module.stopped", module=descriptor.name)

    async def _dispose_module(self, module) -> None:
        if isinstance(module, AsyncDisposable):
            await module.aclose()
        elif isinstance(module, Disposable):
            module.close()

    # ============================================================================
    # Restart
    # ============================================================================

    async def restart(self) -> None:
        logger.info("Lifecycle restart requested.")

        try:
            await self.shutdown_modules()
        except Exception:
            logger.exception("Restart aborted during shutdown.")
            raise

        self.invalidate_execution_plan()
        try:
            await self.initialize_modules()
            await self.start_modules()
        except Exception:
            logger.exception("Restart failed during initialization/startup.")
            raise

        logger.info("Lifecycle restart completed.")

    # ============================================================================
    # Public API
    # ============================================================================

    def notify_registry_changed(self) -> None:
        self.invalidate_execution_plan()
        logger.info("Registry topology changed.")