"""
core/module_registry.py
=======================

Central module registry for StarkOS.

Responsibilities
----------------
* Register modules
* Resolve modules
* Maintain module metadata
* Dependency validation
* Thread-safe access
* Integration with ServiceContainer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from core.service_container import ServiceContainer

logger = logging.getLogger("starkos.module_registry")

# =============================================================================
# Exceptions
# =============================================================================

class ModuleRegistryError(Exception):
    """Base registry exception."""

class DuplicateModuleError(ModuleRegistryError):
    """Raised when attempting to register an existing module."""

class ModuleNotFoundError(ModuleRegistryError):
    """Raised when resolving an unknown module."""

class InvalidModuleError(ModuleRegistryError):
    """Raised when an object does not satisfy the Module protocol."""

# =============================================================================
# Module State
# =============================================================================

class ModuleState(Enum):
    REGISTERED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    STOPPED = auto()
    FAILED = auto()

# =============================================================================
# Protocol
# =============================================================================

@runtime_checkable
class Module(Protocol):
    @property
    def name(self) -> str:
        ...

    async def initialize(self) -> None:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

# =============================================================================
# Descriptor
# =============================================================================

@dataclass(slots=True)
class ModuleDescriptor:
    name: str
    module: Module
    state: ModuleState = ModuleState.REGISTERED
    priority: int = 100
    dependencies: set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Module Registry
# =============================================================================

class ModuleRegistry:
    """
    Thread-safe registry responsible for tracking StarkOS modules.
    """

    def __init__(self, service_container: Optional[ServiceContainer] = None) -> None:
        self._services = service_container
        self._modules: Dict[str, ModuleDescriptor] = {}
        self._lock = RLock()
        self._listeners: list[Callable[[], None]] = []

        logger.info("ModuleRegistry initialized.")

    def _validate_module(self, module: Module) -> None:
        if module is None:
            raise InvalidModuleError("Module cannot be None.")
        if not isinstance(module, Module):
            raise InvalidModuleError(f"{module!r} does not implement Module.")
        if not module.name:
            raise InvalidModuleError("Module must have a valid name.")

    def _assert_unique(self, name: str) -> None:
        if name in self._modules:
            raise DuplicateModuleError(f"Module '{name}' already registered.")

    def _descriptor(self, name: str) -> ModuleDescriptor:
        try:
            return self._modules[name]
        except KeyError as exc:
            raise ModuleNotFoundError(f"Module '{name}' not found.") from exc

    def _notify_change(self) -> None:
        for callback in self._listeners:
            try:
                callback()
            except Exception:
                logger.exception("Registry listener failed.")

    # ============================================================================
    # Public API
    # ============================================================================

    def register_module(
        self,
        module: Module,
        *,
        name: str | None = None,
        priority: int = 100,
        dependencies: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._validate_module(module)

        module_name = name or module.name

        with self._lock:
            self._assert_unique(module_name)

            descriptor = ModuleDescriptor(
                name=module_name,
                module=module,
                priority=priority,
                dependencies=dependencies or set(),
                metadata=metadata or {},
            )

            self._modules[module_name] = descriptor

            logger.info(
                "Module registered.",
                extra={"module_name": module_name, "priority": priority},
            )

        self._notify_change()

    def unregister_module(self, name: str) -> bool:
        with self._lock:
            descriptor = self._modules.pop(name, None)
            if descriptor is None:
                return False

            logger.info("Module removed.", extra={"module_name": name})

        self._notify_change()
        return True

    def resolve_module(self, name: str) -> Module:
        with self._lock:
            descriptor = self._descriptor(name)
            logger.debug("Module resolved.", extra={"module_name": name})
            return descriptor.module

    def contains_module(self, name: str) -> bool:
        with self._lock:
            return name in self._modules

    def list_modules(self, *, ordered: bool = True) -> tuple[str, ...]:
        with self._lock:
            descriptors = list(self._modules.values())

        if ordered:
            descriptors.sort(key=lambda d: (d.priority, d.name))

        return tuple(descriptor.name for descriptor in descriptors)

    def get_module_metadata(self, name: str) -> dict[str, Any]:
        with self._lock:
            descriptor = self._descriptor(name)
            return dict(descriptor.metadata)

    def get_descriptor(self, name: str) -> ModuleDescriptor:
        """
        Public accessor for a module's full descriptor (state, priority,
        dependencies included). LifecycleManager uses this instead of
        reaching into the private ``_descriptor`` method.
        """
        with self._lock:
            return self._descriptor(name)

    def set_module_state(self, name: str, state: ModuleState) -> None:
        """Thread-safe transition of a module's lifecycle state."""
        with self._lock:
            descriptor = self._descriptor(name)
            descriptor.state = state
        logger.debug("Module state changed.", extra={"module_name": name, "state": state.name})

    def list_descriptors(self, *, ordered: bool = True) -> tuple[ModuleDescriptor, ...]:
        """
        Full descriptor snapshot (name, state, priority, dependencies) for
        every registered module. Used by LifecycleManager (dependency graph)
        and HealthMonitor (real per-module status) instead of guessing at
        the shape of plain module names.
        """
        with self._lock:
            descriptors = list(self._modules.values())

        if ordered:
            descriptors.sort(key=lambda d: (d.priority, d.name))

        return tuple(descriptors)

    def add_change_listener(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "registered_modules": len(self._modules),
            "modules": self.list_modules(),
        }