"""
core/service_container.py
=========================

Dependency Injection (DI) container for StarkOS.

Responsibilities
----------------
- Register application services.
- Resolve dependencies by identifier or type.
- Manage service lifetimes.
- Provide a thread-safe foundation for dependency injection.

This module intentionally contains no business logic. It is designed
to be lightweight, deterministic and easily testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import RLock
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger("starkos.service_container")

T = TypeVar("T")

# ============================================================================
# Exceptions
# ============================================================================

class ServiceContainerError(Exception):
    """Base exception for all ServiceContainer failures."""

class DuplicateServiceError(ServiceContainerError):
    """Raised when attempting to register an existing service."""

class ServiceNotFoundError(ServiceContainerError):
    """Raised when a requested service cannot be resolved."""

class InvalidProviderError(ServiceContainerError):
    """Raised when an invalid provider is supplied."""

# ============================================================================
# Enums
# ============================================================================

class ServiceLifetime(Enum):
    """Lifetime policy for registered services."""

    SINGLETON = auto()
    FACTORY = auto()

# ============================================================================
# Protocols
# ============================================================================

@runtime_checkable
class ServiceFactory(Protocol[T]):
    """Factory capable of constructing a service instance."""

    def __call__(self) -> T:
        ...

@runtime_checkable
class Disposable(Protocol):
    """Optional lifecycle contract for services that own resources."""

    def close(self) -> None:
        ...

@runtime_checkable
class AsyncDisposable(Protocol):
    """Optional asynchronous lifecycle contract."""

    async def aclose(self) -> None:
        ...

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(slots=True)
class ServiceDescriptor(Generic[T]):
    """Immutable description of a registered service."""

    identifier: str
    service_type: type[T]
    lifetime: ServiceLifetime
    provider: Callable[[], T]
    instance: Optional[T] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Service Container
# ============================================================================

class ServiceContainer:
    """
    Thread-safe dependency injection container.

    Features
    --------
    - Registration by identifier or type.
    - Singleton and factory lifetimes.
    - Lazy singleton instantiation.
    - Thread-safe operations.
    - Strong typing.
    - Structured logging.
    """

    def __init__(self) -> None:
        self._services: Dict[str, ServiceDescriptor[Any]] = {}
        self._types: Dict[type[Any], str] = {}
        self._lock = RLock()

        logger.info("ServiceContainer initialized.")

    @staticmethod
    def _default_identifier(service_type: type[Any]) -> str:
        return service_type.__name__

    def _assert_not_registered(self, identifier: str) -> None:
        if identifier in self._services:
            raise DuplicateServiceError(f"Service '{identifier}' is already registered.")

    def _assert_provider(self, provider: Any) -> None:
        if not callable(provider):
            raise InvalidProviderError("Service provider must be callable.")

    # ============================================================================
    # Registration
    # ============================================================================

    def register_singleton(
        self,
        service_type: type[T],
        provider: ServiceFactory[T],
        *,
        identifier: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        service_id = identifier or self._default_identifier(service_type)

        with self._lock:
            self._assert_not_registered(service_id)
            self._assert_provider(provider)

            descriptor = ServiceDescriptor(
                identifier=service_id,
                service_type=service_type,
                lifetime=ServiceLifetime.SINGLETON,
                provider=provider,
                metadata=metadata or {},
            )

            self._services[service_id] = descriptor
            self._types[service_type] = service_id

            logger.info(
                "Singleton registered.",
                extra={"service": service_id, "type": service_type.__name__},
            )

    def register_factory(
        self,
        service_type: type[T],
        provider: ServiceFactory[T],
        *,
        identifier: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        service_id = identifier or self._default_identifier(service_type)

        with self._lock:
            self._assert_not_registered(service_id)
            self._assert_provider(provider)

            descriptor = ServiceDescriptor(
                identifier=service_id,
                service_type=service_type,
                lifetime=ServiceLifetime.FACTORY,
                provider=provider,
                metadata=metadata or {},
            )

            self._services[service_id] = descriptor
            self._types[service_type] = service_id

            logger.info(
                "Factory registered.",
                extra={"service": service_id, "type": service_type.__name__},
            )

    def register_instance(
        self,
        instance: T,
        *,
        identifier: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        service_type = type(instance)
        service_id = identifier or self._default_identifier(service_type)

        with self._lock:
            self._assert_not_registered(service_id)

            descriptor = ServiceDescriptor(
                identifier=service_id,
                service_type=service_type,
                lifetime=ServiceLifetime.SINGLETON,
                provider=lambda: instance,
                instance=instance,
                metadata=metadata or {},
            )

            self._services[service_id] = descriptor
            self._types[service_type] = service_id

            logger.info(
                "Instance registered.",
                extra={"service": service_id, "type": service_type.__name__},
            )

    # ============================================================================
    # Resolution
    # ============================================================================

    def resolve(self, target: str | type[T]) -> T:
        with self._lock:
            if isinstance(target, str):
                descriptor = self._services.get(target)
            else:
                identifier = self._types.get(target)
                if identifier is None:
                    raise ServiceNotFoundError(f"Service type '{target.__name__}' is not registered.")
                descriptor = self._services.get(identifier)

            if descriptor is None:
                raise ServiceNotFoundError(f"Service '{target}' is not registered.")

            try:
                if descriptor.lifetime is ServiceLifetime.FACTORY:
                    instance = descriptor.provider()
                    logger.debug("Factory instance created.", extra={"service": descriptor.identifier})
                    return instance

                # Lazy Singleton
                if descriptor.instance is None:
                    descriptor.instance = descriptor.provider()
                    logger.debug("Singleton instantiated.", extra={"service": descriptor.identifier})

                return descriptor.instance

            except Exception as exc:
                logger.exception("Service resolution failed.", extra={"service": descriptor.identifier})
                raise ServiceContainerError(f"Unable to resolve '{descriptor.identifier}'.") from exc

    def resolve_optional(self, target: str | type[T]) -> T | None:
        try:
            return self.resolve(target)
        except ServiceNotFoundError:
            logger.debug("Optional service unavailable.", extra={"target": target if isinstance(target, str) else target.__name__})
            return None

    # ============================================================================
    # Administration
    # ============================================================================

    def contains(self, target: str | type[Any]) -> bool:
        with self._lock:
            if isinstance(target, str):
                return target in self._services
            return target in self._types

    def remove(self, target: str | type[Any]) -> bool:
        with self._lock:
            if isinstance(target, str):
                identifier = target
            else:
                identifier = self._types.get(target)
                if identifier is None:
                    return False

            descriptor = self._services.pop(identifier, None)
            if descriptor is None:
                return False

            self._types.pop(descriptor.service_type, None)

            logger.info("Service removed.", extra={"service": identifier})
            return True

    def clear(self) -> None:
        with self._lock:
            total = len(self._services)
            self._services.clear()
            self._types.clear()
            logger.warning("ServiceContainer cleared.", extra={"removed_services": total})

    def list_services(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._services.keys())

    def describe_services(self) -> tuple[dict[str, Any], ...]:
        """
        Detailed, read-only snapshot of every registered service.

        Unlike ``list_services()`` (identifiers only), this exposes the
        lifetime policy and whether a singleton has already been
        instantiated. Used by HealthMonitor for accurate reporting.
        """
        with self._lock:
            return tuple(
                {
                    "identifier": descriptor.identifier,
                    "type": descriptor.service_type.__name__,
                    "lifetime": descriptor.lifetime.name,
                    "instantiated": descriptor.instance is not None,
                }
                for descriptor in self._services.values()
            )

    async def shutdown(self) -> None:
        logger.info("ServiceContainer shutdown started.")
        with self._lock:
            descriptors = list(self._services.values())

        for descriptor in reversed(descriptors):
            instance = descriptor.instance
            if instance is None:
                continue

            try:
                if isinstance(instance, AsyncDisposable):
                    await instance.aclose()
                elif isinstance(instance, Disposable):
                    instance.close()
            except Exception:
                logger.exception("Service shutdown failed.", extra={"service": descriptor.identifier})

        logger.info("ServiceContainer shutdown completed.")