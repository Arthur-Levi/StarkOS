"""
core/event_bus.py
=================

High-performance asynchronous event bus for StarkOS.

Responsibilities
----------------
- Publish events
- Register subscribers
- Dispatch synchronous and asynchronous handlers
- Maintain thread safety
- Provide observability hooks
- Enable loose coupling between modules

Design Principles
-----------------
- Thread-safe
- Async-first
- Low coupling
- Strong typing
- Production ready
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from threading import RLock
from typing import (
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)
from uuid import UUID, uuid4

logger = logging.getLogger("starkos.event_bus")

# =============================================================================
# Exceptions
# =============================================================================

class EventBusError(Exception):
    """Base exception for EventBus."""

class EventNotFoundError(EventBusError):
    """Raised when an event topic does not exist."""

class DuplicateSubscriptionError(EventBusError):
    """Raised when attempting to subscribe twice."""

class InvalidHandlerError(EventBusError):
    """Raised when a handler is invalid."""

# =============================================================================
# Priority
# =============================================================================

class EventPriority(IntEnum):
    """Event dispatch priority."""

    LOW = 10
    NORMAL = 50
    HIGH = 100
    CRITICAL = 1000

# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class SyncEventHandler(Protocol):
    def __call__(self, event: "Event") -> None:
        ...

@runtime_checkable
class AsyncEventHandler(Protocol):
    async def __call__(self, event: "Event") -> None:
        ...

EventHandler = Union[SyncEventHandler, AsyncEventHandler]

# =============================================================================
# Dataclasses
# =============================================================================

@dataclass(slots=True)
class Event:
    """Immutable event transported through the EventBus."""

    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    source: str = "unknown"
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass(slots=True)
class Subscription:
    """Represents a registered event handler."""

    topic: str
    handler: EventHandler
    priority: int = 0
    once: bool = False

# =============================================================================
# EventBus
# =============================================================================

class EventBus:
    """
    Central event dispatcher for StarkOS.
    """

    def __init__(self) -> None:
        self._subscriptions: DefaultDict[str, List[Subscription]] = defaultdict(list)
        self._lock = RLock()

        logger.info("EventBus initialized.")

    @staticmethod
    def _validate_handler(handler: EventHandler) -> None:
        if not callable(handler):
            raise InvalidHandlerError("Handler must be callable.")

    @staticmethod
    def _is_async(handler: EventHandler) -> bool:
        return inspect.iscoroutinefunction(handler)

    def _subscription_count(self) -> int:
        return sum(len(v) for v in self._subscriptions.values())

    # ============================================================================
    # Subscription Management
    # ============================================================================

    def subscribe(
        self,
        topic: str,
        handler: EventHandler,
        *,
        priority: int = 0,
        once: bool = False,
    ) -> Subscription:
        self._validate_handler(handler)

        with self._lock:
            for subscription in self._subscriptions[topic]:
                if subscription.handler is handler:
                    raise DuplicateSubscriptionError(f"Handler already subscribed to '{topic}'.")

            subscription = Subscription(
                topic=topic,
                handler=handler,
                priority=priority,
                once=once,
            )

            self._subscriptions[topic].append(subscription)
            self._subscriptions[topic].sort(key=lambda item: item.priority, reverse=True)

            logger.info(
                "Handler subscribed.",
                extra={"topic": topic, "priority": priority, "once": once, "subscriptions": self._subscription_count()},
            )

            return subscription

    def unsubscribe(self, subscription: Subscription) -> bool:
        with self._lock:
            handlers = self._subscriptions.get(subscription.topic)
            if handlers is None:
                return False

            try:
                handlers.remove(subscription)
            except ValueError:
                return False

            if not handlers:
                del self._subscriptions[subscription.topic]

            logger.info(
                "Handler unsubscribed.",
                extra={"topic": subscription.topic, "subscriptions": self._subscription_count()},
            )

            return True

    def clear(self, *, topic: str | None = None) -> None:
        with self._lock:
            if topic is None:
                total = self._subscription_count()
                self._subscriptions.clear()
                logger.warning("All subscriptions cleared.", extra={"removed": total})
                return

            removed = len(self._subscriptions.get(topic, ()))
            self._subscriptions.pop(topic, None)
            logger.info("Topic cleared.", extra={"topic": topic, "removed": removed})

    def list_topics(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._subscriptions.keys()))

    # ============================================================================
    # Event Publishing
    # ============================================================================

    async def publish(self, event: Event) -> None:
        logger.debug(
            "Publishing event.",
            extra={"topic": event.topic, "source": event.source, "priority": int(event.priority), "event_id": str(event.event_id)},
        )

        with self._lock:
            subscriptions = list(self._subscriptions.get(event.topic, ()))

        if not subscriptions:
            logger.debug("No subscribers for event.", extra={"topic": event.topic})
            return

        once_to_remove: list[Subscription] = []

        for subscription in subscriptions:
            try:
                if self._is_async(subscription.handler):
                    await subscription.handler(event)
                else:
                    subscription.handler(event)

                if subscription.once:
                    once_to_remove.append(subscription)

            except Exception:
                logger.exception(
                    "Event handler failed.",
                    extra={"topic": event.topic, "handler": getattr(subscription.handler, "__name__", repr(subscription.handler))},
                )

        for subscription in once_to_remove:
            self.unsubscribe(subscription)

        logger.debug("Event dispatch completed.", extra={"topic": event.topic, "handlers": len(subscriptions)})

    def publish_nowait(self, event: Event) -> asyncio.Task[None]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise EventBusError("publish_nowait() requires an active event loop.") from exc

        logger.debug("Scheduling asynchronous event.", extra={"topic": event.topic, "event_id": str(event.event_id)})
        return loop.create_task(self.publish(event))