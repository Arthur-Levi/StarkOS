"""
core/health_monitor.py
======================

System health monitoring infrastructure for StarkOS.

Responsibilities
----------------
* Aggregate runtime health information.
* Collect resource metrics.
* Monitor registered modules and services.
* Produce immutable health reports.
"""

from __future__ import annotations

import logging
import os
import platform
import time

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

from core.module_registry import ModuleRegistry, ModuleState
from core.service_container import ServiceContainer

logger = logging.getLogger(__name__)

# =============================================================================
# Exceptions
# =============================================================================

class HealthMonitorError(RuntimeError):
    """Base exception raised by HealthMonitor."""

class MetricCollectionError(HealthMonitorError):
    """Raised when a metric cannot be collected."""

# =============================================================================
# Health Status
# =============================================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# =============================================================================
# Resource Snapshot
# =============================================================================

@dataclass(slots=True, frozen=True)
class ResourceSnapshot:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    process_memory_mb: float
    uptime_seconds: float

# =============================================================================
# Module Health
# =============================================================================

@dataclass(slots=True, frozen=True)
class ModuleHealth:
    name: str
    initialized: bool
    running: bool
    message: str = ""

# =============================================================================
# Service Health
# =============================================================================

@dataclass(slots=True, frozen=True)
class ServiceHealth:
    service: str
    available: bool
    singleton: bool
    message: str = ""

# =============================================================================
# Health Report
# =============================================================================

@dataclass(slots=True, frozen=True)
class HealthReport:
    generated_at: datetime
    status: HealthStatus
    resources: ResourceSnapshot
    modules: tuple[ModuleHealth, ...]
    services: tuple[ServiceHealth, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Health Monitor
# =============================================================================

class HealthMonitor:
    """
    Aggregates runtime health information.

    The monitor does not own any infrastructure.
    It simply queries existing components and
    produces immutable reports.

    NOTE: `health()` performs blocking work (psutil sampling uses a
    0.1s CPU interval). Callers running inside an asyncio event loop
    should invoke it via `asyncio.to_thread(monitor.health)` rather
    than awaiting it directly -- it is intentionally a plain sync
    method, not a coroutine.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        registry: ModuleRegistry,
    ) -> None:
        self._services = services
        self._registry = registry
        self._started_at = time.perf_counter()

        logger.debug("HealthMonitor initialized.")

    @property
    def uptime(self) -> float:
        return time.perf_counter() - self._started_at

    def _safe_percent(self, value: float) -> float:
        if value < 0:
            return 0.0
        if value > 100:
            return 100.0
        return float(value)

    def _process(self):
        if psutil is None:
            raise MetricCollectionError("psutil is not installed.")
        try:
            return psutil.Process(os.getpid())
        except Exception as exc:
            raise MetricCollectionError("Unable to access current process.") from exc

    def _cpu_percent(self) -> float:
        if psutil is None:
            logger.warning("CPU metrics unavailable.")
            return 0.0
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            return self._safe_percent(cpu)
        except Exception:
            logger.exception("CPU metric failed.")
            return 0.0

    def _memory_percent(self) -> float:
        if psutil is None:
            return 0.0
        try:
            memory = psutil.virtual_memory()
            return self._safe_percent(memory.percent)
        except Exception:
            logger.exception("Memory metric failed.")
            return 0.0

    def _process_memory_mb(self) -> float:
        if psutil is None:
            return 0.0
        try:
            process = self._process()
            rss = process.memory_info().rss
            return rss / (1024 * 1024)
        except Exception:
            logger.exception("Process memory metric failed.")
            return 0.0

    def _collect_resources(self) -> ResourceSnapshot:
        logger.debug("Collecting resource snapshot.")
        snapshot = ResourceSnapshot(
            timestamp=datetime.utcnow(),
            cpu_percent=self._cpu_percent(),
            memory_percent=self._memory_percent(),
            process_memory_mb=self._process_memory_mb(),
            uptime_seconds=self.uptime,
        )
        logger.debug("Resource snapshot collected.")
        return snapshot

    def _collect_modules(self) -> tuple[ModuleHealth, ...]:
        logger.debug("Collecting module health.")
        modules: list[ModuleHealth] = []
        try:
            # NOTE: ModuleRegistry.list_modules() returns bare names (str);
            # we need the real descriptors (with .state) for accurate health,
            # so use list_descriptors() instead.
            descriptors = self._registry.list_descriptors()
        except Exception:
            logger.exception("Unable to query ModuleRegistry.")
            return ()

        for descriptor in descriptors:
            try:
                state = descriptor.state
                initialized = state in (ModuleState.INITIALIZED, ModuleState.RUNNING, ModuleState.STOPPED)
                running = state is ModuleState.RUNNING
                message = "OK" if running else f"State: {state.name}"
                modules.append(
                    ModuleHealth(
                        name=descriptor.name,
                        initialized=initialized,
                        running=running,
                        message=message,
                    )
                )
            except Exception:
                logger.exception("Module inspection failed: %s", getattr(descriptor, "name", "<unknown>"))
                modules.append(
                    ModuleHealth(
                        name=getattr(descriptor, "name", "<unknown>"),
                        initialized=False,
                        running=False,
                        message="Inspection failed",
                    )
                )

        return tuple(modules)

    def _collect_services(self) -> tuple[ServiceHealth, ...]:
        logger.debug("Collecting service health.")
        services: list[ServiceHealth] = []
        try:
            descriptors = self._services.describe_services()
        except Exception:
            logger.exception("Unable to inspect services.")
            return ()

        for entry in descriptors:
            try:
                services.append(
                    ServiceHealth(
                        service=entry["identifier"],
                        available=True,
                        singleton=(entry["lifetime"] == "SINGLETON"),
                        message="Instantiated" if entry["instantiated"] else "Registered (lazy)",
                    )
                )
            except Exception:
                logger.exception("Service inspection failed.")

        return tuple(services)

    def _classify_resources(self, snapshot: ResourceSnapshot) -> HealthStatus:
        if snapshot.cpu_percent >= 95 or snapshot.memory_percent >= 95:
            return HealthStatus.UNHEALTHY
        if snapshot.cpu_percent >= 80 or snapshot.memory_percent >= 80:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _overall_status(self, resources: ResourceSnapshot, modules: tuple[ModuleHealth, ...]) -> HealthStatus:
        status = self._classify_resources(resources)
        failed_modules = sum(1 for module in modules if not module.running)
        if failed_modules == 0:
            return status
        if failed_modules < 3:
            return HealthStatus.DEGRADED
        return HealthStatus.UNHEALTHY

    def _runtime_metadata(self) -> dict[str, Any]:
        return {
            "platform": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "psutil": psutil is not None,
        }

    def health(self) -> HealthReport:
        logger.info("Generating health report.")
        try:
            resources = self._collect_resources()
            modules = self._collect_modules()
            services = self._collect_services()
            status = self._overall_status(resources, modules)

            report = HealthReport(
                generated_at=datetime.utcnow(),
                status=status,
                resources=resources,
                modules=modules,
                services=services,
                metadata=self._runtime_metadata(),
            )

            logger.info("Health report generated (status=%s).", status.value)
            return report
        except Exception as exc:
            logger.exception("Health report failed.")
            raise MetricCollectionError("Unable to generate health report.") from exc

    def diagnostics(self) -> dict[str, Any]:
        report = self.health()
        return {
            "status": report.status.value,
            "cpu": report.resources.cpu_percent,
            "memory": report.resources.memory_percent,
            "process_memory_mb": round(report.resources.process_memory_mb, 2),
            "uptime": round(report.resources.uptime_seconds, 2),
            "modules": len(report.modules),
            "services": len(report.services),
            "metadata": report.metadata,
        }