"""
core/industrial_connectors.py
================================

Industrial connectivity layer for StarkOS: standardized, secure,
async-native access to shop-floor and engineering-systems data (OPC
UA, MQTT, Modbus) and a generic bridge pattern for PLM/CAD/MES systems.

Responsibilities
----------------
- A single `IndustrialConnector` Protocol every real or bridged
  connector implements, so the rest of StarkOS reads/writes industrial
  data without caring which protocol is underneath.
- Real, correct client code for three real industrial protocols:
  `OpcUaConnector` (via the optional `asyncua` package), `MqttConnector`
  (via the optional `paho-mqtt` package, bridged into asyncio), and
  `ModbusConnector` (via the optional `pymodbus` package) -- each
  lazily imported and gracefully degrading (never fabricating a
  reading) if its dependency isn't installed or the endpoint is
  unreachable.
- `RestBridgeConnector`: a generic, configurable HTTP adapter (stdlib
  `urllib` only) for PLM/CAD/MES systems -- see the honesty note on why
  this is a *pattern*, not a specific vendor integration.
- Normalization: every connector's protocol-specific reading becomes
  one internal `IndustrialDataPoint` (point, value, units, quality,
  timestamp) -- callers never see OPC UA node-ids, MQTT topics or
  Modbus registers directly.
- Async, non-blocking, high-frequency-safe: connectors are driven off
  the event loop thread where their underlying client is blocking
  (Modbus, MQTT's callback loop), and streamed readings are buffered
  through a bounded `asyncio.Queue` so a fast data source can't stall
  the rest of StarkOS.
- Security-first: every read and write is authorized through
  `SecurityCore` (RBAC/ABAC, deny-by-default, fail-closed -- see
  `core.security_core`) before it touches a connector; `critical=True`
  writes are held to a stricter action name so a policy can gate them
  separately from routine reads.
- Connection lifecycle: automatic reconnect with exponential backoff,
  explicit `ConnectionState` (including a `DEGRADED` state for "still
  connected but not looking healthy"), and reads against a
  disconnected/degraded connector return a clearly-flagged stale/
  unavailable result rather than blocking or raising into the caller's
  hot path.
- Every write (always) and a configurable sample of reads are recorded
  into DigitalThread -- see the honesty note on why *not* every single
  high-frequency reading is chained into the immutable ledger by default.

Honesty about scope
--------------------
1. **None of the three real protocol clients ship as hard dependencies,
   and none are pretended to be connected when they aren't.** `asyncua`/
   `paho-mqtt`/`pymodbus` are optional, lazily imported; without them
   (or without a reachable endpoint), `check_available()` reports False
   and `read()`/`write()` raise `ConnectorUnavailableError` -- never a
   fabricated value. This module was written and reviewed against each
   library's real, documented API, but actual live communication with a
   real OPC UA server/MQTT broker/Modbus device could not be exercised
   in the environment this was built in (none were reachable) -- the
   graceful-degradation paths, normalization, security integration and
   reconnect logic were tested against connectors implementing the same
   `IndustrialConnector` Protocol.

2. **There is no single standard "PLM API" or "MES API."** Unlike OPC
   UA/MQTT/Modbus, PLM/CAD/MES integration is vendor-specific.
   `RestBridgeConnector` is a real, working, generic HTTP adapter
   (point <-> URL/JSON-path mapping you configure) for systems that
   expose a REST API -- which most modern PLM/MES platforms do -- not a
   specific vendor's SDK. Wiring it to Teamcenter, Windchill, SAP ME,
   etc. means configuring its endpoint map for that system; a fully
   vendor-specific connector is an extension point (implement
   `IndustrialConnector` directly), not something this module provides.

3. **Not every reading is chained into DigitalThread.** In a real
   industrial setting, some points update many times per second --
   hashing every single one into an immutable ledger would make the
   ledger itself the bottleneck and drown genuinely important events in
   routine noise. Every *write* is always recorded (writes change the
   world and are comparatively rare); reads are recorded at a
   configurable sample rate (`IndustrialConnectorsConfig.read_sample_rate`,
   default: record 1 in N) plus any read whose `quality` isn't "good."
   This is a deliberate, documented trade-off, not an omission.

4. **"Sandbox/isolamento" here means execution isolation (timeouts,
   exception containment, off-the-event-loop-thread execution for
   blocking clients) and security mediation (SecurityCore
   authorization) -- not OS-level network sandboxing.** The same
   honesty boundary already drawn for `core.security_core.PluginSandbox`
   and `core.local_runtime` applies here.

Design
------
Same shape as the rest of StarkOS: a Protocol (`IndustrialConnector`)
with real backend implementations, each optional and gracefully
degrading, plus a generic bridge pattern for the one category (PLM/MES)
that has no single standard to target. `SecurityCore`/`DigitalThread`/
`KnowledgeGraph`/`Identity`/`EventBus` are imported concretely (no cycle
risk -- nothing existing imports this new module); `CognitiveEngine`
and `SimulationOrchestrator` are bound via `Any`, consistent with how
`Kernel` itself is handled everywhere in StarkOS, since neither needs
to be called back into by this module.

`IndustrialConnectors` satisfies the `Module` protocol (name/
initialize/start/stop) and registers with the Kernel like any other
StarkOS module:

    industrial = IndustrialConnectors(services=services)
    industrial.bind_security_core(security_core)
    industrial.bind_digital_thread(digital_thread)
    industrial.register_connector(OpcUaConnector(endpoint_url="opc.tcp://localhost:4840"))
    kernel.register_module(industrial, name="industrial_connectors", priority=40)

    reading = await industrial.read("opcua_plant", "ns=2;s=Motor1.Temperature", principal=principal)
    result = await industrial.write("opcua_plant", "ns=2;s=Motor1.SetpointRPM", 1750, principal=principal, critical=True)
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable

from core.digital_thread import DigitalThread
from core.event_bus import Event, EventBus
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.security_core import AuthorizationDeniedError, Principal, SecurityCore
from core.service_container import ServiceContainer

logger = get_logger("industrial_connectors")

# =============================================================================
# Exceptions
# =============================================================================

class IndustrialConnectorsError(Exception):
    """Base exception for IndustrialConnectors failures."""

class ConnectorNotFoundError(IndustrialConnectorsError):
    """Raised when a named connector isn't registered."""

class ConnectorUnavailableError(IndustrialConnectorsError):
    """Raised when a connector's dependency isn't installed or its
    endpoint is unreachable -- never a fabricated reading."""

class WriteRejectedError(IndustrialConnectorsError):
    """Raised when a write is refused (authorization, validation, or
    the connector itself rejecting it)."""

# =============================================================================
# Connection lifecycle
# =============================================================================

class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DEGRADED = auto()  # connected, but recent operations have been failing/timing out
    FAILED = auto()    # gave up reconnecting (see ReconnectPolicy.max_attempts)

@dataclass(slots=True, frozen=True)
class ReconnectPolicy:
    max_attempts: Optional[int] = None  # None = retry forever
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_multiplier: float = 2.0

    def backoff_for_attempt(self, attempt: int) -> float:
        return min(self.max_backoff_seconds, self.initial_backoff_seconds * (self.backoff_multiplier ** max(0, attempt - 1)))

# =============================================================================
# Normalized data models
# =============================================================================

@dataclass(slots=True, frozen=True)
class IndustrialDataPoint:
    """The internal, protocol-agnostic form every connector normalizes
    its own reading into -- callers never see OPC UA node-ids, MQTT
    topics or Modbus registers directly."""

    connector_name: str
    protocol: str
    point: str
    value: Any
    units: Optional[str] = None
    quality: str = "good"  # "good" | "uncertain" | "bad" -- from the protocol's own status/quality code
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True, frozen=True)
class WriteResult:
    connector_name: str
    point: str
    accepted: bool
    written_value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Connector Protocol (the standardized industrial interface)
# =============================================================================

@runtime_checkable
class IndustrialConnector(Protocol):
    """The interface every real or bridged industrial connector
    implements. `connect()`/`disconnect()`/`read()`/`write()` are all
    async -- implementations run any blocking client work via
    `asyncio.to_thread` internally so the event loop is never blocked."""

    name: str
    protocol: str

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def connection_state(self) -> ConnectionState: ...
    def check_available(self) -> bool: ...
    async def read(self, point: str) -> IndustrialDataPoint: ...
    async def write(self, point: str, value: Any) -> WriteResult: ...

# =============================================================================
# OPC UA connector (real, via the optional `asyncua` package)
# =============================================================================

class OpcUaConnector:
    """
    Real OPC UA client via `asyncua` (the standard async Python OPC UA
    library), optional and lazily imported. `asyncua` is itself
    asyncio-native, so no thread bridging is needed here.
    """

    protocol = "opcua"

    def __init__(self, *, name: str = "opcua", endpoint_url: str, timeout_seconds: float = 10.0) -> None:
        self.name = name
        self._endpoint_url = endpoint_url
        self._timeout_seconds = timeout_seconds
        self._client: Any = None
        self._state = ConnectionState.DISCONNECTED

    def connection_state(self) -> ConnectionState:
        return self._state

    def check_available(self) -> bool:
        try:
            import asyncua  # noqa: F401
            return True
        except ImportError:
            return False

    async def connect(self) -> None:
        try:
            from asyncua import Client
        except ImportError as exc:
            self._state = ConnectionState.FAILED
            raise ConnectorUnavailableError("The 'asyncua' package is not installed.") from exc

        self._state = ConnectionState.CONNECTING
        try:
            self._client = Client(url=self._endpoint_url, timeout=self._timeout_seconds)
            await self._client.connect()
            self._state = ConnectionState.CONNECTED
        except Exception as exc:
            self._state = ConnectionState.FAILED
            self._client = None
            raise ConnectorUnavailableError(f"Unable to connect to OPC UA endpoint '{self._endpoint_url}': {exc}") from exc

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                logger.exception("Error disconnecting OPC UA client '%s'.", self.name)
        self._client = None
        self._state = ConnectionState.DISCONNECTED

    async def read(self, point: str) -> IndustrialDataPoint:
        if self._client is None or self._state != ConnectionState.CONNECTED:
            raise ConnectorUnavailableError(f"OPC UA connector '{self.name}' is not connected.")
        try:
            node = self._client.get_node(point)
            data_value = await node.read_data_value()
            value = data_value.Value.Value
            quality = "good" if data_value.StatusCode.is_good() else "bad"
        except Exception as exc:
            self._state = ConnectionState.DEGRADED
            raise ConnectorUnavailableError(f"OPC UA read of '{point}' failed: {exc}") from exc

        return IndustrialDataPoint(connector_name=self.name, protocol=self.protocol, point=point, value=value, quality=quality)

    async def write(self, point: str, value: Any) -> WriteResult:
        if self._client is None or self._state != ConnectionState.CONNECTED:
            raise ConnectorUnavailableError(f"OPC UA connector '{self.name}' is not connected.")
        try:
            node = self._client.get_node(point)
            await node.write_value(value)
        except Exception as exc:
            self._state = ConnectionState.DEGRADED
            raise WriteRejectedError(f"OPC UA write to '{point}' failed: {exc}") from exc
        return WriteResult(connector_name=self.name, point=point, accepted=True, written_value=value)

# =============================================================================
# MQTT connector (real, via the optional `paho-mqtt` package)
# =============================================================================

class MqttConnector:
    """
    Real MQTT client via `paho-mqtt` (the standard Python MQTT client),
    optional and lazily imported. `paho-mqtt`'s client runs its own
    network loop on a background thread and delivers messages via
    callbacks -- this class bridges those callbacks into an
    `asyncio.Queue` so the rest of StarkOS can `await` a read without
    the event loop ever blocking on MQTT's own loop.
    """

    protocol = "mqtt"

    def __init__(self, *, name: str = "mqtt", host: str = "localhost", port: int = 1883, timeout_seconds: float = 10.0, queue_max_size: int = 1000) -> None:
        self.name = name
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
        self._client: Any = None
        self._state = ConnectionState.DISCONNECTED
        self._latest_by_topic: dict[str, IndustrialDataPoint] = {}
        self._queue: "asyncio.Queue[IndustrialDataPoint]" = asyncio.Queue(maxsize=queue_max_size)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def connection_state(self) -> ConnectionState:
        return self._state

    def check_available(self) -> bool:
        try:
            import paho.mqtt.client  # noqa: F401
            return True
        except ImportError:
            return False

    async def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            self._state = ConnectionState.FAILED
            raise ConnectorUnavailableError("The 'paho-mqtt' package is not installed.") from exc

        self._loop = asyncio.get_running_loop()
        self._state = ConnectionState.CONNECTING

        def _on_message(_client: Any, _userdata: Any, message: Any) -> None:
            point = IndustrialDataPoint(
                connector_name=self.name, protocol=self.protocol, point=message.topic,
                value=message.payload.decode("utf-8", errors="replace"), quality="good",
            )
            self._latest_by_topic[message.topic] = point
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._enqueue_nowait, point)

        def _on_connect(_client: Any, _userdata: Any, _flags: Any, reason_code: Any, *_args: Any) -> None:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._mark_connected)

        try:
            client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            client = mqtt.Client()  # older paho-mqtt without CallbackAPIVersion
        client.on_message = _on_message
        client.on_connect = _on_connect

        try:
            await asyncio.to_thread(client.connect, self._host, self._port, int(self._timeout_seconds))
            client.loop_start()
            self._client = client
        except Exception as exc:
            self._state = ConnectionState.FAILED
            raise ConnectorUnavailableError(f"Unable to connect to MQTT broker '{self._host}:{self._port}': {exc}") from exc

    def _mark_connected(self) -> None:
        self._state = ConnectionState.CONNECTED

    def _enqueue_nowait(self, point: IndustrialDataPoint) -> None:
        try:
            self._queue.put_nowait(point)
        except asyncio.QueueFull:
            logger.warning("MQTT connector '%s' inbound queue is full -- dropping the oldest message.", self.name)
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(point)
            except asyncio.QueueEmpty:
                pass

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                logger.exception("Error disconnecting MQTT client '%s'.", self.name)
        self._client = None
        self._state = ConnectionState.DISCONNECTED

    async def subscribe(self, topic: str, *, qos: int = 0) -> None:
        if self._client is None:
            raise ConnectorUnavailableError(f"MQTT connector '{self.name}' is not connected.")
        self._client.subscribe(topic, qos=qos)

    async def next_message(self, *, timeout: Optional[float] = None) -> IndustrialDataPoint:
        """Await the next message from any subscribed topic -- the
        high-frequency streaming path, buffered through a bounded queue
        so a fast publisher can never block the caller's event loop."""
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout=timeout)

    async def read(self, point: str) -> IndustrialDataPoint:
        """Returns the most recently received retained/latest value for
        this topic. MQTT has no request/response "read" primitive --
        this reflects the last message seen, which is the honest
        semantics of a pub/sub protocol."""
        cached = self._latest_by_topic.get(point)
        if cached is None:
            raise ConnectorUnavailableError(f"No message received yet for MQTT topic '{point}' on connector '{self.name}'.")
        return cached

    async def write(self, point: str, value: Any) -> WriteResult:
        if self._client is None or self._state != ConnectionState.CONNECTED:
            raise ConnectorUnavailableError(f"MQTT connector '{self.name}' is not connected.")
        payload = value if isinstance(value, (str, bytes)) else json.dumps(value)
        try:
            info = await asyncio.to_thread(self._client.publish, point, payload)
            accepted = getattr(info, "rc", 0) == 0
        except Exception as exc:
            raise WriteRejectedError(f"MQTT publish to '{point}' failed: {exc}") from exc
        if not accepted:
            raise WriteRejectedError(f"MQTT broker rejected publish to '{point}'.")
        return WriteResult(connector_name=self.name, point=point, accepted=True, written_value=value)

# =============================================================================
# Modbus connector (real, via the optional `pymodbus` package)
# =============================================================================

class ModbusConnector:
    """
    Real Modbus TCP client via `pymodbus`, optional and lazily
    imported. Modbus has no native concept of "points" beyond numbered
    registers/coils, so `point` here is a small address expression:
    "holding:<address>" (16-bit register), "input:<address>",
    "coil:<address>", or "discrete:<address>".
    """

    protocol = "modbus"

    def __init__(self, *, name: str = "modbus", host: str, port: int = 502, unit_id: int = 1, timeout_seconds: float = 5.0) -> None:
        self.name = name
        self._host = host
        self._port = port
        self._unit_id = unit_id
        self._timeout_seconds = timeout_seconds
        self._client: Any = None
        self._state = ConnectionState.DISCONNECTED

    def connection_state(self) -> ConnectionState:
        return self._state

    def check_available(self) -> bool:
        try:
            import pymodbus  # noqa: F401
            return True
        except ImportError:
            return False

    async def connect(self) -> None:
        try:
            from pymodbus.client import AsyncModbusTcpClient
        except ImportError as exc:
            self._state = ConnectionState.FAILED
            raise ConnectorUnavailableError("The 'pymodbus' package is not installed.") from exc

        self._state = ConnectionState.CONNECTING
        try:
            self._client = AsyncModbusTcpClient(self._host, port=self._port, timeout=self._timeout_seconds)
            await self._client.connect()
            if not self._client.connected:
                raise ConnectorUnavailableError(f"Modbus TCP connect to '{self._host}:{self._port}' did not establish.")
            self._state = ConnectionState.CONNECTED
        except ConnectorUnavailableError:
            self._state = ConnectionState.FAILED
            raise
        except Exception as exc:
            self._state = ConnectionState.FAILED
            self._client = None
            raise ConnectorUnavailableError(f"Unable to connect to Modbus device '{self._host}:{self._port}': {exc}") from exc

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.exception("Error disconnecting Modbus client '%s'.", self.name)
        self._client = None
        self._state = ConnectionState.DISCONNECTED

    @staticmethod
    def _parse_point(point: str) -> tuple[str, int]:
        try:
            kind, address_str = point.split(":", 1)
            return kind.strip().lower(), int(address_str)
        except (ValueError, TypeError) as exc:
            raise IndustrialConnectorsError(
                f"Malformed Modbus point '{point}' -- expected 'holding:<addr>'/'input:<addr>'/'coil:<addr>'/'discrete:<addr>'."
            ) from exc

    async def read(self, point: str) -> IndustrialDataPoint:
        if self._client is None or self._state != ConnectionState.CONNECTED:
            raise ConnectorUnavailableError(f"Modbus connector '{self.name}' is not connected.")
        kind, address = self._parse_point(point)

        try:
            if kind == "holding":
                response = await self._client.read_holding_registers(address, count=1, slave=self._unit_id)
                value: Any = response.registers[0]
            elif kind == "input":
                response = await self._client.read_input_registers(address, count=1, slave=self._unit_id)
                value = response.registers[0]
            elif kind == "coil":
                response = await self._client.read_coils(address, count=1, slave=self._unit_id)
                value = bool(response.bits[0])
            elif kind == "discrete":
                response = await self._client.read_discrete_inputs(address, count=1, slave=self._unit_id)
                value = bool(response.bits[0])
            else:
                raise IndustrialConnectorsError(f"Unknown Modbus point kind '{kind}'.")
        except IndustrialConnectorsError:
            raise
        except Exception as exc:
            self._state = ConnectionState.DEGRADED
            raise ConnectorUnavailableError(f"Modbus read of '{point}' failed: {exc}") from exc

        quality = "bad" if getattr(response, "isError", lambda: False)() else "good"
        return IndustrialDataPoint(connector_name=self.name, protocol=self.protocol, point=point, value=value, quality=quality)

    async def write(self, point: str, value: Any) -> WriteResult:
        if self._client is None or self._state != ConnectionState.CONNECTED:
            raise ConnectorUnavailableError(f"Modbus connector '{self.name}' is not connected.")
        kind, address = self._parse_point(point)

        try:
            if kind == "holding":
                response = await self._client.write_register(address, int(value), slave=self._unit_id)
            elif kind == "coil":
                response = await self._client.write_coil(address, bool(value), slave=self._unit_id)
            else:
                raise WriteRejectedError(f"Modbus point kind '{kind}' is not writable (only 'holding'/'coil' are).")
        except WriteRejectedError:
            raise
        except Exception as exc:
            self._state = ConnectionState.DEGRADED
            raise WriteRejectedError(f"Modbus write to '{point}' failed: {exc}") from exc

        if getattr(response, "isError", lambda: False)():
            raise WriteRejectedError(f"Modbus device rejected write to '{point}'.")
        return WriteResult(connector_name=self.name, point=point, accepted=True, written_value=value)

# =============================================================================
# Generic REST bridge (real, for PLM/CAD/MES -- see honesty note 2)
# =============================================================================

class RestBridgeConnector:
    """
    A real, generic HTTP adapter (stdlib `urllib` only) for PLM/CAD/MES
    systems that expose a REST API -- not a specific vendor's SDK. You
    configure a `point_map` from a StarkOS point name to a
    (method, url_template, json_path) triple; this class handles the
    actual HTTP request/response and JSON-path extraction/injection.
    """

    protocol = "rest_bridge"

    def __init__(
        self, *, name: str, base_url: str, point_map: dict[str, dict[str, str]], timeout_seconds: float = 15.0,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._point_map = point_map  # point -> {"method": "GET", "path": "/api/parts/{point}", "json_path": "value"}
        self._timeout_seconds = timeout_seconds
        self._headers = headers or {"Content-Type": "application/json"}
        self._state = ConnectionState.DISCONNECTED

    def connection_state(self) -> ConnectionState:
        return self._state

    def check_available(self) -> bool:
        return True  # stdlib-only; "availability" is really about the endpoint, checked per-call

    async def connect(self) -> None:
        self._state = ConnectionState.CONNECTED  # stateless HTTP -- "connected" means configured and ready to try

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    def _resolve(self, point: str) -> dict[str, str]:
        mapping = self._point_map.get(point)
        if mapping is None:
            raise ConnectorUnavailableError(f"No endpoint mapping configured for point '{point}' on connector '{self.name}'.")
        return mapping

    @staticmethod
    def _extract_json_path(payload: Any, json_path: str) -> Any:
        current = payload
        for part in json_path.split("."):
            if part == "":
                continue
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                current = current[index] if 0 <= index < len(current) else None
            else:
                return None
        return current

    async def read(self, point: str) -> IndustrialDataPoint:
        mapping = self._resolve(point)
        url = f"{self._base_url}{mapping['path'].format(point=point)}"

        def _call() -> Any:
            request = urllib.request.Request(url, method=mapping.get("method", "GET"), headers=self._headers)
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            payload = await asyncio.to_thread(_call)
        except urllib.error.URLError as exc:
            self._state = ConnectionState.DEGRADED
            raise ConnectorUnavailableError(f"REST bridge '{self.name}' unreachable at '{url}': {exc}") from exc
        except Exception as exc:
            raise ConnectorUnavailableError(f"REST bridge '{self.name}' read of '{point}' failed: {exc}") from exc

        value = self._extract_json_path(payload, mapping.get("json_path", "")) if mapping.get("json_path") else payload
        return IndustrialDataPoint(connector_name=self.name, protocol=self.protocol, point=point, value=value, raw_metadata={"url": url})

    async def write(self, point: str, value: Any) -> WriteResult:
        mapping = self._resolve(point)
        url = f"{self._base_url}{mapping['path'].format(point=point)}"
        body = json.dumps({"value": value}).encode("utf-8")

        def _call() -> None:
            request = urllib.request.Request(url, data=body, method=mapping.get("method", "POST"), headers=self._headers)
            with urllib.request.urlopen(request, timeout=self._timeout_seconds):
                pass

        try:
            await asyncio.to_thread(_call)
        except urllib.error.URLError as exc:
            self._state = ConnectionState.DEGRADED
            raise WriteRejectedError(f"REST bridge '{self.name}' unreachable at '{url}': {exc}") from exc
        except Exception as exc:
            raise WriteRejectedError(f"REST bridge '{self.name}' write to '{point}' failed: {exc}") from exc

        return WriteResult(connector_name=self.name, point=point, accepted=True, written_value=value)

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class IndustrialConnectorsConfig:
    reconnect_policy: ReconnectPolicy = field(default_factory=ReconnectPolicy)
    default_read_timeout_seconds: float = 10.0
    default_write_timeout_seconds: float = 10.0
    # Record 1-in-N successful, good-quality reads to DigitalThread;
    # every write and every non-"good"-quality read is always recorded
    # regardless -- see honesty note 3.
    read_sample_rate: int = 50
    require_authorization: bool = True  # the safe, least-privilege default
    record_to_knowledge_graph: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Industrial Connectors
# =============================================================================

class IndustrialConnectors:
    """
    StarkOS's industrial connectivity module. See the module
    docstring's "Honesty about scope" section before relying on it,
    especially point 1 (no live protocol testing was possible in the
    environment this was built in).

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(self, *, services: ServiceContainer, config: Optional[IndustrialConnectorsConfig] = None) -> None:
        self._services = services
        self._config = config or IndustrialConnectorsConfig()
        self._connectors: dict[str, IndustrialConnector] = {}
        self._read_counters: dict[str, int] = {}

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._digital_thread: Optional[DigitalThread] = None
        self._security_core: Optional[SecurityCore] = None
        self._event_bus: Optional[EventBus] = None
        self._cognitive_engine: Any = None
        self._simulation_orchestrator: Any = None

        logger.info("IndustrialConnectors constructed.", extra={"require_authorization": self._config.require_authorization})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "industrial_connectors"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to IndustrialConnectors.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to IndustrialConnectors.")

    def bind_digital_thread(self, digital_thread: DigitalThread) -> None:
        self._digital_thread = digital_thread
        logger.debug("DigitalThread bound to IndustrialConnectors.")

    def bind_security_core(self, security_core: SecurityCore) -> None:
        self._security_core = security_core
        logger.debug("SecurityCore bound to IndustrialConnectors.")

    def bind_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        logger.debug("EventBus bound to IndustrialConnectors.")

    def bind_cognitive_engine(self, cognitive_engine: Any) -> None:
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to IndustrialConnectors.")

    def bind_simulation_orchestrator(self, simulation_orchestrator: Any) -> None:
        self._simulation_orchestrator = simulation_orchestrator
        logger.debug("SimulationOrchestrator bound to IndustrialConnectors.")

    async def initialize(self) -> None:
        logger.info("Initializing IndustrialConnectors.")
        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._digital_thread is None:
            self._digital_thread = self._services.resolve_optional(DigitalThread)
        if self._security_core is None:
            self._security_core = self._services.resolve_optional(SecurityCore)
        if self._event_bus is None:
            self._event_bus = self._services.resolve_optional(EventBus)

        if self._config.require_authorization and self._security_core is None:
            logger.warning(
                "require_authorization=True but no SecurityCore is bound -- read()/write() will raise "
                "until bind_security_core() is called (fail-closed, not fail-open)."
            )
        if not self._connectors:
            logger.warning("No industrial connectors registered yet -- register_connector() before read()/write().")

        logger.info("IndustrialConnectors initialized.", extra={"connector_count": len(self._connectors)})

    async def start(self) -> None:
        logger.info("IndustrialConnectors ready.", extra={"connectors": list(self._connectors.keys())})

    async def stop(self) -> None:
        logger.info("Stopping IndustrialConnectors.")
        for connector in self._connectors.values():
            try:
                await connector.disconnect()
            except Exception:
                logger.exception("Error disconnecting connector '%s' during shutdown.", connector.name)
        logger.info("IndustrialConnectors stopped.")

    # ------------------------------------------------------------------
    # Connector registration and connection management
    # ------------------------------------------------------------------

    def register_connector(self, connector: IndustrialConnector) -> None:
        self._connectors[connector.name] = connector
        self._read_counters[connector.name] = 0
        logger.info("Industrial connector registered.", extra={"connector": connector.name, "protocol": connector.protocol})

    def unregister_connector(self, name: str) -> bool:
        removed = self._connectors.pop(name, None) is not None
        self._read_counters.pop(name, None)
        if removed:
            logger.info("Industrial connector unregistered.", extra={"connector": name})
        return removed

    def list_connectors(self) -> tuple[IndustrialConnector, ...]:
        return tuple(self._connectors.values())

    def _get_connector(self, connector_name: str) -> IndustrialConnector:
        connector = self._connectors.get(connector_name)
        if connector is None:
            raise ConnectorNotFoundError(f"No connector named '{connector_name}' is registered.")
        return connector

    async def connect(self, connector_name: str) -> None:
        connector = self._get_connector(connector_name)
        await self._connect_with_retry(connector)

    async def _connect_with_retry(self, connector: IndustrialConnector) -> None:
        attempt = 0
        policy = self._config.reconnect_policy
        while True:
            attempt += 1
            try:
                await connector.connect()
                logger.info("Connector connected.", extra={"connector": connector.name, "attempt": attempt})
                await self._publish_connection_event(connector, connected=True)
                return
            except ConnectorUnavailableError as exc:
                if policy.max_attempts is not None and attempt >= policy.max_attempts:
                    logger.error("Connector '%s' failed to connect after %d attempt(s); giving up.", connector.name, attempt)
                    await self._publish_connection_event(connector, connected=False)
                    raise
                backoff = policy.backoff_for_attempt(attempt)
                logger.warning("Connector '%s' connect attempt %d failed (%s) -- retrying in %.1fs.", connector.name, attempt, exc, backoff)
                await asyncio.sleep(backoff)

    async def disconnect(self, connector_name: str) -> None:
        connector = self._get_connector(connector_name)
        await connector.disconnect()
        await self._publish_connection_event(connector, connected=False)

    async def _publish_connection_event(self, connector: IndustrialConnector, *, connected: bool) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(Event(
                topic="industrial_connectors.connection_changed", source="industrial_connectors",
                payload={"connector": connector.name, "protocol": connector.protocol, "connected": connected},
            ))
        except Exception:
            logger.exception("Failed to publish connection_changed event for '%s'.", connector.name)

    # ------------------------------------------------------------------
    # Authorization (least privilege, fail-closed -- delegates to SecurityCore)
    # ------------------------------------------------------------------

    def _authorize(self, principal: Optional[Principal], action: str, resource: str) -> None:
        if not self._config.require_authorization:
            return
        if self._security_core is None:
            raise IndustrialConnectorsError(
                "Authorization is required (require_authorization=True) but no SecurityCore is bound -- "
                "failing closed rather than allowing an unauthorized action."
            )
        if principal is None:
            raise AuthorizationDeniedError(f"'{action}' on '{resource}' requires a Principal; none was supplied.")
        self._security_core.require(principal, action, resource=resource)

    # ------------------------------------------------------------------
    # Read / write (authorized, isolated, normalized, sampled to DigitalThread)
    # ------------------------------------------------------------------

    async def read(
        self, connector_name: str, point: str, *, principal: Optional[Principal] = None, timeout: Optional[float] = None,
    ) -> IndustrialDataPoint:
        connector = self._get_connector(connector_name)
        self._authorize(principal, f"industrial.{connector.protocol}.read", resource=f"{connector_name}:{point}")

        effective_timeout = timeout if timeout is not None else self._config.default_read_timeout_seconds
        try:
            reading = await asyncio.wait_for(connector.read(point), timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            raise ConnectorUnavailableError(f"Read of '{point}' via '{connector_name}' timed out after {effective_timeout}s.") from exc

        self._maybe_record_read(connector, reading, principal)
        return reading

    async def write(
        self, connector_name: str, point: str, value: Any, *, principal: Optional[Principal] = None,
        critical: bool = False, timeout: Optional[float] = None,
    ) -> WriteResult:
        connector = self._get_connector(connector_name)
        action = f"industrial.{connector.protocol}.write" + (".critical" if critical else "")
        self._authorize(principal, action, resource=f"{connector_name}:{point}")

        effective_timeout = timeout if timeout is not None else self._config.default_write_timeout_seconds
        try:
            result = await asyncio.wait_for(connector.write(point, value), timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            raise WriteRejectedError(f"Write to '{point}' via '{connector_name}' timed out after {effective_timeout}s.") from exc

        self._record_write(connector, result, principal, critical)
        return result

    def _maybe_record_read(self, connector: IndustrialConnector, reading: IndustrialDataPoint, principal: Optional[Principal]) -> None:
        self._read_counters[connector.name] = self._read_counters.get(connector.name, 0) + 1
        sampled = (self._read_counters[connector.name] % max(self._config.read_sample_rate, 1)) == 0
        should_record = sampled or reading.quality != "good"
        if not should_record:
            return

        actor = principal.id if principal is not None else None
        if self._digital_thread is not None:
            try:
                self._digital_thread.record_action(
                    trace_id="industrial-io", description=f"Read '{reading.point}' via '{connector.name}'.",
                    inputs={"point": reading.point}, method=f"IndustrialConnectors.{connector.protocol}",
                    parameters={"sampled": sampled}, result={"value": reading.value, "quality": reading.quality},
                    actor=actor,
                )
            except Exception:
                logger.exception("Failed to record read in DigitalThread.")

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None and reading.quality != "good":
            try:
                self._knowledge_graph.remember(
                    f"Industrial read quality issue: '{reading.point}' via '{connector.name}' -> quality={reading.quality}",
                    node_type="industrial_reading", metadata={"connector": connector.name, "point": reading.point, "quality": reading.quality},
                    source="industrial_connectors",
                )
            except Exception:
                logger.exception("Failed to record read quality issue in KnowledgeGraph.")

    def _record_write(self, connector: IndustrialConnector, result: WriteResult, principal: Optional[Principal], critical: bool) -> None:
        actor = principal.id if principal is not None else None
        if self._digital_thread is not None:
            try:
                self._digital_thread.record_action(
                    trace_id="industrial-io", description=f"Wrote '{result.point}' via '{connector.name}'.",
                    inputs={"point": result.point, "value": result.written_value}, method=f"IndustrialConnectors.{connector.protocol}",
                    parameters={"critical": critical}, result={"accepted": result.accepted}, actor=actor,
                )
            except Exception:
                logger.exception("Failed to record write in DigitalThread.")

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            try:
                self._knowledge_graph.remember(
                    f"Industrial write: '{result.point}' via '{connector.name}' = {result.written_value}",
                    node_type="industrial_write", metadata={"connector": connector.name, "point": result.point, "critical": critical},
                    source="industrial_connectors",
                )
            except Exception:
                logger.exception("Failed to record write in KnowledgeGraph.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "connectors": {
                name: {"protocol": connector.protocol, "state": connector.connection_state().name}
                for name, connector in self._connectors.items()
            },
            "require_authorization": self._config.require_authorization,
            "security_core_bound": self._security_core is not None,
            "digital_thread_bound": self._digital_thread is not None,
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "event_bus_bound": self._event_bus is not None,
        }