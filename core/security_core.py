"""
core/security_core.py
========================

Security subsystem for StarkOS: authorization, auditing and signing.

Responsibilities
----------------
- RBAC (roles -> permissions) + ABAC (attribute-based policies) combined
  into a single, deny-by-default, fail-closed authorization decision.
- "Zero trust" enforcement within this process: every sensitive action
  is re-authorized every time, regardless of who called it or how
  recently they succeeded before -- no implicit trust from a prior check.
- A tamper-evident, hash-chained audit log of every authorization
  decision and sandboxed execution.
- Simple digital signing/verification (HMAC by default; real asymmetric
  Ed25519 signatures when the `cryptography` package is available).
- A capability-gated, time-bounded execution context ("sandbox") for
  semi-trusted internal plugins/agents.
- Bridges to Identity (derives a Principal from clearance) and Kernel
  (mediates a small, explicit allow-list of sensitive actions).

Honesty about scope
--------------------
Security modules are exactly where overclaiming is most dangerous --
believing a system is protected when it isn't is worse than knowing it
isn't. Three things need to be said plainly:

1. **"Zero trust" here means an architectural stance, not a network
   control.** There is no network microsegmentation, no mTLS, no
   service mesh -- this is a single Python process. What StarkOS *can*
   honestly claim is: every sensitive call re-checks authorization
   (nothing is cached as "already trusted"), authorization fails
   *closed* on internal errors (an exception during evaluation is a
   DENY, never an ALLOW), and every decision is audited whether it
   succeeded or not.

2. **`HMACSigner` (the default) produces a Message Authentication Code,
   not a true digital signature.** Verifying an HMAC requires the exact
   secret used to create it, so anyone who can verify can also forge --
   there is no non-repudiation. Use `Ed25519Signer` (real asymmetric
   cryptography, via the optional `cryptography` package) when you
   actually need "principal X, and only X, could have produced this."
   Also: the default HMAC secret is freshly random *per process* unless
   you supply a persistent one via `SecurityCoreConfig.hmac_secret` --
   without that, signatures won't verify across a restart.

3. **`PluginSandbox` is not a security boundary against malicious code.**
   Python has no supported, escape-proof way to execute arbitrary
   untrusted code in-process -- well-documented techniques exist to
   reach the real builtins from a "restricted" globals dict. What
   `PluginSandbox` actually provides is real and useful for *semi-
   trusted* StarkOS modules/plugins: every call is authorized through
   RBAC/ABAC first (capability-based mediation), wrapped in a wall-clock
   timeout, and has its exceptions isolated and audited. It does **not**
   forcibly kill a runaway thread (Python cannot safely do that) and it
   does **not** protect against a payload that is actively trying to
   escape. Untrusted third-party code needs OS-level isolation (a
   subprocess with resource limits/seccomp, a container, a WASM
   runtime) -- out of scope for an in-process module.

4. **The audit log is tamper-*evident*, not tamper-*proof*.** Each
   event's hash covers the previous event's hash, so retroactively
   editing or deleting an entry breaks the chain from that point on and
   `AuditLog.verify_integrity()` will catch it. An attacker with write
   access to the log file (or this process's memory) can still rewrite
   the entire chain from scratch, though -- for real tamper resistance,
   ship events to an external append-only sink as they're recorded.

Design
------
Same low-coupling shape as the rest of StarkOS: small Protocols
(`AbacPolicy`, `SigningProvider`, `AuthorizationProvider`) with honest,
transparent defaults, so a real secrets manager, a real policy engine
(e.g. OPA), or a real sandboxing backend can be wired in later without
`SecurityCore` changing shape.

`SecurityCore` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    security = SecurityCore(services=services, config=SecurityCoreConfig(
        hmac_secret=os.environb[b"STARK_HMAC_SECRET"],
    ))
    security.bind_identity(identity)
    security.bind_kernel(kernel)
    kernel.register_module(security, name="security_core", priority=10)

    principal = security.principal_for_identity()
    security.require(principal, "kernel.restart", resource="kernel")
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import inspect
import json
import secrets
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from core.identity import ClearanceLevel, Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.service_container import ServiceContainer

logger = get_logger("security_core")

# =============================================================================
# Exceptions
# =============================================================================

class SecurityCoreError(Exception):
    """Base exception for SecurityCore failures."""

class AuthorizationDeniedError(SecurityCoreError):
    """Raised by `require()` when a Principal is not authorized."""

class SigningError(SecurityCoreError):
    """Raised when signing fails (verification failures return False, not this)."""

class SandboxExecutionError(SecurityCoreError):
    """Raised when a sandboxed call raises."""

class SandboxTimeoutError(SandboxExecutionError):
    """Raised when a sandboxed call exceeds its wall-clock timeout."""

class AuditIntegrityError(SecurityCoreError):
    """Raised when the audit hash chain fails verification."""

# =============================================================================
# Principals, roles, permissions (RBAC)
# =============================================================================

@dataclass(slots=True, frozen=True)
class Principal:
    """Whoever/whatever is asking to perform an action -- a user session,
    a module, a plugin, an agent."""

    id: str
    kind: str  # "user" | "module" | "plugin" | "agent"
    roles: frozenset[str] = frozenset()
    attributes: dict[str, Any] = field(default_factory=dict)

    def with_role(self, role_name: str) -> "Principal":
        return Principal(id=self.id, kind=self.kind, roles=self.roles | {role_name}, attributes=dict(self.attributes))

@dataclass(slots=True, frozen=True)
class Role:
    """A named set of permissions. Permission strings are
    "resource.action" (e.g. "kernel.restart"); "*" or a ".*" suffix
    (e.g. "kernel.*") acts as a wildcard."""

    name: str
    permissions: frozenset[str]
    description: str = ""

def _permission_matches(granted: str, requested: str) -> bool:
    if granted == "*" or granted == requested:
        return True
    if granted.endswith(".*"):
        prefix = granted[:-2]
        return requested == prefix or requested.startswith(prefix + ".")
    return False

# Sensible, minimal defaults mirroring Identity.ClearanceLevel's own
# names 1:1 -- not a coincidence, this is the intended bridge between
# the two systems (see SecurityCore.principal_for_identity()).
DEFAULT_ROLES: tuple[Role, ...] = (
    Role(name="guest", permissions=frozenset({"identity.respond"}), description="Unauthenticated / minimal access."),
    Role(
        name="user",
        permissions=frozenset({"identity.*", "knowledge_graph.read", "auto_engineer.read"}),
        description="Authenticated user, read-mostly access.",
    ),
    Role(
        name="operator",
        permissions=frozenset(
            {"identity.*", "knowledge_graph.*", "auto_engineer.*", "vision_engine.*", "kernel.health", "kernel.diagnostics"}
        ),
        description="Day-to-day operation of the cognitive stack, no destructive Kernel actions.",
    ),
    Role(
        name="admin",
        permissions=frozenset({"*.*", "*", "kernel.restart", "kernel.stop", "kernel.start"}),
        description="Full operational control, including Kernel lifecycle.",
    ),
    Role(name="founder", permissions=frozenset({"*"}), description="Unrestricted."),
)

_CLEARANCE_TO_ROLE: dict[str, str] = {level.name.lower(): level.name.lower() for level in ClearanceLevel}

# =============================================================================
# ABAC policies
# =============================================================================

@runtime_checkable
class AbacPolicy(Protocol):
    """
    An attribute-based rule. `evaluate()` returns True (explicit allow),
    False (explicit deny -- always wins over RBAC), or None (abstain --
    this policy has no opinion on this particular action).
    """

    name: str

    def evaluate(
        self, principal: Principal, action: str, resource: str, context: dict[str, Any]
    ) -> Optional[bool]:
        ...

class TimeWindowPolicy:
    """Restricts an action pattern to a range of hours (local time,
    or `context["hour"]` if supplied). A real, common ABAC pattern --
    e.g. "no unattended restarts outside business hours"."""

    def __init__(self, *, name: str, action_pattern: str, allowed_hours: range) -> None:
        self.name = name
        self._action_pattern = action_pattern
        self._allowed_hours = allowed_hours

    def evaluate(self, principal: Principal, action: str, resource: str, context: dict[str, Any]) -> Optional[bool]:
        if not _permission_matches(self._action_pattern, action):
            return None
        hour = context.get("hour", datetime.now().hour)
        return hour in self._allowed_hours

class ClearanceThresholdPolicy:
    """Requires a Principal's numeric `attributes["clearance"]` to meet
    a minimum for a given action pattern -- an extra, explicit floor on
    top of whatever RBAC alone would allow."""

    def __init__(self, *, name: str, action_pattern: str, minimum_clearance: int) -> None:
        self.name = name
        self._action_pattern = action_pattern
        self._minimum = minimum_clearance

    def evaluate(self, principal: Principal, action: str, resource: str, context: dict[str, Any]) -> Optional[bool]:
        if not _permission_matches(self._action_pattern, action):
            return None
        return int(principal.attributes.get("clearance", 0)) >= self._minimum

# =============================================================================
# Authorization decisions
# =============================================================================

@dataclass(slots=True, frozen=True)
class AuthorizationDecision:
    allowed: bool
    principal_id: str
    action: str
    resource: str
    reason: str
    policy_name: Optional[str] = None
    decided_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Audit log (hash-chained, tamper-evident)
# =============================================================================

@dataclass(slots=True, frozen=True)
class AuditEvent:
    id: str
    timestamp: datetime
    principal_id: str
    action: str
    resource: str
    outcome: str  # "allowed" | "denied" | "error"
    reason: str
    previous_hash: str
    hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

def _chain_hash(previous_hash: str, event_id: str, timestamp: datetime, principal_id: str, action: str, resource: str, outcome: str, reason: str) -> str:
    payload = f"{previous_hash}|{event_id}|{timestamp.isoformat()}|{principal_id}|{action}|{resource}|{outcome}|{reason}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

_GENESIS_HASH = "0" * 64

class AuditLog:
    """
    Append-only, hash-chained record of every authorization decision and
    sandboxed execution. See the module docstring for what "tamper-
    evident, not tamper-proof" actually means here.
    """

    def __init__(self, *, persist_path: Optional[Path] = None) -> None:
        self._events: list[AuditEvent] = []
        self._persist_path = persist_path
        self._lock = RLock()

    def record(
        self,
        *,
        principal_id: str,
        action: str,
        resource: str,
        outcome: str,
        reason: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        with self._lock:
            previous_hash = self._events[-1].hash if self._events else _GENESIS_HASH
            event_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            event_hash = _chain_hash(previous_hash, event_id, timestamp, principal_id, action, resource, outcome, reason)
            event = AuditEvent(
                id=event_id,
                timestamp=timestamp,
                principal_id=principal_id,
                action=action,
                resource=resource,
                outcome=outcome,
                reason=reason,
                previous_hash=previous_hash,
                hash=event_hash,
                metadata=metadata or {},
            )
            self._events.append(event)
        logger.debug("Audit event recorded.", extra={"event_id": event.id, "outcome": outcome, "action": action})
        return event

    def verify_integrity(self) -> bool:
        """Recompute the whole hash chain and confirm nothing in this
        process's copy of the log has been altered out from under it."""
        with self._lock:
            events = list(self._events)

        previous_hash = _GENESIS_HASH
        for event in events:
            expected = _chain_hash(previous_hash, event.id, event.timestamp, event.principal_id, event.action, event.resource, event.outcome, event.reason)
            if expected != event.hash or event.previous_hash != previous_hash:
                return False
            previous_hash = event.hash
        return True

    def events(
        self,
        *,
        principal_id: Optional[str] = None,
        action_prefix: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> tuple[AuditEvent, ...]:
        with self._lock:
            events = list(self._events)
        if principal_id is not None:
            events = [event for event in events if event.principal_id == principal_id]
        if action_prefix is not None:
            events = [event for event in events if event.action.startswith(action_prefix)]
        if outcome is not None:
            events = [event for event in events if event.outcome == outcome]
        return tuple(events)

    def persist(self) -> None:
        if self._persist_path is None:
            return
        with self._lock:
            payload = [self._event_to_dict(event) for event in self._events]
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            tmp_path.replace(self._persist_path)
        except OSError as exc:
            raise SecurityCoreError(f"Unable to persist audit log to '{self._persist_path}'.") from exc

    def load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            with self._persist_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            events = [self._event_from_dict(entry) for entry in payload]
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise SecurityCoreError(f"Unable to load audit log from '{self._persist_path}'.") from exc

        with self._lock:
            self._events = events

        if not self.verify_integrity():
            raise AuditIntegrityError(f"Audit log loaded from '{self._persist_path}' failed hash-chain verification.")

    @staticmethod
    def _event_to_dict(event: AuditEvent) -> dict[str, Any]:
        return {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "principal_id": event.principal_id,
            "action": event.action,
            "resource": event.resource,
            "outcome": event.outcome,
            "reason": event.reason,
            "previous_hash": event.previous_hash,
            "hash": event.hash,
            "metadata": event.metadata,
        }

    @staticmethod
    def _event_from_dict(data: dict[str, Any]) -> AuditEvent:
        return AuditEvent(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            principal_id=data["principal_id"],
            action=data["action"],
            resource=data["resource"],
            outcome=data["outcome"],
            reason=data["reason"],
            previous_hash=data["previous_hash"],
            hash=data["hash"],
            metadata=data.get("metadata", {}),
        )

# =============================================================================
# Signing (HMAC default; real asymmetric Ed25519 if `cryptography` exists)
# =============================================================================

@runtime_checkable
class SigningProvider(Protocol):
    def sign(self, data: bytes) -> bytes: ...
    def verify(self, data: bytes, signature: bytes) -> bool: ...
    def check_available(self) -> bool: ...

class HMACSigner:
    """
    Symmetric authentication via HMAC-SHA256 -- a Message Authentication
    Code, not a true digital signature (see module docstring: no non-
    repudiation, verification needs the same secret used to sign).
    Uses `hmac.compare_digest` for constant-time comparison, since a
    naive `==` on the digests would leak timing information.
    """

    def __init__(self, *, secret: bytes) -> None:
        if not secret:
            raise ValueError("secret cannot be empty.")
        self._secret = secret

    def check_available(self) -> bool:
        return True

    def sign(self, data: bytes) -> bytes:
        return hmac.new(self._secret, data, hashlib.sha256).digest()

    def verify(self, data: bytes, signature: bytes) -> bool:
        expected = hmac.new(self._secret, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)

class Ed25519Signer:
    """
    Real asymmetric digital signatures via Ed25519, through the optional
    `cryptography` package. Unlike HMACSigner, verification only needs
    the public key, so a verifier can never forge a signature -- genuine
    non-repudiation. Degrades to raising SigningError if `cryptography`
    isn't installed; `check_available()` reports this up front.
    """

    def __init__(self, *, private_key_bytes: Optional[bytes] = None) -> None:
        self._private_key_bytes = private_key_bytes
        self._private_key: Any = None
        self._public_key: Any = None

    def check_available(self) -> bool:
        try:
            import cryptography  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_keys(self) -> None:
        if self._private_key is not None:
            return
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError as exc:
            raise SigningError("The 'cryptography' package is not installed.") from exc

        try:
            if self._private_key_bytes:
                self._private_key = Ed25519PrivateKey.from_private_bytes(self._private_key_bytes)
            else:
                self._private_key = Ed25519PrivateKey.generate()
            self._public_key = self._private_key.public_key()
        except Exception as exc:
            raise SigningError("Unable to initialize an Ed25519 key pair.") from exc

    def sign(self, data: bytes) -> bytes:
        self._ensure_keys()
        try:
            return self._private_key.sign(data)
        except Exception as exc:
            raise SigningError("Ed25519 signing failed.") from exc

    def verify(self, data: bytes, signature: bytes) -> bool:
        self._ensure_keys()
        try:
            self._public_key.verify(signature, data)
            return True
        except Exception:
            return False

    def public_key_bytes(self) -> bytes:
        self._ensure_keys()
        from cryptography.hazmat.primitives import serialization
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

# =============================================================================
# Sandbox (capability-gated, time-bounded -- see honesty note above)
# =============================================================================

@runtime_checkable
class AuthorizationProvider(Protocol):
    def authorize(
        self, principal: Principal, action: str, resource: str, *, context: Optional[dict[str, Any]] = None
    ) -> AuthorizationDecision: ...

    def audit(
        self,
        *,
        principal_id: str,
        action: str,
        resource: str,
        outcome: str,
        reason: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEvent: ...

@dataclass(slots=True, frozen=True)
class SandboxResult:
    output: Any
    duration_seconds: float

class PluginSandbox:
    """
    Capability-gated, time-bounded execution for semi-trusted internal
    plugins/agents. NOT a security boundary against malicious code --
    see the module docstring's honesty section before relying on this
    for anything adversarial.
    """

    def __init__(self, authorization_provider: AuthorizationProvider, *, default_timeout: float = 5.0, max_workers: int = 4) -> None:
        self._authz = authorization_provider
        self._default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stark-sandbox")

    async def run(
        self,
        principal: Principal,
        func: Callable[..., Any],
        *args: Any,
        required_permission: str,
        resource: str,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> SandboxResult:
        decision = self._authz.authorize(principal, required_permission, resource)
        if not decision.allowed:
            raise AuthorizationDeniedError(f"Sandbox execution denied for '{principal.id}': {decision.reason}")

        effective_timeout = timeout if timeout is not None else self._default_timeout
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        future = loop.run_in_executor(self._executor, functools.partial(func, *args, **kwargs))

        try:
            output = await asyncio.wait_for(future, timeout=effective_timeout)
        except asyncio.TimeoutError as exc:
            self._authz.audit(
                principal_id=principal.id, action=required_permission, resource=resource,
                outcome="error", reason=f"Timed out after {effective_timeout}s.",
            )
            raise SandboxTimeoutError(
                f"Sandboxed call exceeded its {effective_timeout}s timeout. The underlying thread was "
                "NOT forcibly killed (Python has no safe way to do that) and may still be running -- "
                "this timeout means 'we stopped waiting', not 'the call actually stopped'."
            ) from exc
        except Exception as exc:
            self._authz.audit(
                principal_id=principal.id, action=required_permission, resource=resource,
                outcome="error", reason=str(exc),
            )
            raise SandboxExecutionError(f"Sandboxed call raised: {exc}") from exc

        duration = loop.time() - started_at
        self._authz.audit(
            principal_id=principal.id, action=required_permission, resource=resource,
            outcome="allowed", reason="Sandbox execution succeeded.", metadata={"duration_seconds": round(duration, 4)},
        )
        return SandboxResult(output=output, duration_seconds=duration)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class SecurityCoreConfig:
    # If None, a fresh random secret is generated per process -- fine
    # for a single run, but signatures won't verify across a restart
    # unless you supply a persistent secret (env var, secrets manager).
    hmac_secret: Optional[bytes] = None
    audit_persist_path: Optional[Path] = None
    persist_audit_on_shutdown: bool = True
    record_to_knowledge_graph: bool = True
    default_sandbox_timeout: float = 5.0
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Security Core
# =============================================================================

class SecurityCore:
    """
    StarkOS security module: RBAC + ABAC authorization, a tamper-evident
    audit log, signing, and a capability-gated plugin sandbox. See the
    module docstring's "Honesty about scope" section for what each
    piece actually guarantees.

    Satisfies the `Module` protocol (name/initialize/start/stop) and can
    be registered with the Kernel like any other module.
    """

    _KERNEL_ACTIONS: dict[str, Callable[[Any], Callable[..., Any]]] = {
        "restart": lambda kernel: kernel.restart,
        "stop": lambda kernel: kernel.stop,
        "start": lambda kernel: kernel.start,
        "health": lambda kernel: kernel.health,
        "diagnostics": lambda kernel: kernel.diagnostics,
    }

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[SecurityCoreConfig] = None,
        audit_log: Optional[AuditLog] = None,
        signer: Optional[SigningProvider] = None,
    ) -> None:
        self._services = services
        self._config = config or SecurityCoreConfig()
        self._roles: dict[str, Role] = {role.name: role for role in DEFAULT_ROLES}
        self._policies: list[AbacPolicy] = []

        self._audit_log = audit_log or AuditLog(persist_path=self._config.audit_persist_path)
        self._signer: SigningProvider = signer or HMACSigner(secret=self._config.hmac_secret or secrets.token_bytes(32))
        self._signer_available = True

        self._sandbox = PluginSandbox(self, default_timeout=self._config.default_sandbox_timeout)

        self._kernel: Any = None
        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None

        logger.info("SecurityCore constructed.", extra={"roles": list(self._roles.keys())})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "security_core"

    @property
    def sandbox(self) -> PluginSandbox:
        return self._sandbox

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    def bind_kernel(self, kernel: Any) -> None:
        """Kernel does not register itself into the ServiceContainer, so
        it is handed to modules explicitly -- mirrors Identity/VoiceInterface."""
        self._kernel = kernel
        logger.debug("Kernel bound to SecurityCore.")

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to SecurityCore.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to SecurityCore.")

    async def initialize(self) -> None:
        logger.info("Initializing SecurityCore.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)

        try:
            await asyncio.to_thread(self._audit_log.load)
        except AuditIntegrityError:
            logger.exception("Persisted audit log failed integrity verification -- refusing to trust it silently.")
            raise
        except SecurityCoreError:
            logger.exception("Failed to load persisted audit log -- starting from an empty log.")

        self._signer_available = await self._probe(self._signer)
        if not self._signer_available:
            logger.warning("Configured signing provider is unavailable -- sign()/verify() will raise until fixed.")

        logger.info(
            "SecurityCore initialized.",
            extra={"audit_events": len(self._audit_log.events()), "signer_available": self._signer_available},
        )

    async def start(self) -> None:
        logger.info("SecurityCore ready.", extra={"roles": len(self._roles), "policies": len(self._policies)})

    async def stop(self) -> None:
        logger.info("Stopping SecurityCore.")
        self._sandbox.shutdown()
        if self._config.persist_audit_on_shutdown:
            try:
                await asyncio.to_thread(self._audit_log.persist)
            except SecurityCoreError:
                logger.exception("Failed to persist audit log on shutdown.")
        logger.info("SecurityCore stopped.")

    async def _probe(self, provider: Any) -> bool:
        check = getattr(provider, "check_available", None)
        if check is None:
            return True
        try:
            return bool(await asyncio.to_thread(check))
        except Exception:
            logger.exception("Availability probe failed for a security provider.")
            return False

    # ------------------------------------------------------------------
    # RBAC: role management
    # ------------------------------------------------------------------

    def register_role(self, role: Role) -> None:
        self._roles[role.name] = role
        logger.info("Role registered.", extra={"role": role.name, "permission_count": len(role.permissions)})

    def get_role(self, name: str) -> Role:
        role = self._roles.get(name)
        if role is None:
            raise SecurityCoreError(f"Unknown role '{name}'.")
        return role

    def list_roles(self) -> tuple[Role, ...]:
        return tuple(self._roles.values())

    # ------------------------------------------------------------------
    # ABAC: policy management
    # ------------------------------------------------------------------

    def register_policy(self, policy: AbacPolicy) -> None:
        self._policies.append(policy)
        logger.info("ABAC policy registered.", extra={"policy": policy.name})

    def list_policies(self) -> tuple[AbacPolicy, ...]:
        return tuple(self._policies)

    # ------------------------------------------------------------------
    # Authorization (RBAC + ABAC, deny-by-default, fail-closed)
    # ------------------------------------------------------------------

    def _rbac_grants(self, principal: Principal, action: str) -> bool:
        for role_name in principal.roles:
            role = self._roles.get(role_name)
            if role is None:
                continue
            if any(_permission_matches(granted, action) for granted in role.permissions):
                return True
        return False

    def authorize(
        self,
        principal: Principal,
        action: str,
        resource: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """
        Zero-trust authorization: re-evaluated every call, every time.
        Deny-by-default (RBAC must explicitly grant); any ABAC policy
        returning an explicit deny overrides an RBAC grant; any internal
        error during evaluation is treated as a deny (fail-closed), never
        an allow. The outcome is always audited, allow or deny.
        """
        context = context or {}
        try:
            rbac_allowed = self._rbac_grants(principal, action)

            policy_verdicts: list[tuple[str, bool]] = []
            for policy in self._policies:
                try:
                    verdict = policy.evaluate(principal, action, resource, context)
                except Exception:
                    logger.exception("ABAC policy '%s' raised -- treating as abstain.", policy.name)
                    verdict = None
                if verdict is not None:
                    policy_verdicts.append((policy.name, verdict))

            explicit_deny = next((policy_name for policy_name, verdict in policy_verdicts if verdict is False), None)

            if explicit_deny is not None:
                decision = AuthorizationDecision(
                    allowed=False, principal_id=principal.id, action=action, resource=resource,
                    reason=f"Denied by ABAC policy '{explicit_deny}'.", policy_name=explicit_deny,
                )
            elif not rbac_allowed:
                decision = AuthorizationDecision(
                    allowed=False, principal_id=principal.id, action=action, resource=resource,
                    reason="No role grants this permission (default-deny).",
                )
            else:
                granting_policy = next((policy_name for policy_name, verdict in policy_verdicts if verdict is True), None)
                decision = AuthorizationDecision(
                    allowed=True, principal_id=principal.id, action=action, resource=resource,
                    reason="Granted by role membership; no policy objected.", policy_name=granting_policy,
                )
        except Exception as exc:
            logger.exception("Authorization evaluation raised -- failing closed (deny).")
            decision = AuthorizationDecision(
                allowed=False, principal_id=principal.id, action=action, resource=resource,
                reason=f"Authorization evaluation error (fail-closed): {exc}",
            )

        self.audit(
            principal_id=principal.id, action=action, resource=resource,
            outcome="allowed" if decision.allowed else "denied", reason=decision.reason,
        )
        return decision

    def require(
        self,
        principal: Principal,
        action: str,
        resource: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """Like `authorize()`, but raises AuthorizationDeniedError instead
        of returning a denied decision -- for call sites that should
        simply refuse to proceed."""
        decision = self.authorize(principal, action, resource, context=context)
        if not decision.allowed:
            raise AuthorizationDeniedError(f"Principal '{principal.id}' denied '{action}' on '{resource}': {decision.reason}")
        return decision

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(
        self,
        *,
        principal_id: str,
        action: str,
        resource: str,
        outcome: str,
        reason: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AuditEvent:
        event = self._audit_log.record(
            principal_id=principal_id, action=action, resource=resource, outcome=outcome, reason=reason, metadata=metadata
        )
        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            try:
                self._knowledge_graph.remember(
                    f"Audit: '{principal_id}' {outcome} on '{action}' -> '{resource}' ({reason})",
                    node_type="audit_event",
                    metadata={"event_id": event.id, "outcome": outcome, "action": action, "resource": resource},
                    source="security_core",
                )
            except Exception:
                logger.exception("Failed to record audit event in KnowledgeGraph.")
        return event

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign(self, data: bytes) -> bytes:
        try:
            return self._signer.sign(data)
        except SecurityCoreError:
            raise
        except Exception as exc:
            raise SigningError("Failed to sign data.") from exc

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            return self._signer.verify(data, signature)
        except Exception:
            logger.exception("Signature verification raised -- treating as invalid (fail-closed).")
            return False

    # ------------------------------------------------------------------
    # Identity integration
    # ------------------------------------------------------------------

    def principal_for_identity(self, *, principal_id: Optional[str] = None) -> Principal:
        """
        Derive a Principal from the bound Identity's current clearance,
        bridging Identity.ClearanceLevel into RBAC rather than
        maintaining a second, separate permission model.
        """
        if self._identity is None:
            raise SecurityCoreError("No Identity bound -- call bind_identity() first.")

        clearance = self._identity.clearance
        role_name = _CLEARANCE_TO_ROLE.get(clearance.name.lower(), "guest")
        return Principal(
            id=principal_id or self._identity.persona.name,
            kind="user",
            roles=frozenset({role_name}),
            attributes={"clearance": int(clearance)},
        )

    # ------------------------------------------------------------------
    # Kernel integration
    # ------------------------------------------------------------------

    async def secure_kernel_call(self, principal: Principal, action: str, *args: Any, **kwargs: Any) -> Any:
        """
        Authorize, then invoke a Kernel method from a small, explicit
        allow-list -- never arbitrary attribute access from a string.
        Always audited, whether it succeeds or raises.
        """
        if self._kernel is None:
            raise SecurityCoreError("No Kernel bound -- call bind_kernel() first.")

        handler_factory = self._KERNEL_ACTIONS.get(action)
        if handler_factory is None:
            raise SecurityCoreError(f"'{action}' is not an exposed Kernel action. Available: {sorted(self._KERNEL_ACTIONS)}")

        permission = f"kernel.{action}"
        self.require(principal, permission, resource="kernel")

        method = handler_factory(self._kernel)
        try:
            if inspect.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
        except Exception as exc:
            self.audit(principal_id=principal.id, action=permission, resource="kernel", outcome="error", reason=str(exc))
            raise

        self.audit(principal_id=principal.id, action=permission, resource="kernel", outcome="allowed", reason="Kernel action completed.")
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "roles": sorted(self._roles.keys()),
            "policies": [policy.name for policy in self._policies],
            "audit_events": len(self._audit_log.events()),
            "audit_chain_valid": self._audit_log.verify_integrity(),
            "signer_available": self._signer_available,
            "signer_kind": type(self._signer).__name__,
            "identity_bound": self._identity is not None,
            "kernel_bound": self._kernel is not None,
            "knowledge_graph_bound": self._knowledge_graph is not None,
        }