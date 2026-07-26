"""
core/digital_thread.py
========================

The Digital Thread: an immutable, hash-chained record of every
decision and action StarkOS takes, from a goal's submission to its
final execution result.

Responsibilities
----------------
- Record every decision/action with: what data was read (`inputs`), the
  actual component/method that acted (`method` -- see the honesty note
  on why this is never a fabricated model name), whatever
  prompt/parameters it used (`parameters`), the validation outcome (if
  any), the result, and who/what was responsible (`actor`).
- Store entries in an append-only, hash-chained structure: each entry's
  hash covers the previous entry's hash plus its own fields, so
  retroactively editing or deleting one breaks the chain from that
  point forward -- see the honesty note on what "immutable" actually
  means here.
- End-to-end traceability: every entry carries a `trace_id` (shared by
  everything from one goal to its final result) and a `parent_entry_id`
  (the entry that caused it), so `parent_chain()` can walk from any
  result all the way back to the goal that started it.
- Versioning without mutation: a revised decision is a *new* entry with
  `supersedes` pointing at the one it replaces -- the old entry is never
  touched, and `get_latest_version()` walks forward to the current one.
- Audit support: indexed queries (by actor/trace/type/time range) and
  export to JSON (full fidelity) or a human-readable Markdown report.
- Integration with KnowledgeGraph (entries mirrored as searchable
  memory), EventBus (system-level events recorded automatically, same
  pattern as `core.knowledge_graph`), Identity (actor attribution), and
  convenience translators for CognitiveEngine's Goal/Plan/
  PlanExecutionResult and RAGEngine's RAGAnswer into fully-linked entries.

Honesty about scope
--------------------
1. **"Immutable" means tamper-*evident*, not tamper-*proof* -- the exact
   same honest distinction as `core.security_core.AuditLog` (this
   module reuses that reasoning, generalized to a richer schema and
   full goal-to-execution traceability). Each entry's hash covers the
   previous entry's hash, so `verify_integrity()` will catch any
   retroactive edit, insertion or deletion in this process's copy of
   the thread. An attacker with write access to the persisted file (or
   this process's memory) could still rewrite the entire chain from
   scratch, though -- for real tamper resistance in a genuinely
   critical environment, ship entries to an external, append-only sink
   (write-once storage, a separate logging service, or a real
   blockchain/notary) as they're recorded, and treat this module's own
   copy as the fast, local, *verifiable* record rather than the sole
   source of truth.

2. **"Modelo utilizado" is always a real, named StarkOS component --
   never a fabricated AI model name.** StarkOS has no general-purpose
   language model. When this module records what acted on a decision,
   `method` holds the real class/function name that actually ran (e.g.
   "HeuristicGoalInterpreter", "WeightedSumEvaluator",
   "ExtractiveSynthesizer", "auto_engineer.optimize") -- never an
   invented model identifier. If a genuine AI Runtime/LLM provider is
   added to StarkOS later, its real model identifier belongs here
   exactly the same way.

3. **Recording is the caller's responsibility; this module doesn't
   silently instrument other modules.** CognitiveEngine, RAGEngine, etc.
   don't automatically call into `DigitalThread` -- their public result
   objects (`PlanExecutionResult`, `RAGAnswer`, ...) are what
   `record_from_plan_execution()`/`record_from_rag_answer()` translate
   into linked entries, called explicitly by whoever orchestrated the
   work. The one exception is EventBus-published system events (kernel/
   module lifecycle), which are recorded automatically once
   `bind_event_bus()` is called -- the same opt-in subscription pattern
   `core.knowledge_graph` already uses.

Design
------
An append-only list plus three indexes (`by_id`, `by_trace`, `by_actor`)
built incrementally as entries are recorded -- `get_entry()`/`get_trace()`
are O(1)/O(k) rather than an O(N) scan of the whole thread; only
`verify_integrity()` and unindexed `query()` filters are O(N), which is
the honest, unavoidable cost of actually re-checking every hash or
scanning by a dimension with no index.

`DigitalThread` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    thread = DigitalThread(services=services, config=DigitalThreadConfig(
        persist_path=Path("data/digital_thread.json"),
    ))
    thread.bind_knowledge_graph(knowledge_graph)
    thread.bind_identity(identity)
    kernel.register_module(thread, name="digital_thread", priority=15)

    trace_id = thread.begin_trace("Optimize the motor mount")
    decision = thread.record_decision(
        trace_id=trace_id, description="Chose hill-climbing over random search",
        inputs={"spec_name": spec.name}, method="AutoEngineer.optimize",
        parameters={"iterations": 100}, result={"best_score": 12.4},
    )
    thread.record_result(trace_id=trace_id, description="Design finalized",
                          result={"accepted": True}, parent_entry_id=decision.id)
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from core.cognitive_engine import Goal, Plan, PlanExecutionResult
from core.event_bus import Event, EventBus
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.rag_engine import RAGAnswer
from core.service_container import ServiceContainer

logger = get_logger("digital_thread")

# =============================================================================
# Exceptions
# =============================================================================

class DigitalThreadError(Exception):
    """Base exception for DigitalThread failures."""

class EntryNotFoundError(DigitalThreadError):
    """Raised when a referenced entry id doesn't exist."""

class IntegrityError(DigitalThreadError):
    """Raised when the hash chain fails verification."""

class InvalidEntryError(DigitalThreadError):
    """Raised when record() arguments are malformed."""

# =============================================================================
# Entry schema
# =============================================================================

class EntryType(Enum):
    GOAL = auto()
    DECISION = auto()
    ACTION = auto()
    VALIDATION = auto()
    RESULT = auto()
    SYSTEM_EVENT = auto()

@dataclass(slots=True, frozen=True)
class DigitalThreadEntry:
    """
    One immutable record. Never mutated after creation -- a revision is
    always a *new* entry with `supersedes` pointing at this one (see
    `record_revision()`/`get_latest_version()`).
    """

    id: str
    trace_id: str
    entry_type: str
    sequence: int  # monotonic position across the WHOLE thread -- total ordering, immune to timestamp ties
    timestamp: datetime
    actor: str
    description: str
    inputs: dict[str, Any]  # "dados lidos"
    method: str  # "modelo utilizado" -- always a real component name, see honesty note
    parameters: dict[str, Any]  # e.g. a prompt/task configuration
    validation: Optional[dict[str, Any]]
    result: Any
    parent_entry_id: Optional[str]  # the entry that led to this one -- builds the goal-to-execution chain
    supersedes: Optional[str]  # the entry this one revises, if any
    previous_hash: str
    hash: str

_GENESIS_HASH = "0" * 64

def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)

def _chain_hash(
    previous_hash: str, entry_id: str, timestamp: datetime, actor: str, entry_type: str, description: str, safe_result: Any
) -> str:
    payload = f"{previous_hash}|{entry_id}|{timestamp.isoformat()}|{actor}|{entry_type}|{description}|{_stable_json(safe_result)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class DigitalThreadConfig:
    persist_path: Optional[Path] = None
    persist_on_shutdown: bool = True
    record_to_knowledge_graph: bool = True
    record_system_events: bool = True
    tracked_event_topics: tuple[str, ...] = (
        "kernel.initialized", "kernel.running", "kernel.stopped", "kernel.restarted",
        "module.started", "module.stopped",
    )
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Digital Thread
# =============================================================================

class DigitalThread:
    """
    StarkOS's immutable, hash-chained record of decisions and actions.
    See the module docstring's "Honesty about scope" section before
    relying on this for anything beyond what it actually guarantees.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(self, *, services: ServiceContainer, config: Optional[DigitalThreadConfig] = None) -> None:
        self._services = services
        self._config = config or DigitalThreadConfig()
        self._lock = RLock()

        self._entries: list[DigitalThreadEntry] = []
        self._sequence_counter = 0
        self._by_id: dict[str, DigitalThreadEntry] = {}
        self._by_trace: dict[str, list[str]] = defaultdict(list)
        self._by_actor: dict[str, list[str]] = defaultdict(list)
        self._superseded_by: dict[str, str] = {}

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._event_bus: Optional[EventBus] = None
        self._event_subscriptions: list[Any] = []

        logger.info("DigitalThread constructed.", extra={"persist_path": str(self._config.persist_path or "")})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "digital_thread"

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to DigitalThread.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to DigitalThread.")

    def bind_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        logger.debug("EventBus bound to DigitalThread.")

    async def initialize(self) -> None:
        logger.info("Initializing DigitalThread.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._event_bus is None:
            self._event_bus = self._services.resolve_optional(EventBus)

        try:
            await asyncio.to_thread(self._load)
        except IntegrityError:
            logger.exception("Persisted digital thread failed integrity verification -- refusing to trust it silently.")
            raise
        except DigitalThreadError:
            logger.exception("Failed to load persisted digital thread -- starting from an empty thread.")

        if self._config.record_system_events and self._event_bus is not None:
            self._subscribe_to_events()

        logger.info("DigitalThread initialized.", extra={"entry_count": self.entry_count})

    async def start(self) -> None:
        logger.info("DigitalThread ready.", extra={"entry_count": self.entry_count})

    async def stop(self) -> None:
        logger.info("Stopping DigitalThread.")
        self._unsubscribe_from_events()
        if self._config.persist_on_shutdown:
            try:
                await asyncio.to_thread(self._persist)
            except DigitalThreadError:
                logger.exception("Failed to persist digital thread on shutdown.")
        logger.info("DigitalThread stopped.")

    # ------------------------------------------------------------------
    # EventBus integration (same opt-in pattern as core.knowledge_graph)
    # ------------------------------------------------------------------

    def _subscribe_to_events(self) -> None:
        if self._event_bus is None or self._event_subscriptions:
            return
        for topic in self._config.tracked_event_topics:
            try:
                subscription = self._event_bus.subscribe(topic, self._on_system_event)
                self._event_subscriptions.append(subscription)
            except Exception:
                logger.exception("Failed to subscribe to event topic '%s'.", topic)
        logger.info("Subscribed to system events.", extra={"topics": list(self._config.tracked_event_topics)})

    def _unsubscribe_from_events(self) -> None:
        if self._event_bus is None:
            return
        for subscription in self._event_subscriptions:
            try:
                self._event_bus.unsubscribe(subscription)
            except Exception:
                logger.exception("Failed to unsubscribe from an event topic.")
        self._event_subscriptions.clear()

    def _on_system_event(self, event: Event) -> None:
        try:
            self.record(
                trace_id="system-events",
                entry_type=EntryType.SYSTEM_EVENT.name,
                actor="event_bus",
                description=f"System event: {event.topic}",
                inputs={"payload": event.payload},
                method="EventBus.subscribe",
                parameters={"topic": event.topic, "source": event.source},
                result=None,
            )
        except Exception:
            logger.exception("Failed to record system event '%s'.", event.topic)

    # ------------------------------------------------------------------
    # Actor resolution
    # ------------------------------------------------------------------

    def _resolve_actor(self) -> str:
        if self._identity is not None:
            try:
                return self._identity.persona.name
            except Exception:
                logger.exception("Failed to resolve actor from Identity -- falling back to 'system'.")
        return "system"

    # ------------------------------------------------------------------
    # Core recording
    # ------------------------------------------------------------------

    def begin_trace(self, description: str, *, actor: Optional[str] = None) -> str:
        """Start a new end-to-end trace (typically one goal's full
        lifecycle) and return its trace_id. Every subsequent record for
        this trace should pass this same trace_id."""
        trace_id = str(uuid.uuid4())
        self.record(
            trace_id=trace_id, entry_type=EntryType.GOAL.name, actor=actor,
            description=description, inputs={}, method="begin_trace", parameters={}, result=None,
        )
        return trace_id

    def record(
        self,
        *,
        trace_id: str,
        entry_type: str,
        description: str,
        actor: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None,
        method: str = "unknown",
        parameters: Optional[dict[str, Any]] = None,
        validation: Optional[dict[str, Any]] = None,
        result: Any = None,
        parent_entry_id: Optional[str] = None,
        supersedes: Optional[str] = None,
    ) -> DigitalThreadEntry:
        if not trace_id:
            raise InvalidEntryError("trace_id is required.")
        if not description:
            raise InvalidEntryError("description is required.")

        resolved_actor = actor or self._resolve_actor()
        safe_inputs = self._json_safe(inputs or {})
        safe_parameters = self._json_safe(parameters or {})
        safe_validation = self._json_safe(validation) if validation is not None else None
        safe_result = self._json_safe(result)

        with self._lock:
            if parent_entry_id is not None and parent_entry_id not in self._by_id:
                raise EntryNotFoundError(f"parent_entry_id '{parent_entry_id}' does not exist.")
            if supersedes is not None and supersedes not in self._by_id:
                raise EntryNotFoundError(f"supersedes '{supersedes}' does not exist.")

            entry_id = str(uuid.uuid4())
            sequence = self._sequence_counter
            self._sequence_counter += 1
            timestamp = datetime.utcnow()
            previous_hash = self._entries[-1].hash if self._entries else _GENESIS_HASH
            entry_hash = _chain_hash(previous_hash, entry_id, timestamp, resolved_actor, entry_type, description, safe_result)

            entry = DigitalThreadEntry(
                id=entry_id, trace_id=trace_id, entry_type=entry_type, sequence=sequence, timestamp=timestamp,
                actor=resolved_actor, description=description, inputs=safe_inputs, method=method,
                parameters=safe_parameters, validation=safe_validation, result=safe_result,
                parent_entry_id=parent_entry_id, supersedes=supersedes, previous_hash=previous_hash, hash=entry_hash,
            )

            self._entries.append(entry)
            self._by_id[entry_id] = entry
            self._by_trace[trace_id].append(entry_id)
            self._by_actor[resolved_actor].append(entry_id)
            if supersedes is not None:
                self._superseded_by[supersedes] = entry_id

        logger.info(
            "Thread entry recorded.",
            extra={"entry_id": entry_id, "trace_id": trace_id, "entry_type": entry_type, "sequence": sequence, "actor": resolved_actor},
        )

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            self._mirror_to_knowledge_graph(entry)

        return entry

    # -- Convenience wrappers matching the requested schema exactly --

    def record_decision(
        self, *, trace_id: str, description: str, inputs: dict[str, Any], method: str, parameters: dict[str, Any],
        result: Any, validation: Optional[dict[str, Any]] = None, actor: Optional[str] = None,
        parent_entry_id: Optional[str] = None,
    ) -> DigitalThreadEntry:
        return self.record(
            trace_id=trace_id, entry_type=EntryType.DECISION.name, description=description, actor=actor,
            inputs=inputs, method=method, parameters=parameters, validation=validation, result=result,
            parent_entry_id=parent_entry_id,
        )

    def record_action(
        self, *, trace_id: str, description: str, inputs: dict[str, Any], method: str, parameters: dict[str, Any],
        result: Any, actor: Optional[str] = None, parent_entry_id: Optional[str] = None,
    ) -> DigitalThreadEntry:
        return self.record(
            trace_id=trace_id, entry_type=EntryType.ACTION.name, description=description, actor=actor,
            inputs=inputs, method=method, parameters=parameters, result=result, parent_entry_id=parent_entry_id,
        )

    def record_validation(
        self, *, trace_id: str, description: str, validation: dict[str, Any], result: Any = None,
        actor: Optional[str] = None, parent_entry_id: Optional[str] = None,
    ) -> DigitalThreadEntry:
        return self.record(
            trace_id=trace_id, entry_type=EntryType.VALIDATION.name, description=description, actor=actor,
            inputs={}, method="validation", parameters={}, validation=validation, result=result,
            parent_entry_id=parent_entry_id,
        )

    def record_result(
        self, *, trace_id: str, description: str, result: Any, actor: Optional[str] = None,
        parent_entry_id: Optional[str] = None,
    ) -> DigitalThreadEntry:
        return self.record(
            trace_id=trace_id, entry_type=EntryType.RESULT.name, description=description, actor=actor,
            inputs={}, method="result", parameters={}, result=result, parent_entry_id=parent_entry_id,
        )

    # ------------------------------------------------------------------
    # Versioning without mutation
    # ------------------------------------------------------------------

    def record_revision(
        self, *, supersedes: str, description: str, method: str, parameters: dict[str, Any], result: Any,
        actor: Optional[str] = None,
    ) -> DigitalThreadEntry:
        """Record a new version of an existing decision/state. The
        original entry is never touched -- this creates a new entry in
        the same trace, linked to the original via `supersedes`."""
        original = self.get_entry(supersedes)
        return self.record(
            trace_id=original.trace_id, entry_type=original.entry_type, description=description, actor=actor,
            inputs=original.inputs, method=method, parameters=parameters, result=result,
            parent_entry_id=original.id, supersedes=supersedes,
        )

    def get_latest_version(self, entry_id: str) -> DigitalThreadEntry:
        """Walk forward through supersession links to the current
        (latest) version of a logical decision/state."""
        current_id = entry_id
        visited = {current_id}
        while current_id in self._superseded_by:
            current_id = self._superseded_by[current_id]
            if current_id in visited:
                raise IntegrityError(f"Supersession cycle detected involving '{entry_id}'.")
            visited.add(current_id)
        return self.get_entry(current_id)

    # ------------------------------------------------------------------
    # Retrieval / traceability
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> DigitalThreadEntry:
        entry = self._by_id.get(entry_id)
        if entry is None:
            raise EntryNotFoundError(f"Unknown entry '{entry_id}'.")
        return entry

    def get_trace(self, trace_id: str) -> tuple[DigitalThreadEntry, ...]:
        """Every entry recorded under this trace_id, in the order it
        was recorded (which is also sequence order)."""
        return tuple(self._by_id[entry_id] for entry_id in self._by_trace.get(trace_id, ()))

    def parent_chain(self, entry_id: str) -> tuple[DigitalThreadEntry, ...]:
        """Walk backward via parent_entry_id from `entry_id` (typically
        a final result) all the way to its root (typically the
        original goal) -- the literal goal-to-execution trace.
        Root first, the given entry last."""
        current = self.get_entry(entry_id)
        chain = [current]
        visited = {current.id}
        while current.parent_entry_id is not None:
            parent = self._by_id.get(current.parent_entry_id)
            if parent is None or parent.id in visited:
                break
            chain.append(parent)
            visited.add(parent.id)
            current = parent
        return tuple(reversed(chain))

    def query(
        self,
        *,
        trace_id: Optional[str] = None,
        actor: Optional[str] = None,
        entry_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> tuple[DigitalThreadEntry, ...]:
        """Filtered audit query. Uses the trace/actor indexes when
        possible; falls back to a full scan for time-range/entry_type
        filters (there's no index for those -- an honest O(N) cost)."""
        if trace_id is not None:
            candidates = self.get_trace(trace_id)
        elif actor is not None:
            candidates = tuple(self._by_id[entry_id] for entry_id in self._by_actor.get(actor, ()))
        else:
            with self._lock:
                candidates = tuple(self._entries)

        if actor is not None and trace_id is not None:
            candidates = tuple(entry for entry in candidates if entry.actor == actor)
        if entry_type is not None:
            candidates = tuple(entry for entry in candidates if entry.entry_type == entry_type)
        if since is not None:
            candidates = tuple(entry for entry in candidates if entry.timestamp >= since)
        if until is not None:
            candidates = tuple(entry for entry in candidates if entry.timestamp <= until)
        return candidates

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify_integrity(self) -> bool:
        """Recompute the whole hash chain. O(N) -- an honest, unavoidable
        cost of actually re-checking every entry rather than trusting
        the stored hashes at face value."""
        with self._lock:
            entries = list(self._entries)
        previous_hash = _GENESIS_HASH
        for entry in entries:
            expected = _chain_hash(previous_hash, entry.id, entry.timestamp, entry.actor, entry.entry_type, entry.description, entry.result)
            if expected != entry.hash or entry.previous_hash != previous_hash:
                return False
            previous_hash = entry.hash
        return True

    # ------------------------------------------------------------------
    # CognitiveEngine / RAGEngine translators
    # ------------------------------------------------------------------

    def record_from_plan_execution(
        self, goal: Goal, plan: Plan, execution_result: PlanExecutionResult, *, actor: Optional[str] = None
    ) -> str:
        """
        Translate an already-completed CognitiveEngine Goal/Plan/
        PlanExecutionResult into a fully-linked chain of entries (goal
        -> plan -> each task in sequence -> final result) in one call.
        CognitiveEngine doesn't need to know about DigitalThread --
        call this yourself after `execute_plan()` returns.
        """
        trace_id = self.begin_trace(f"Goal: {goal.description}", actor=actor)
        goal_entry_id = self._by_trace[trace_id][0]

        plan_entry = self.record_decision(
            trace_id=trace_id, description=f"Plan created with {len(plan.tasks)} task(s).",
            inputs={"goal_id": goal.id, "goal_description": goal.description}, method="CognitiveEngine.plan",
            parameters={"plan_id": plan.id, "task_count": len(plan.tasks)}, result={"task_ids": [task.id for task in plan.tasks]},
            actor=actor, parent_entry_id=goal_entry_id,
        )

        parent_id = plan_entry.id
        for task_result in execution_result.task_results:
            task = plan.task(task_result.task_id)
            entry = self.record_action(
                trace_id=trace_id, description=task.description, inputs=task.parameters,
                method=f"{task.specialist}.{task.action}", parameters={"task_id": task.id},
                result={"status": task_result.status.name, "error": task_result.error, "attempts": task_result.attempts},
                actor=actor, parent_entry_id=parent_id,
            )
            parent_id = entry.id

        self.record_result(
            trace_id=trace_id, description="Plan execution completed.",
            result={"succeeded": execution_result.succeeded, "plan_id": plan.id},
            actor=actor, parent_entry_id=parent_id,
        )
        return trace_id

    def record_from_rag_answer(
        self, answer: RAGAnswer, *, trace_id: Optional[str] = None, actor: Optional[str] = None,
        parent_entry_id: Optional[str] = None,
    ) -> DigitalThreadEntry:
        """Translate a RAGEngine RAGAnswer into a linked entry, tagging
        it with the answer's own honestly-labeled synthesis_method and
        confidence (see core.rag_engine's honesty note on what
        "confidence" means)."""
        effective_trace_id = trace_id or self.begin_trace(f"RAG query: {answer.query}", actor=actor)
        return self.record_decision(
            trace_id=effective_trace_id, description=f"RAG answer for: {answer.query}",
            inputs={"source_node_ids": [source.node_id for source in answer.sources]},
            method=f"RAGEngine ({answer.synthesis_method})", parameters={"query": answer.query},
            validation={"confidence": answer.confidence, "confidence_basis": answer.confidence_basis},
            result={"answer_text": answer.answer_text}, actor=actor, parent_entry_id=parent_entry_id,
        )

    # ------------------------------------------------------------------
    # KnowledgeGraph integration
    # ------------------------------------------------------------------

    def _mirror_to_knowledge_graph(self, entry: DigitalThreadEntry) -> None:
        if self._knowledge_graph is None:
            return
        content = f"[{entry.entry_type}] {entry.description} (actor={entry.actor}, method={entry.method})"
        metadata = {
            "entry_id": entry.id, "trace_id": entry.trace_id, "sequence": entry.sequence,
            "hash": entry.hash, "method": entry.method,
        }
        try:
            self._knowledge_graph.remember(content, node_type="digital_thread_entry", metadata=metadata, source="digital_thread")
        except Exception:
            logger.exception("Failed to mirror digital thread entry into KnowledgeGraph.")

    # ------------------------------------------------------------------
    # Export / persistence
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Full-fidelity export of every entry -- the audit-ready format."""
        with self._lock:
            payload = [self._entry_to_dict(entry) for entry in self._entries]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except OSError as exc:
            raise DigitalThreadError(f"Unable to export digital thread to '{path}'.") from exc
        logger.info("Digital thread exported.", extra={"path": str(path), "entry_count": len(payload)})

    def export_markdown(self, trace_id: str, path: Path) -> None:
        """Human-readable audit report for one trace, in order, goal to
        result."""
        entries = self.get_trace(trace_id)
        if not entries:
            raise EntryNotFoundError(f"No entries found for trace '{trace_id}'.")

        lines = [f"# Digital Thread Report -- trace `{trace_id}`", ""]
        for entry in entries:
            lines.append(f"## [{entry.sequence}] {entry.entry_type} -- {entry.description}")
            lines.append(f"- **Actor:** {entry.actor}")
            lines.append(f"- **Method:** {entry.method}")
            lines.append(f"- **Timestamp:** {entry.timestamp.isoformat()}")
            if entry.validation is not None:
                lines.append(f"- **Validation:** {entry.validation}")
            lines.append(f"- **Result:** {entry.result}")
            lines.append(f"- **Hash:** `{entry.hash[:16]}...`")
            lines.append("")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            raise DigitalThreadError(f"Unable to export report to '{path}'.") from exc
        logger.info("Digital thread report exported.", extra={"path": str(path), "trace_id": trace_id})

    def _persist(self) -> None:
        if self._config.persist_path is None:
            return
        self.export_json(self._config.persist_path)

    def _load(self) -> None:
        if self._config.persist_path is None or not self._config.persist_path.exists():
            return
        try:
            payload = json.loads(self._config.persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DigitalThreadError(f"Unable to load digital thread from '{self._config.persist_path}'.") from exc

        try:
            entries = [self._entry_from_dict(data) for data in payload]
        except (KeyError, ValueError) as exc:
            raise DigitalThreadError(f"Corrupt digital thread file '{self._config.persist_path}'.") from exc

        with self._lock:
            self._entries = entries
            self._sequence_counter = (entries[-1].sequence + 1) if entries else 0
            self._by_id = {entry.id: entry for entry in entries}
            self._by_trace = defaultdict(list)
            self._by_actor = defaultdict(list)
            self._superseded_by = {}
            for entry in entries:
                self._by_trace[entry.trace_id].append(entry.id)
                self._by_actor[entry.actor].append(entry.id)
                if entry.supersedes is not None:
                    self._superseded_by[entry.supersedes] = entry.id

        if not self.verify_integrity():
            raise IntegrityError(f"Digital thread loaded from '{self._config.persist_path}' failed hash-chain verification.")

    @staticmethod
    def _entry_to_dict(entry: DigitalThreadEntry) -> dict[str, Any]:
        return {
            "id": entry.id, "trace_id": entry.trace_id, "entry_type": entry.entry_type, "sequence": entry.sequence,
            "timestamp": entry.timestamp.isoformat(), "actor": entry.actor, "description": entry.description,
            "inputs": entry.inputs, "method": entry.method, "parameters": entry.parameters,
            "validation": entry.validation, "result": entry.result, "parent_entry_id": entry.parent_entry_id,
            "supersedes": entry.supersedes, "previous_hash": entry.previous_hash, "hash": entry.hash,
        }

    @staticmethod
    def _entry_from_dict(data: dict[str, Any]) -> DigitalThreadEntry:
        return DigitalThreadEntry(
            id=data["id"], trace_id=data["trace_id"], entry_type=data["entry_type"], sequence=int(data["sequence"]),
            timestamp=datetime.fromisoformat(data["timestamp"]), actor=data["actor"], description=data["description"],
            inputs=data.get("inputs", {}), method=data.get("method", "unknown"), parameters=data.get("parameters", {}),
            validation=data.get("validation"), result=data.get("result"), parent_entry_id=data.get("parent_entry_id"),
            supersedes=data.get("supersedes"), previous_hash=data["previous_hash"], hash=data["hash"],
        )

    # ------------------------------------------------------------------
    # JSON-safety conversion (handles the heterogeneous result/inputs
    # types flowing in from every bound module's own dataclasses)
    # ------------------------------------------------------------------

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.name
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._json_safe(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._json_safe(item) for item in value]
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "trace_count": len(self._by_trace),
            "chain_valid": self.verify_integrity(),
            "tracking_system_events": bool(self._event_subscriptions),
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "identity_bound": self._identity is not None,
            "persist_path": str(self._config.persist_path) if self._config.persist_path else None,
        }