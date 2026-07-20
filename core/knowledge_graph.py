"""
core/knowledge_graph.py
========================

Knowledge storage and long-term memory subsystem for StarkOS.

Responsibilities
----------------
- Store knowledge as a graph: nodes (concepts, memories, events, entities)
  connected by typed, weighted relations.
- Generate and store embeddings for nodes, enabling semantic search.
- Provide long-term memory: an optional durable (JSON-file) snapshot so
  knowledge survives process restarts, in addition to the in-memory graph.
- Bridge to Identity: ingest its short-term conversation history into
  durable, searchable memory nodes.
- Bridge to Kernel: subscribe to system-level events (kernel/module
  lifecycle) as episodic memory, and snapshot Kernel diagnostics on demand.
- Degrade gracefully when no real embedding model is configured -- falls
  back to deterministic hashing-based vectors, then to plain keyword
  overlap, rather than failing outright.

Design
------
Two concerns are pushed behind small Protocols, exactly like the TTS/STT
split in `core.voice_interface`, so the storage/embedding backend can be
upgraded later (a real vector database, a hosted embedding API) without
touching `KnowledgeGraph` itself:

- `KnowledgeStoreProvider` -- CRUD + persistence for nodes/relations.
  Default: `InMemoryKnowledgeStore`, an in-memory graph with optional
  JSON-file persistence (`persist()`/`load()`).
- `EmbeddingProvider` -- turns text into a fixed-size vector.
  Default: `HashingEmbeddingProvider`, a dependency-free, deterministic
  "hashing trick" embedding (feature hashing with signed buckets,
  normalized to unit length). It captures lexical overlap, not learned
  semantics -- a functional placeholder, not a claim of state-of-the-art
  retrieval quality. Swap in a real sentence-embedding model (local or
  hosted) by implementing `EmbeddingProvider`.

`KnowledgeGraph` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    kg = KnowledgeGraph(services=services, config=KnowledgeGraphConfig(
        persist_path=Path("data/knowledge_graph.json"),
    ))
    kg.bind_kernel(kernel)
    kernel.register_module(kg, name="knowledge_graph", priority=150)

`bind_kernel()`/`bind_identity()` mirror the pattern already used by
`Identity` and `VoiceInterface`: Kernel does not register itself into the
ServiceContainer, so modules that need it receive it via an explicit bind
call from the composition root.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Optional, Protocol, runtime_checkable

from core.event_bus import Event, EventBus
from core.identity import ConversationTurn, Identity
from core.service_container import ServiceContainer

logger = logging.getLogger("starkos.knowledge_graph")

# =============================================================================
# Exceptions
# =============================================================================

class KnowledgeGraphError(Exception):
    """Base exception for KnowledgeGraph failures."""

class NodeNotFoundError(KnowledgeGraphError):
    """Raised when a referenced node does not exist."""

class RelationNotFoundError(KnowledgeGraphError):
    """Raised when a referenced relation does not exist."""

class DuplicateNodeError(KnowledgeGraphError):
    """Raised when a node id is already in use."""

class EmbeddingError(KnowledgeGraphError):
    """Raised when embedding generation fails outright (bad input)."""

class StoreError(KnowledgeGraphError):
    """Raised when the storage backend fails to persist or load data."""

class InvalidQueryError(KnowledgeGraphError):
    """Raised when a query/argument is malformed (empty text, bad params)."""

# =============================================================================
# Data Models
# =============================================================================

@dataclass(slots=True, frozen=True)
class KnowledgeNode:
    """A single unit of knowledge: a concept, memory, event or entity."""

    id: str
    label: str
    content: str
    node_type: str = "concept"
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[tuple[float, ...]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"

@dataclass(slots=True, frozen=True)
class KnowledgeRelation:
    """A typed, weighted, directed edge between two nodes."""

    id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True, frozen=True)
class SearchResult:
    """One ranked hit from a semantic or keyword search."""

    node: KnowledgeNode
    score: float

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class KnowledgeGraphConfig:
    """Runtime configuration for KnowledgeGraph."""

    embedding_dimensions: int = 256
    # If set, the store persists to (and loads from) this JSON file --
    # this is what makes memory "long-term" across process restarts.
    # None means in-memory only (lost on shutdown).
    persist_path: Optional[Path] = None
    persist_on_shutdown: bool = True
    # If True (default) and an EventBus is resolvable, KnowledgeGraph
    # subscribes to `tracked_event_topics` and records each as an
    # episodic memory node -- the system remembering its own history.
    record_system_events: bool = True
    tracked_event_topics: tuple[str, ...] = (
        "kernel.initialized",
        "kernel.running",
        "kernel.stopped",
        "kernel.restarted",
        "module.started",
        "module.stopped",
    )
    max_search_results: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Provider Protocols
# =============================================================================

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into a fixed-size vector. Implementations may block."""

    def embed(self, text: str) -> tuple[float, ...]:
        ...

    def check_available(self) -> bool:
        ...

@runtime_checkable
class KnowledgeStoreProvider(Protocol):
    """CRUD + persistence contract for the underlying graph storage."""

    def save_node(self, node: KnowledgeNode) -> None: ...
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]: ...
    def delete_node(self, node_id: str) -> bool: ...
    def all_nodes(self) -> tuple[KnowledgeNode, ...]: ...

    def save_relation(self, relation: KnowledgeRelation) -> None: ...
    def get_relation(self, relation_id: str) -> Optional[KnowledgeRelation]: ...
    def delete_relation(self, relation_id: str) -> bool: ...
    def relations_of(self, node_id: str) -> tuple[KnowledgeRelation, ...]: ...
    def all_relations(self) -> tuple[KnowledgeRelation, ...]: ...

    def persist(self) -> None: ...
    def load(self) -> None: ...
    def check_available(self) -> bool: ...

# =============================================================================
# Default Embedding Provider
# =============================================================================

class HashingEmbeddingProvider:
    """
    Dependency-free, deterministic embedding via feature hashing (the
    "hashing trick", similar in spirit to scikit-learn's
    HashingVectorizer): each token is hashed into a bucket with a
    hashed sign, and the resulting vector is L2-normalized. No model
    weights, no network, no training -- which also means it captures
    lexical overlap, not learned semantic similarity. It exists so
    semantic search works out of the box on any machine; swap in a real
    sentence-embedding model by implementing `EmbeddingProvider` --
    KnowledgeGraph itself never changes.
    """

    _TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self._dimensions = dimensions

    def check_available(self) -> bool:
        return True

    def embed(self, text: str) -> tuple[float, ...]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        vector = [0.0] * self._dimensions
        tokens = self._TOKEN_PATTERN.findall(text.lower())
        for token in tokens:
            index = self._hash_token(token, salt=0) % self._dimensions
            sign = 1.0 if (self._hash_token(token, salt=1) % 2 == 0) else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm > 0.0:
            vector = [component / norm for component in vector]
        return tuple(vector)

    @staticmethod
    def _hash_token(token: str, *, salt: int) -> int:
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).hexdigest()
        return int(digest, 16)

# =============================================================================
# Default Storage Provider
# =============================================================================

class InMemoryKnowledgeStore:
    """
    Default KnowledgeStoreProvider: the graph lives in memory, with
    optional JSON-file persistence for long-term memory across restarts.
    Not a substitute for a real embedded/vector database at scale (that
    is on the StarkOS roadmap as a dedicated Database Layer) -- this is
    the dependency-free default; swap in another `KnowledgeStoreProvider`
    without touching `KnowledgeGraph`.
    """

    def __init__(self, *, persist_path: Optional[Path] = None) -> None:
        self._nodes: dict[str, KnowledgeNode] = {}
        self._relations: dict[str, KnowledgeRelation] = {}
        self._persist_path = persist_path
        self._lock = RLock()

    def check_available(self) -> bool:
        return True

    # -- Nodes -----------------------------------------------------------

    def save_node(self, node: KnowledgeNode) -> None:
        with self._lock:
            self._nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        with self._lock:
            return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> bool:
        with self._lock:
            existed = self._nodes.pop(node_id, None) is not None
            if existed:
                # Cascade: a dangling relation pointing at a deleted node
                # would corrupt every future traversal, so relations that
                # touch this node are removed along with it.
                doomed = [
                    relation_id
                    for relation_id, relation in self._relations.items()
                    if relation.source_id == node_id or relation.target_id == node_id
                ]
                for relation_id in doomed:
                    del self._relations[relation_id]
            return existed

    def all_nodes(self) -> tuple[KnowledgeNode, ...]:
        with self._lock:
            return tuple(self._nodes.values())

    # -- Relations ---------------------------------------------------------

    def save_relation(self, relation: KnowledgeRelation) -> None:
        with self._lock:
            self._relations[relation.id] = relation

    def get_relation(self, relation_id: str) -> Optional[KnowledgeRelation]:
        with self._lock:
            return self._relations.get(relation_id)

    def delete_relation(self, relation_id: str) -> bool:
        with self._lock:
            return self._relations.pop(relation_id, None) is not None

    def relations_of(self, node_id: str) -> tuple[KnowledgeRelation, ...]:
        with self._lock:
            return tuple(
                relation
                for relation in self._relations.values()
                if relation.source_id == node_id or relation.target_id == node_id
            )

    def all_relations(self) -> tuple[KnowledgeRelation, ...]:
        with self._lock:
            return tuple(self._relations.values())

    # -- Persistence ---------------------------------------------------------

    def persist(self) -> None:
        if self._persist_path is None:
            return

        with self._lock:
            payload = {
                "nodes": [self._node_to_dict(node) for node in self._nodes.values()],
                "relations": [self._relation_to_dict(relation) for relation in self._relations.values()],
            }

        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            tmp_path.replace(self._persist_path)
        except OSError as exc:
            raise StoreError(f"Unable to persist knowledge graph to '{self._persist_path}'.") from exc

    def load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return

        try:
            with self._persist_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"Unable to load knowledge graph from '{self._persist_path}'.") from exc

        try:
            nodes = {entry["id"]: self._node_from_dict(entry) for entry in payload.get("nodes", [])}
            relations = {entry["id"]: self._relation_from_dict(entry) for entry in payload.get("relations", [])}
        except (KeyError, ValueError) as exc:
            raise StoreError(f"Corrupt knowledge graph file '{self._persist_path}'.") from exc

        with self._lock:
            self._nodes = nodes
            self._relations = relations

    @staticmethod
    def _node_to_dict(node: KnowledgeNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "label": node.label,
            "content": node.content,
            "node_type": node.node_type,
            "attributes": node.attributes,
            "embedding": list(node.embedding) if node.embedding is not None else None,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "source": node.source,
        }

    @staticmethod
    def _node_from_dict(data: dict[str, Any]) -> KnowledgeNode:
        embedding = data.get("embedding")
        return KnowledgeNode(
            id=data["id"],
            label=data["label"],
            content=data["content"],
            node_type=data.get("node_type", "concept"),
            attributes=data.get("attributes", {}),
            embedding=tuple(embedding) if embedding is not None else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            source=data.get("source", "unknown"),
        )

    @staticmethod
    def _relation_to_dict(relation: KnowledgeRelation) -> dict[str, Any]:
        return {
            "id": relation.id,
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "relation_type": relation.relation_type,
            "weight": relation.weight,
            "attributes": relation.attributes,
            "created_at": relation.created_at.isoformat(),
        }

    @staticmethod
    def _relation_from_dict(data: dict[str, Any]) -> KnowledgeRelation:
        return KnowledgeRelation(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
        )

# =============================================================================
# Knowledge Graph
# =============================================================================

class KnowledgeGraph:
    """
    StarkOS knowledge storage and long-term memory module.

    Satisfies the `Module` protocol (name/initialize/start/stop) and can
    be registered with the Kernel like any other module. Provides node/
    relation CRUD, graph traversal, semantic (embedding-based, falling
    back to keyword) search, and memory convenience methods (`remember`/
    `recall`) built on top of that.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[KnowledgeGraphConfig] = None,
        store: Optional[KnowledgeStoreProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._services = services
        self._config = config or KnowledgeGraphConfig()
        self._store: KnowledgeStoreProvider = store or InMemoryKnowledgeStore(
            persist_path=self._config.persist_path
        )
        self._embeddings: EmbeddingProvider = embedding_provider or HashingEmbeddingProvider(
            dimensions=self._config.embedding_dimensions
        )

        self._kernel: Any = None
        self._identity: Optional[Identity] = None
        self._event_bus: Optional[EventBus] = None
        self._event_subscriptions: list[Any] = []
        # Optimistic default so the graph is usable standalone (without
        # ever going through initialize()); refined by the real probe
        # once the Module lifecycle does run.
        self._embeddings_available = True

        logger.info(
            "KnowledgeGraph constructed.",
            extra={"embedding_dimensions": self._config.embedding_dimensions, "persist_path": str(self._config.persist_path or "")},
        )

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "knowledge_graph"

    @property
    def node_count(self) -> int:
        return len(self._store.all_nodes())

    @property
    def relation_count(self) -> int:
        return len(self._store.all_relations())

    def bind_kernel(self, kernel: Any) -> None:
        """Kernel does not register itself into the ServiceContainer, so
        it is handed to modules explicitly -- mirrors Identity/VoiceInterface."""
        self._kernel = kernel
        logger.debug("Kernel bound to KnowledgeGraph.")

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to KnowledgeGraph.")

    async def initialize(self) -> None:
        logger.info("Initializing KnowledgeGraph.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._event_bus is None:
            self._event_bus = self._services.resolve_optional(EventBus)

        try:
            await asyncio.to_thread(self._store.load)
        except StoreError:
            logger.exception("Failed to load persisted knowledge graph -- starting from an empty graph.")

        self._embeddings_available = await self._probe(self._embeddings)
        if not self._embeddings_available:
            logger.warning("Embedding provider unavailable -- semantic search will fall back to keyword matching.")

        if self._config.record_system_events and self._event_bus is not None:
            self._subscribe_to_events()

        logger.info(
            "KnowledgeGraph initialized.",
            extra={"node_count": self.node_count, "relation_count": self.relation_count},
        )

    async def start(self) -> None:
        logger.info("KnowledgeGraph ready.", extra={"node_count": self.node_count})

    async def stop(self) -> None:
        logger.info("Stopping KnowledgeGraph.")
        self._unsubscribe_from_events()

        if self._config.persist_on_shutdown:
            try:
                await asyncio.to_thread(self._store.persist)
            except StoreError:
                logger.exception("Failed to persist knowledge graph on shutdown.")

        logger.info("KnowledgeGraph stopped.")

    async def _probe(self, provider: Any) -> bool:
        check = getattr(provider, "check_available", None)
        if check is None:
            return True
        try:
            return bool(await asyncio.to_thread(check))
        except Exception:
            logger.exception("Embedding provider availability probe failed.")
            return False

    # ------------------------------------------------------------------
    # Kernel event integration (episodic memory)
    # ------------------------------------------------------------------

    def _subscribe_to_events(self) -> None:
        if self._event_bus is None or self._event_subscriptions:
            # Already subscribed (e.g. initialize() called twice) --
            # EventBus's own duplicate check compares handler identity,
            # which is unreliable for bound methods, so we guard here.
            return

        for topic in self._config.tracked_event_topics:
            try:
                subscription = self._event_bus.subscribe(topic, self._on_system_event)
                self._event_subscriptions.append(subscription)
            except Exception:
                logger.exception("Failed to subscribe to event topic '%s'.", topic)

        logger.info(
            "Subscribed to system events for episodic memory.",
            extra={"topics": list(self._config.tracked_event_topics)},
        )

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
            self.remember(
                f"System event: {event.topic}",
                node_type="event",
                metadata={"topic": event.topic, "source": event.source, "payload": event.payload},
                source="event_bus",
            )
        except KnowledgeGraphError:
            logger.exception("Failed to record system event '%s' as memory.", event.topic)

    async def snapshot_kernel_state(self) -> Optional[KnowledgeNode]:
        """
        Capture the Kernel's current diagnostics as a single memory node
        -- useful for periodic snapshots, or right before a risky
        operation (restart, shutdown). Returns None if no Kernel is bound.
        """
        if self._kernel is None:
            logger.warning("snapshot_kernel_state() called with no Kernel bound.")
            return None

        diagnostics = self._kernel.diagnostics()
        content = (
            f"Kernel snapshot: state={diagnostics.get('kernel_state')}, "
            f"modules={diagnostics.get('registered_modules')}, "
            f"services={diagnostics.get('registered_services')}"
        )
        return self.remember(content, node_type="kernel_snapshot", metadata=diagnostics, source="kernel")

    # ------------------------------------------------------------------
    # Identity integration (conversation -> long-term memory)
    # ------------------------------------------------------------------

    def ingest_identity_history(self, *, limit: Optional[int] = None) -> int:
        """
        Pull Identity's short-term conversation history (`session_history()`)
        into long-term memory as linked nodes. Returns how many turns were
        newly ingested. Safe to call repeatedly: each turn gets a
        deterministic id derived from its speaker/timestamp/message, so
        re-ingesting the same session never duplicates nodes.
        """
        if self._identity is None:
            raise KnowledgeGraphError("No Identity bound -- call bind_identity() first.")

        turns = self._identity.session_history()
        if limit is not None:
            turns = turns[-limit:]

        ingested = 0
        previous_id: Optional[str] = None
        for turn in turns:
            turn_id = self._deterministic_turn_id(turn)
            if self._store.get_node(turn_id) is not None:
                previous_id = turn_id
                continue

            node = self.add_node(
                label=f"{turn.speaker}: {turn.message[:40]}",
                content=turn.message,
                node_type="conversation_turn",
                attributes={"speaker": turn.speaker, "timestamp": turn.timestamp.isoformat()},
                source="identity",
                node_id=turn_id,
            )
            if previous_id is not None:
                try:
                    self.add_relation(previous_id, node.id, "followed_by")
                except KnowledgeGraphError:
                    logger.debug("Could not link conversation turn to its predecessor.")
            previous_id = node.id
            ingested += 1

        logger.info("Identity conversation history ingested.", extra={"ingested": ingested})
        return ingested

    @staticmethod
    def _deterministic_turn_id(turn: ConversationTurn) -> str:
        raw = f"{turn.speaker}|{turn.timestamp.isoformat()}|{turn.message}"
        return "turn-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(
        self,
        label: str,
        content: str,
        *,
        node_type: str = "concept",
        attributes: Optional[dict[str, Any]] = None,
        source: str = "unknown",
        node_id: Optional[str] = None,
    ) -> KnowledgeNode:
        if not label or not content:
            raise InvalidQueryError("Both label and content are required to add a node.")

        identifier = node_id or str(uuid.uuid4())
        if self._store.get_node(identifier) is not None:
            raise DuplicateNodeError(f"Node '{identifier}' already exists.")

        node = KnowledgeNode(
            id=identifier,
            label=label,
            content=content,
            node_type=node_type,
            attributes=attributes or {},
            embedding=self._safe_embed(content),
            source=source,
        )
        self._store.save_node(node)
        logger.info("Node added.", extra={"node_id": identifier, "node_type": node_type})
        return node

    def get_node(self, node_id: str) -> KnowledgeNode:
        node = self._store.get_node(node_id)
        if node is None:
            raise NodeNotFoundError(f"Node '{node_id}' not found.")
        return node

    def get_node_optional(self, node_id: str) -> Optional[KnowledgeNode]:
        return self._store.get_node(node_id)

    def update_node(
        self,
        node_id: str,
        *,
        content: Optional[str] = None,
        label: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> KnowledgeNode:
        node = self.get_node(node_id)

        new_content = content if content is not None else node.content
        new_label = label if label is not None else node.label
        new_attributes = dict(node.attributes)
        if attributes:
            new_attributes.update(attributes)

        embedding = node.embedding
        if content is not None and content != node.content:
            embedding = self._safe_embed(new_content)

        updated = KnowledgeNode(
            id=node.id,
            label=new_label,
            content=new_content,
            node_type=node.node_type,
            attributes=new_attributes,
            embedding=embedding,
            created_at=node.created_at,
            updated_at=datetime.utcnow(),
            source=node.source,
        )
        self._store.save_node(updated)
        logger.info("Node updated.", extra={"node_id": node_id})
        return updated

    def remove_node(self, node_id: str) -> bool:
        removed = self._store.delete_node(node_id)
        if removed:
            logger.info("Node removed.", extra={"node_id": node_id})
        return removed

    def all_nodes(self) -> tuple[KnowledgeNode, ...]:
        return self._store.all_nodes()

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        weight: float = 1.0,
        attributes: Optional[dict[str, Any]] = None,
        relation_id: Optional[str] = None,
    ) -> KnowledgeRelation:
        if self._store.get_node(source_id) is None:
            raise NodeNotFoundError(f"Source node '{source_id}' not found.")
        if self._store.get_node(target_id) is None:
            raise NodeNotFoundError(f"Target node '{target_id}' not found.")
        if not relation_type:
            raise InvalidQueryError("relation_type is required.")

        identifier = relation_id or str(uuid.uuid4())
        relation = KnowledgeRelation(
            id=identifier,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            attributes=attributes or {},
        )
        self._store.save_relation(relation)
        logger.info("Relation added.", extra={"relation_id": identifier, "relation_type": relation_type})
        return relation

    def get_relation(self, relation_id: str) -> KnowledgeRelation:
        relation = self._store.get_relation(relation_id)
        if relation is None:
            raise RelationNotFoundError(f"Relation '{relation_id}' not found.")
        return relation

    def remove_relation(self, relation_id: str) -> bool:
        removed = self._store.delete_relation(relation_id)
        if removed:
            logger.info("Relation removed.", extra={"relation_id": relation_id})
        return removed

    def all_relations(self) -> tuple[KnowledgeRelation, ...]:
        return self._store.all_relations()

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        *,
        relation_type: Optional[str] = None,
        direction: str = "both",
    ) -> tuple[KnowledgeNode, ...]:
        if direction not in ("out", "in", "both"):
            raise InvalidQueryError("direction must be 'out', 'in' or 'both'.")
        self.get_node(node_id)  # raises NodeNotFoundError if missing

        neighbor_ids: set[str] = set()
        for relation in self._store.relations_of(node_id):
            if relation_type is not None and relation.relation_type != relation_type:
                continue
            if direction in ("out", "both") and relation.source_id == node_id:
                neighbor_ids.add(relation.target_id)
            if direction in ("in", "both") and relation.target_id == node_id:
                neighbor_ids.add(relation.source_id)

        neighbors = (self._store.get_node(nid) for nid in neighbor_ids)
        return tuple(node for node in neighbors if node is not None)

    def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 6,
    ) -> Optional[tuple[str, ...]]:
        """Breadth-first search for the shortest relation path between
        two nodes, ignoring relation direction. Returns None if no path
        exists within `max_depth` hops."""
        self.get_node(source_id)
        self.get_node(target_id)
        if source_id == target_id:
            return (source_id,)

        visited = {source_id}
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(source_id, (source_id,))])

        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= max_depth:
                continue

            for relation in self._store.relations_of(current):
                if relation.source_id == current:
                    next_id = relation.target_id
                elif relation.target_id == current:
                    next_id = relation.source_id
                else:
                    continue

                if next_id in visited:
                    continue
                if next_id == target_id:
                    return path + (next_id,)

                visited.add(next_id)
                queue.append((next_id, path + (next_id,)))

        return None

    # ------------------------------------------------------------------
    # Semantic / keyword search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        node_type: Optional[str] = None,
    ) -> tuple[SearchResult, ...]:
        if not query or not query.strip():
            raise InvalidQueryError("Search query cannot be empty.")
        if top_k <= 0:
            raise InvalidQueryError("top_k must be positive.")
        top_k = min(top_k, self._config.max_search_results)

        candidates = [node for node in self._store.all_nodes() if node_type is None or node.node_type == node_type]
        if not candidates:
            return ()

        if self._embeddings_available:
            try:
                query_vector = self._embeddings.embed(query)
            except Exception:
                logger.exception("Query embedding failed -- falling back to keyword search.")
                return self._keyword_search(query, candidates, top_k)

            scored = [
                SearchResult(node=node, score=self._cosine_similarity(query_vector, node.embedding))
                for node in candidates
                if node.embedding is not None
            ]
            scored.sort(key=lambda result: result.score, reverse=True)
            return tuple(scored[:top_k])

        return self._keyword_search(query, candidates, top_k)

    async def search_async(
        self,
        query: str,
        *,
        top_k: int = 5,
        node_type: Optional[str] = None,
    ) -> tuple[SearchResult, ...]:
        """Async wrapper for `search()`, useful for very large graphs
        where the brute-force scan is heavy enough to want off the event
        loop thread."""
        return await asyncio.to_thread(self.search, query, top_k=top_k, node_type=node_type)

    def _keyword_search(
        self,
        query: str,
        candidates: list[KnowledgeNode],
        top_k: int,
    ) -> tuple[SearchResult, ...]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[SearchResult] = []
        for node in candidates:
            content_tokens = set(re.findall(r"[a-z0-9]+", f"{node.label} {node.content}".lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap > 0:
                scored.append(SearchResult(node=node, score=float(overlap)))
        scored.sort(key=lambda result: result.score, reverse=True)
        return tuple(scored[:top_k])

    @staticmethod
    def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _safe_embed(self, text: str) -> Optional[tuple[float, ...]]:
        if not self._embeddings_available:
            return None
        try:
            return self._embeddings.embed(text)
        except Exception:
            logger.exception("Embedding generation failed for a node -- storing without a vector.")
            return None

    # ------------------------------------------------------------------
    # Long-term memory convenience API
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        node_type: str = "memory",
        metadata: Optional[dict[str, Any]] = None,
        source: str = "unknown",
        link_to: Optional[str] = None,
        relation_type: str = "related_to",
    ) -> KnowledgeNode:
        """Store a fact/observation as a long-term memory node. Optionally
        link it to an existing node (e.g. the memory's subject)."""
        label = content if len(content) <= 60 else content[:57] + "..."
        node = self.add_node(label=label, content=content, node_type=node_type, attributes=metadata or {}, source=source)

        if link_to is not None and self._store.get_node(link_to) is not None:
            self.add_relation(link_to, node.id, relation_type)

        return node

    def recall(self, query: str, *, top_k: int = 5) -> tuple[SearchResult, ...]:
        """Semantic recall over everything remembered so far."""
        return self.search(query, top_k=top_k, node_type=None)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "relation_count": self.relation_count,
            "embeddings_available": self._embeddings_available,
            "tracking_system_events": bool(self._event_subscriptions),
            "persist_path": str(self._config.persist_path) if self._config.persist_path else None,
        }