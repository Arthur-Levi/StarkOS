"""
core/obsidian_bridge.py
========================

Bridge between StarkOS and an Obsidian vault (a folder of plain
Markdown files with YAML frontmatter, `#tags` and `[[wiki-links]]`).

Responsibilities
----------------
- Read a vault: parse every Markdown note's frontmatter, tags (inline
  and frontmatter), wiki-links, and modification time.
- A smart filter that scores notes on priority tags, recency, active
  project/context and (optionally) semantic relevance to a query, then
  keeps only the top-scoring notes under a hard volume cap -- this is
  the actual anti-overload mechanism, not a suggestion.
- Bidirectional sync:
  - Obsidian -> StarkOS: filtered notes become KnowledgeGraph nodes
    (one per note, `node_type="note"`), tags become their own nodes
    with `tagged_with` edges, and `[[wiki-links]]` between selected
    notes become `links_to` edges.
  - StarkOS -> Obsidian: KnowledgeGraph nodes are rendered back out as
    Markdown files with frontmatter, so a human can review StarkOS's
    own memory in their normal note-taking tool.
- Bridge to CognitiveEngine: hand off "review what just synced" as a
  new goal. Bridge to Identity: attribution in logs/events.

Honesty about scope
--------------------
1. **Bidirectional sync never overwrites a note it doesn't own.**
   StarkOS only ever *writes* into a dedicated subfolder of the vault
   (`ObsidianBridgeConfig.output_subfolder`, default "StarkOS/") --
   never anywhere else. For notes inside that subfolder that it wrote
   before, it hashes the file's current content against the hash
   recorded at the last sync; if they differ, a human edited it since,
   and this module refuses to overwrite it (raises internally, logged
   and skipped) rather than silently clobbering an edit. This is
   conflict *detection*, not automatic merging -- resolving a real
   conflict is left to the human.

2. **Tag/link parsing approximates Obsidian's own syntax, not a byte-
   exact reimplementation of it.** `#tags` and `[[wiki-links]]` cover
   the common cases (including `[[Note#Heading|Display]]`); unusual
   edge cases in Obsidian's own parser may not be replicated exactly.

3. **"Semantic relevance" uses the same honest, dependency-free hashing
   embedding as `core.knowledge_graph` by default** (lexical overlap,
   not learned semantics -- see that module's own honesty note). A real
   embedding model can be substituted via the `embedding_provider`
   constructor argument.

4. **Link resolution only connects notes that were both selected by the
   filter in the same sync pass.** A `[[link]]` to a note that exists
   in the vault but didn't make the cut (or genuinely doesn't exist)
   is simply not turned into a graph edge -- this module doesn't
   silently pull in extra notes just because something links to them,
   since that would undermine the volume cap's entire purpose.

Design
------
Same shape as the rest of StarkOS: real parsing/scoring logic (no
fabrication -- Markdown/YAML parsing and weighted scoring are fully
honest, deterministic operations), pluggable `EmbeddingProvider` for
the semantic-relevance factor, and explicit `bind_*` methods for
Kernel-style dependency injection.

`ObsidianBridge` satisfies the `Module` protocol (name/initialize/
start/stop) and registers with the Kernel like any other module:

    bridge = ObsidianBridge(services=services, vault_path=Path("~/Notes").expanduser())
    bridge.bind_knowledge_graph(knowledge_graph)
    kernel.register_module(bridge, name="obsidian_bridge", priority=230)

    result = await bridge.sync_from_vault(criteria=FilterCriteria(
        active_context="motor-project", semantic_query="gearbox torque requirements",
    ))
    await bridge.sync_to_vault(node_types=("memory", "design_iteration"))
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml

from core.cognitive_engine import CognitiveEngine
from core.identity import Identity
from core.knowledge_graph import EmbeddingProvider, HashingEmbeddingProvider, KnowledgeGraph, KnowledgeGraphError
from core.logger import get_logger
from core.service_container import ServiceContainer

logger = get_logger("obsidian_bridge")

# =============================================================================
# Exceptions
# =============================================================================

class ObsidianBridgeError(Exception):
    """Base exception for ObsidianBridge failures."""

class VaultNotFoundError(ObsidianBridgeError):
    """Raised when the configured vault path doesn't exist/isn't a directory."""

class NoteParsingError(ObsidianBridgeError):
    """Raised when a single note fails to parse (unreadable file, bad YAML)."""

class SyncConflictError(ObsidianBridgeError):
    """Raised internally when a StarkOS-owned vault file was edited by a
    human since the last sync -- caught and logged, never propagated
    out of `sync_to_vault()` as a hard failure for the whole batch."""

# =============================================================================
# Parsing regexes (approximate Obsidian's syntax -- see honesty note above)
# =============================================================================

_FRONTMATTER_PATTERN = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)
_INLINE_TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z][a-zA-Z0-9_\-/]*)")
_WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]")
_UNSAFE_FILENAME_CHARS = re.compile(r'[/\\:*?"<>|]')

# =============================================================================
# Data models
# =============================================================================

@dataclass(slots=True, frozen=True)
class ObsidianLink:
    """One `[[wiki-link]]` found in a note's body."""

    target: str
    heading: Optional[str] = None
    display_text: Optional[str] = None

@dataclass(slots=True, frozen=True)
class ObsidianNote:
    """A single parsed Markdown note from the vault."""

    path: Path  # relative to the vault root
    title: str
    content: str  # body, frontmatter stripped
    raw_content: str  # full file content, for hashing/round-tripping
    frontmatter: dict[str, Any]
    tags: frozenset[str]
    links: tuple[ObsidianLink, ...]
    modified_at: datetime
    content_hash: str

@dataclass(slots=True, frozen=True)
class FilterWeights:
    """Relative weight of each scoring factor. All additive, all >= 0;
    a note's final score is the sum of whichever factors apply."""

    priority_tag: float = 3.0
    active_context: float = 2.5
    recency: float = 2.0
    semantic: float = 4.0

@dataclass(slots=True)
class FilterCriteria:
    """Everything the smart filter needs to score and cap a set of notes."""

    priority_tags: frozenset[str] = field(default_factory=lambda: frozenset({"importante", "projeto", "engenharia"}))
    active_context: Optional[str] = None
    max_age_days: Optional[float] = None  # hard cutoff -- None means no age limit at all
    max_notes: int = 50  # hard cap on note COUNT -- the primary anti-overload guarantee
    max_total_chars: Optional[int] = 200_000  # hard cap on total content VOLUME as well
    semantic_query: Optional[str] = None
    weights: FilterWeights = field(default_factory=FilterWeights)

@dataclass(slots=True, frozen=True)
class SyncRecord:
    """State needed to make a sync idempotent (skip unchanged notes) and
    to detect conflicts (a StarkOS-written file edited by a human)."""

    vault_path: str
    node_id: str
    content_hash: str
    synced_at: datetime
    direction: str  # "from_vault" | "to_vault"

@dataclass(slots=True, frozen=True)
class SyncFromVaultResult:
    notes_seen: int
    notes_selected: int
    created: int
    updated: int
    skipped_unchanged: int
    links_created: int
    dry_run: bool = False

@dataclass(slots=True, frozen=True)
class SyncToVaultResult:
    nodes_considered: int
    written: int
    conflicts: int

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class ObsidianBridgeConfig:
    # StarkOS -> Obsidian writes ONLY ever land here, never anywhere
    # else in the vault -- see the module docstring's honesty note.
    output_subfolder: str = "StarkOS"
    ignore_folders: tuple[str, ...] = (".obsidian", ".trash")
    recency_window_days: float = 30.0
    sync_state_path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

# =============================================================================
# Obsidian Bridge
# =============================================================================

class ObsidianBridge:
    """
    StarkOS <-> Obsidian vault bridge. See the module docstring's
    "Honesty about scope" section, especially point 1 (sync safety),
    before relying on this with a vault you care about.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        vault_path: Union[str, Path],
        config: Optional[ObsidianBridgeConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._services = services
        self._vault_path = Path(vault_path)
        self._config = config or ObsidianBridgeConfig()
        self._embeddings: EmbeddingProvider = embedding_provider or HashingEmbeddingProvider()

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._cognitive_engine: Optional[CognitiveEngine] = None
        self._active_context: Optional[str] = None

        self._sync_records_from_vault: dict[str, SyncRecord] = {}
        self._sync_records_to_vault: dict[str, SyncRecord] = {}
        self._ensured_relations_cache: set[tuple[str, str, str]] = set()

        logger.info("ObsidianBridge constructed.", extra={"vault_path": str(self._vault_path)})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "obsidian_bridge"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to ObsidianBridge.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to ObsidianBridge.")

    def bind_cognitive_engine(self, cognitive_engine: CognitiveEngine) -> None:
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to ObsidianBridge.")

    def set_active_context(self, context: Optional[str]) -> None:
        """Set the default active project/context the smart filter
        biases toward when a FilterCriteria doesn't specify its own."""
        self._active_context = context
        logger.info("Active context set.", extra={"context": context or "none"})

    async def initialize(self) -> None:
        logger.info("Initializing ObsidianBridge.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._cognitive_engine is None:
            self._cognitive_engine = self._services.resolve_optional(CognitiveEngine)

        if not self._vault_path.exists():
            logger.warning("Vault path does not exist yet: '%s' -- sync methods will raise until it does.", self._vault_path)

        try:
            await asyncio.to_thread(self._load_sync_state)
        except ObsidianBridgeError:
            logger.exception("Failed to load persisted sync state -- starting fresh.")

        logger.info(
            "ObsidianBridge initialized.",
            extra={"vault_path": str(self._vault_path), "knowledge_graph_bound": self._knowledge_graph is not None},
        )

    async def start(self) -> None:
        logger.info("ObsidianBridge ready.")

    async def stop(self) -> None:
        logger.info("Stopping ObsidianBridge.")
        try:
            await asyncio.to_thread(self._persist_sync_state)
        except ObsidianBridgeError:
            logger.exception("Failed to persist sync state on shutdown.")
        logger.info("ObsidianBridge stopped.")

    # ------------------------------------------------------------------
    # Vault reading (real Markdown/YAML/tag/link parsing)
    # ------------------------------------------------------------------

    def read_vault(self) -> tuple[ObsidianNote, ...]:
        if not self._vault_path.exists() or not self._vault_path.is_dir():
            raise VaultNotFoundError(f"Vault not found: '{self._vault_path}'.")

        notes: list[ObsidianNote] = []
        for md_path in sorted(self._vault_path.rglob("*.md")):
            relative = md_path.relative_to(self._vault_path)
            if self._is_ignored(relative):
                continue
            try:
                notes.append(self._parse_note(md_path))
            except NoteParsingError:
                logger.exception("Skipping unparseable note '%s'.", relative)

        logger.info("Vault read.", extra={"notes_found": len(notes)})
        return tuple(notes)

    async def read_vault_async(self) -> tuple[ObsidianNote, ...]:
        return await asyncio.to_thread(self.read_vault)

    def _is_ignored(self, relative_path: Path) -> bool:
        ignore_set = set(self._config.ignore_folders) | {self._config.output_subfolder}
        return any(part in ignore_set for part in relative_path.parts)

    def _parse_note(self, path: Path) -> ObsidianNote:
        try:
            raw_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise NoteParsingError(f"'{path}' is not valid UTF-8 text.") from exc
        except OSError as exc:
            raise NoteParsingError(f"Unable to read '{path}'.") from exc

        frontmatter, body = self._parse_frontmatter(raw_content, path)
        tags = self._extract_tags(frontmatter, body)
        links = self._extract_links(body)

        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError as exc:
            raise NoteParsingError(f"Unable to stat '{path}'.") from exc

        title = str(frontmatter.get("title") or path.stem)
        return ObsidianNote(
            path=path.relative_to(self._vault_path),
            title=title,
            content=body.strip(),
            raw_content=raw_content,
            frontmatter=frontmatter,
            tags=frozenset(tags),
            links=links,
            modified_at=modified_at,
            content_hash=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def _parse_frontmatter(raw_content: str, path: Path) -> tuple[dict[str, Any], str]:
        match = _FRONTMATTER_PATTERN.match(raw_content)
        if not match:
            return {}, raw_content
        try:
            parsed = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise NoteParsingError(f"Invalid YAML frontmatter in '{path}': {exc}") from exc
        frontmatter = parsed if isinstance(parsed, dict) else {}
        return frontmatter, raw_content[match.end():]

    @staticmethod
    def _extract_tags(frontmatter: dict[str, Any], body: str) -> set[str]:
        tags: set[str] = set()
        raw_tags = frontmatter.get("tags", frontmatter.get("tag", []))
        if isinstance(raw_tags, str):
            raw_tags = [part.strip() for part in raw_tags.split(",") if part.strip()]
        if isinstance(raw_tags, list):
            tags.update(str(tag).lstrip("#").lower() for tag in raw_tags if str(tag).strip())
        tags.update(match.lower() for match in _INLINE_TAG_PATTERN.findall(body))
        return tags

    @staticmethod
    def _extract_links(body: str) -> tuple[ObsidianLink, ...]:
        return tuple(
            ObsidianLink(
                target=target.strip(),
                heading=heading.strip() if heading else None,
                display_text=display.strip() if display else None,
            )
            for target, heading, display in _WIKILINK_PATTERN.findall(body)
        )

    # ------------------------------------------------------------------
    # Smart filter (priority tags, recency, active context, semantic, volume cap)
    # ------------------------------------------------------------------

    def filter_notes(
        self,
        notes: Sequence[ObsidianNote],
        criteria: Optional[FilterCriteria] = None,
    ) -> tuple[ObsidianNote, ...]:
        """
        Score every note against `criteria` and return only the top
        ones under the hard volume cap (`max_notes` and, if set,
        `max_total_chars`) -- this cap is the actual anti-overload
        guarantee: the result is never larger than it allows, no matter
        how many notes are in the vault.
        """
        active_criteria = criteria or FilterCriteria(active_context=self._active_context)
        if active_criteria.max_notes <= 0:
            raise ObsidianBridgeError("FilterCriteria.max_notes must be positive.")

        candidates = list(notes)
        if active_criteria.max_age_days is not None:
            cutoff_days = active_criteria.max_age_days
            now = datetime.utcnow()
            candidates = [
                note for note in candidates
                if (now - note.modified_at).total_seconds() / 86400.0 <= cutoff_days
            ]

        semantic_scores: dict[Path, float] = {}
        if active_criteria.semantic_query:
            semantic_scores = self._semantic_scores(candidates, active_criteria.semantic_query)

        scored = [(note, self._score_note(note, active_criteria, semantic_scores)) for note in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)

        selected: list[ObsidianNote] = []
        total_chars = 0
        for note, _score in scored:
            if len(selected) >= active_criteria.max_notes:
                break
            note_size = len(note.content)
            if active_criteria.max_total_chars is not None and total_chars + note_size > active_criteria.max_total_chars:
                continue  # too big to fit the remaining budget -- try the next (smaller/lower-scored) note instead
            selected.append(note)
            total_chars += note_size

        logger.info(
            "Notes filtered.",
            extra={"total_candidates": len(notes), "after_age_filter": len(candidates), "selected": len(selected), "total_chars": total_chars},
        )
        return tuple(selected)

    def _score_note(self, note: ObsidianNote, criteria: FilterCriteria, semantic_scores: dict[Path, float]) -> float:
        weights = criteria.weights
        score = 0.0

        if note.tags & criteria.priority_tags:
            score += weights.priority_tag

        if criteria.active_context:
            context_lower = criteria.active_context.lower()
            if context_lower in note.tags or context_lower in note.title.lower():
                score += weights.active_context

        age_days = max(0.0, (datetime.utcnow() - note.modified_at).total_seconds() / 86400.0)
        window = max(self._config.recency_window_days, 1e-6)
        recency_score = max(0.0, 1.0 - age_days / window)
        score += recency_score * weights.recency

        score += semantic_scores.get(note.path, 0.0) * weights.semantic
        return score

    def _semantic_scores(self, notes: Sequence[ObsidianNote], query: str) -> dict[Path, float]:
        try:
            query_vector = self._embeddings.embed(query)
        except Exception:
            logger.exception("Semantic scoring failed for the filter query -- continuing without it.")
            return {}

        scores: dict[Path, float] = {}
        for note in notes:
            try:
                note_vector = self._embeddings.embed(f"{note.title} {note.content[:2000]}")
            except Exception:
                continue
            scores[note.path] = _cosine_similarity(query_vector, note_vector)
        return scores

    # ------------------------------------------------------------------
    # Sync: Obsidian -> StarkOS
    # ------------------------------------------------------------------

    async def sync_from_vault(
        self,
        *,
        criteria: Optional[FilterCriteria] = None,
        dry_run: bool = False,
    ) -> SyncFromVaultResult:
        notes = await self.read_vault_async()
        selected = self.filter_notes(notes, criteria)

        if dry_run:
            would_create = would_update = would_skip = 0
            for note in selected:
                record = self._sync_records_from_vault.get(str(note.path))
                if record is not None and record.content_hash == note.content_hash:
                    would_skip += 1
                elif record is not None:
                    would_update += 1
                else:
                    would_create += 1
            return SyncFromVaultResult(
                notes_seen=len(notes), notes_selected=len(selected),
                created=would_create, updated=would_update, skipped_unchanged=would_skip,
                links_created=0, dry_run=True,
            )

        if self._knowledge_graph is None:
            raise ObsidianBridgeError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")

        return await asyncio.to_thread(self._sync_from_vault_sync, notes, selected)

    def _sync_from_vault_sync(self, notes: Sequence[ObsidianNote], selected: Sequence[ObsidianNote]) -> SyncFromVaultResult:
        assert self._knowledge_graph is not None
        created = updated = skipped_unchanged = 0
        tag_node_cache: dict[str, str] = {}
        path_to_node_id: dict[Path, str] = {}
        title_to_path = {note.title: note.path for note in selected}

        for note in selected:
            node_id = self._deterministic_note_id(note.path)
            path_to_node_id[note.path] = node_id

            existing_record = self._sync_records_from_vault.get(str(note.path))
            if existing_record is not None and existing_record.content_hash == note.content_hash:
                skipped_unchanged += 1
            else:
                existing_node = self._knowledge_graph.get_node_optional(node_id)
                attributes = {"vault_path": str(note.path), "tags": sorted(note.tags), "frontmatter": self._json_safe(note.frontmatter)}
                if existing_node is None:
                    self._knowledge_graph.add_node(
                        label=note.title, content=note.content, node_type="note",
                        attributes=attributes, source="obsidian", node_id=node_id,
                    )
                    created += 1
                else:
                    self._knowledge_graph.update_node(node_id, content=note.content, attributes=attributes)
                    updated += 1

                self._sync_records_from_vault[str(note.path)] = SyncRecord(
                    vault_path=str(note.path), node_id=node_id, content_hash=note.content_hash,
                    synced_at=datetime.utcnow(), direction="from_vault",
                )

            for tag in note.tags:
                tag_node_id = tag_node_cache.get(tag)
                if tag_node_id is None:
                    tag_node_id = self._deterministic_tag_id(tag)
                    if self._knowledge_graph.get_node_optional(tag_node_id) is None:
                        self._knowledge_graph.add_node(
                            label=f"#{tag}", content=f"Tag: {tag}", node_type="tag", node_id=tag_node_id, source="obsidian",
                        )
                    tag_node_cache[tag] = tag_node_id
                self._ensure_relation(node_id, tag_node_id, "tagged_with")

        links_created = 0
        for note in selected:
            source_node_id = path_to_node_id[note.path]
            for link in note.links:
                target_path = title_to_path.get(link.target)
                if target_path is None:
                    continue  # target wasn't selected (or doesn't exist) -- never pull it in just because of a link
                if self._ensure_relation(source_node_id, path_to_node_id[target_path], "links_to"):
                    links_created += 1

        self._persist_sync_state()

        result = SyncFromVaultResult(
            notes_seen=len(notes), notes_selected=len(selected), created=created,
            updated=updated, skipped_unchanged=skipped_unchanged, links_created=links_created,
        )
        logger.info("Sync from vault completed.", extra=dataclasses.asdict(result))
        return result

    def _ensure_relation(self, source_id: str, target_id: str, relation_type: str) -> bool:
        """Create the relation if it doesn't already exist (checked via
        this session's cache first, then the graph itself) -- keeps
        repeated syncs from piling up duplicate edges."""
        assert self._knowledge_graph is not None
        key = (source_id, target_id, relation_type)
        if key in self._ensured_relations_cache:
            return False

        existing = [
            relation for relation in self._knowledge_graph.relations_of(source_id)
            if relation.target_id == target_id and relation.relation_type == relation_type
        ]
        created = False
        if not existing:
            try:
                self._knowledge_graph.add_relation(source_id, target_id, relation_type)
                created = True
            except KnowledgeGraphError:
                logger.debug("Could not create relation %s -> %s (%s).", source_id, target_id, relation_type)

        self._ensured_relations_cache.add(key)
        return created

    @staticmethod
    def _deterministic_note_id(vault_path: Path) -> str:
        return "obsidian-note-" + hashlib.sha256(str(vault_path).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _deterministic_tag_id(tag: str) -> str:
        return "obsidian-tag-" + hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    # ------------------------------------------------------------------
    # Sync: StarkOS -> Obsidian
    # ------------------------------------------------------------------

    async def sync_to_vault(
        self,
        *,
        node_types: Sequence[str] = ("memory", "goal", "design_iteration", "risk_assessment"),
        max_notes: int = 50,
        since: Optional[datetime] = None,
    ) -> SyncToVaultResult:
        """
        Render matching KnowledgeGraph nodes back out as Markdown files
        under `ObsidianBridgeConfig.output_subfolder` -- never anywhere
        else in the vault. A file this bridge previously wrote that was
        since edited by a human is detected and skipped (counted as a
        conflict), never silently overwritten.
        """
        if self._knowledge_graph is None:
            raise ObsidianBridgeError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")

        return await asyncio.to_thread(self._sync_to_vault_sync, node_types, max_notes, since)

    def _sync_to_vault_sync(self, node_types: Sequence[str], max_notes: int, since: Optional[datetime]) -> SyncToVaultResult:
        assert self._knowledge_graph is not None
        output_dir = self._vault_path / self._config.output_subfolder
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ObsidianBridgeError(f"Unable to create output folder '{output_dir}'.") from exc

        candidates = [node for node in self._knowledge_graph.all_nodes() if node.node_type in node_types]
        if since is not None:
            candidates = [node for node in candidates if node.updated_at >= since]
        candidates.sort(key=lambda node: node.updated_at, reverse=True)
        candidates = candidates[:max_notes]

        written = conflicts = 0
        for node in candidates:
            try:
                if self._write_node_as_note(node, output_dir):
                    written += 1
            except SyncConflictError:
                conflicts += 1
                logger.warning(
                    "Skipped writing node '%s' -- the vault file was edited by a human since StarkOS last wrote it.",
                    node.id,
                )
            except OSError:
                logger.exception("Failed to write node '%s' to the vault.", node.id)

        self._persist_sync_state()
        result = SyncToVaultResult(nodes_considered=len(candidates), written=written, conflicts=conflicts)
        logger.info("Sync to vault completed.", extra=dataclasses.asdict(result))
        return result

    def _write_node_as_note(self, node: Any, output_dir: Path) -> bool:
        filename = self._safe_filename(node.label) + ".md"
        file_path = output_dir / filename

        existing_record = self._sync_records_to_vault.get(node.id)
        if existing_record is not None and file_path.exists():
            current_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if current_hash != existing_record.content_hash:
                raise SyncConflictError(f"'{file_path}' was modified outside StarkOS since the last sync.")

        frontmatter = {
            "tags": ["starkos", node.node_type],
            "source": node.source,
            "created": node.created_at.isoformat(),
            "updated": node.updated_at.isoformat(),
            "starkos_node_id": node.id,
        }
        content = f"---\n{yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False)}---\n\n# {node.label}\n\n{node.content}\n"

        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ObsidianBridgeError(f"Unable to write '{file_path}'.") from exc

        self._sync_records_to_vault[node.id] = SyncRecord(
            vault_path=str(file_path.relative_to(self._vault_path)), node_id=node.id,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            synced_at=datetime.utcnow(), direction="to_vault",
        )
        return True

    @staticmethod
    def _safe_filename(label: str) -> str:
        sanitized = _UNSAFE_FILENAME_CHARS.sub("_", label).strip().strip(".")
        return sanitized[:100] if sanitized else "untitled"

    # ------------------------------------------------------------------
    # CognitiveEngine integration
    # ------------------------------------------------------------------

    async def request_next_steps(self, sync_result: SyncFromVaultResult) -> Any:
        """Hand a just-completed sync's outcome to CognitiveEngine as a
        new goal, so it can plan appropriate next steps."""
        if self._cognitive_engine is None:
            raise ObsidianBridgeError("No CognitiveEngine bound -- call bind_cognitive_engine() first.")

        description = (
            f"Review {sync_result.created + sync_result.updated} notes synced from Obsidian "
            f"({sync_result.created} new, {sync_result.updated} updated, {sync_result.links_created} links)."
        )
        return await self._cognitive_engine.pursue_goal(
            description, metadata={"obsidian_sync": dataclasses.asdict(sync_result)}
        )

    # ------------------------------------------------------------------
    # Sync state persistence
    # ------------------------------------------------------------------

    def _persist_sync_state(self) -> None:
        if self._config.sync_state_path is None:
            return
        payload = {
            "from_vault": {path: self._record_to_dict(record) for path, record in self._sync_records_from_vault.items()},
            "to_vault": {node_id: self._record_to_dict(record) for node_id, record in self._sync_records_to_vault.items()},
        }
        try:
            self._config.sync_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._config.sync_state_path.with_suffix(self._config.sync_state_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(self._config.sync_state_path)
        except OSError as exc:
            raise ObsidianBridgeError(f"Unable to persist sync state to '{self._config.sync_state_path}'.") from exc

    def _load_sync_state(self) -> None:
        if self._config.sync_state_path is None or not self._config.sync_state_path.exists():
            return
        try:
            payload = json.loads(self._config.sync_state_path.read_text(encoding="utf-8"))
            self._sync_records_from_vault = {
                path: self._record_from_dict(data) for path, data in payload.get("from_vault", {}).items()
            }
            self._sync_records_to_vault = {
                node_id: self._record_from_dict(data) for node_id, data in payload.get("to_vault", {}).items()
            }
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ObsidianBridgeError(f"Unable to load sync state from '{self._config.sync_state_path}'.") from exc

    @staticmethod
    def _record_to_dict(record: SyncRecord) -> dict[str, Any]:
        return {
            "vault_path": record.vault_path,
            "node_id": record.node_id,
            "content_hash": record.content_hash,
            "synced_at": record.synced_at.isoformat(),
            "direction": record.direction,
        }

    @staticmethod
    def _record_from_dict(data: dict[str, Any]) -> SyncRecord:
        return SyncRecord(
            vault_path=data["vault_path"],
            node_id=data["node_id"],
            content_hash=data["content_hash"],
            synced_at=datetime.fromisoformat(data["synced_at"]),
            direction=data["direction"],
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "vault_path": str(self._vault_path),
            "vault_exists": self._vault_path.exists(),
            "active_context": self._active_context,
            "output_subfolder": self._config.output_subfolder,
            "notes_synced_from_vault": len(self._sync_records_from_vault),
            "nodes_synced_to_vault": len(self._sync_records_to_vault),
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "cognitive_engine_bound": self._cognitive_engine is not None,
        }