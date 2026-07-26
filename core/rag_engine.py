"""
core/rag_engine.py
=====================

Retrieval-Augmented Generation engine for StarkOS.

Responsibilities
----------------
- Hybrid retrieval: combine KnowledgeGraph's semantic (embedding) search
  with an independent keyword pass via Reciprocal Rank Fusion (RRF) --
  a real, well-established IR technique for merging ranked lists without
  needing to calibrate scores from different scales against each other.
- A transparent, multi-factor ranker: retrieval score, recency, active-
  project match and tag match, each a named, inspectable contribution
  to the final score -- never an opaque single number.
- A context builder with a real size budget (characters, plus a token
  *estimate* -- see the honesty note below), so a downstream consumer
  never receives an unboundedly large context.
- Filtering by node type, tags, age and active project.
- Answer synthesis with sources and a confidence estimate -- see the
  honesty note on what "generation" and "confidence" actually mean here.
- Integration with KnowledgeGraph (the retrieval backend) and
  ObsidianBridge (reads its active-context setting to bias ranking).

Honesty about scope
--------------------
1. **StarkOS has no LLM, so "generation" here is extractive, not
   generative, by default.** `ExtractiveSynthesizer` (the shipped
   default) does not synthesize new prose with a language model -- it
   surfaces the most relevant *stored* passage(s) directly and labels
   itself `synthesis_method = "extractive"`. The `GenerationProvider`
   Protocol exists precisely so a real LLM call (once StarkOS has an AI
   Runtime) can be wired in via `set_generation_provider()` and self-
   report `synthesis_method = "generated"` -- `RAGEngine` itself never
   claims more than whatever provider is actually configured declares.

2. **"Confidence" is a heuristic derived from real retrieval scores, not
   a calibrated probability.** It's computed from the top result's rank
   score and its margin over the runner-up -- genuine signal, honestly
   summarized, but "confidence: 0.8" here means "retrieval was strong
   and had a clear leader," not "80% chance this is correct." The
   `confidence_basis` field on every `RAGAnswer` spells out exactly how
   the number was derived so it's never a black box.

3. **Token counts are an estimate (~4 characters/token), not real
   tokenization**, unless a real tokenizer is wired in via the
   `TokenEstimator` Protocol. This is the same rough heuristic commonly
   used for quick English-text budgeting; it will be off for other
   languages, code, or unusual text -- treat `context_used_tokens_estimate`
   as a budgeting aid, not an exact count for any specific model.

Design
------
Same shape as the rest of StarkOS's cognitive stack: real, honest logic
where it's genuinely achievable (retrieval, fusion, ranking, budgeting
are all fully real, deterministic, testable operations), and a Protocol
boundary (`GenerationProvider`, `TokenEstimator`) exactly where a
capability StarkOS doesn't have yet (real language generation, exact
tokenization) would otherwise have to be faked.

`RAGEngine` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    rag = RAGEngine(services=services)
    rag.bind_knowledge_graph(knowledge_graph)
    rag.bind_obsidian_bridge(obsidian_bridge)
    kernel.register_module(rag, name="rag_engine", priority=240)

    result = rag.answer(
        "What gearbox ratio did we settle on?",
        filter=RAGFilter(tags=frozenset({"engenharia"}), max_age_days=90),
    )
    print(result.answer_text, result.confidence, [s.label for s in result.sources])
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from core.identity import Identity
from core.knowledge_graph import InvalidQueryError, KnowledgeGraph, KnowledgeNode
from core.logger import get_logger
from core.obsidian_bridge import ObsidianBridge
from core.service_container import ServiceContainer

logger = get_logger("rag_engine")

# =============================================================================
# Exceptions
# =============================================================================

class RAGEngineError(Exception):
    """Base exception for RAGEngine failures."""

class InvalidRAGQueryError(RAGEngineError):
    """Raised when a query/filter argument is malformed."""

class RetrievalError(RAGEngineError):
    """Raised when the retrieval backend fails outright."""

# =============================================================================
# Data models
# =============================================================================

@dataclass(slots=True, frozen=True)
class RetrievalHit:
    """One candidate surfaced by hybrid retrieval, before ranking."""

    node: KnowledgeNode
    fused_score: float  # Reciprocal Rank Fusion score across semantic + keyword rankings

@dataclass(slots=True, frozen=True)
class RankedResult:
    """One retrieval hit after the multi-factor ranker has scored it.
    `factors` is the full, transparent breakdown -- nothing is folded
    into an opaque number without also being inspectable."""

    node: KnowledgeNode
    retrieval_score: float
    rank_score: float
    factors: dict[str, float]

@dataclass(slots=True, frozen=True)
class SourceCitation:
    node_id: str
    label: str
    node_type: str
    score: float
    excerpt: str

@dataclass(slots=True, frozen=True)
class RAGAnswer:
    query: str
    answer_text: str
    synthesis_method: str  # self-declared by the GenerationProvider used
    confidence: float
    confidence_basis: str
    sources: tuple[SourceCitation, ...]
    context_used_chars: int
    context_used_tokens_estimate: int
    generated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True)
class RAGFilter:
    """Hard filters applied before ranking, plus the active-project hint
    used to bias (not necessarily exclude) results during ranking."""

    node_types: Optional[Sequence[str]] = None
    tags: Optional[frozenset[str]] = None
    max_age_days: Optional[float] = None
    active_project: Optional[str] = None
    require_active_project: bool = False

@dataclass(slots=True, frozen=True)
class RankWeights:
    retrieval: float = 5.0
    recency: float = 1.5
    active_project: float = 2.0
    tag_match: float = 1.0

# =============================================================================
# Token estimation (real tokenizer if available, honest heuristic otherwise)
# =============================================================================

@runtime_checkable
class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...
    def check_available(self) -> bool: ...

class HeuristicTokenEstimator:
    """~4 characters per token -- a common, honest approximation for
    English text, not real tokenization. See module docstring point 3."""

    def __init__(self, *, chars_per_token: float = 4.0) -> None:
        self._chars_per_token = max(chars_per_token, 0.1)

    def check_available(self) -> bool:
        return True

    def estimate(self, text: str) -> int:
        return int(len(text) / self._chars_per_token)

# =============================================================================
# Generation (extractive by default -- see honesty note above)
# =============================================================================

@runtime_checkable
class GenerationProvider(Protocol):
    """Turns retrieved context + a query into an answer. Must self-
    declare `synthesis_method` honestly ("extractive" | "generated" |
    whatever else accurately describes it) -- RAGEngine never overrides
    or embellishes this."""

    synthesis_method: str

    def generate(self, query: str, context: str, sources: Sequence[SourceCitation]) -> str: ...
    def check_available(self) -> bool: ...

class ExtractiveSynthesizer:
    """
    Default GenerationProvider: does not generate new text with a
    language model (StarkOS has none) -- it surfaces the most relevant
    stored passage(s) directly. `synthesis_method` is always
    "extractive". Wire in a real LLM-backed provider (once StarkOS has
    an AI Runtime) via `RAGEngine.set_generation_provider()`.
    """

    synthesis_method = "extractive"

    def check_available(self) -> bool:
        return True

    def generate(self, query: str, context: str, sources: Sequence[SourceCitation]) -> str:
        if not sources:
            return "No relevant information was found in the knowledge graph for this query."

        lead = sources[0]
        text = f"Based on the most relevant stored record (\"{lead.label}\"): {lead.excerpt}"
        if len(sources) > 1:
            text += f"\n\n{len(sources) - 1} additional related record(s) were also found -- see sources for details."
        return text

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class RAGEngineConfig:
    default_top_k: int = 5
    candidate_pool_multiplier: int = 4  # retrieve this many x top_k candidates for the ranker to reorder
    rrf_k: int = 60  # standard Reciprocal Rank Fusion constant
    # RRF only looks at RANK POSITION, not the underlying score magnitude
    # -- without a floor, the "best" semantic match would always count
    # for something even when nothing is actually relevant (there's
    # always a rank-0 result if the graph isn't empty). This threshold
    # excludes semantic candidates below it from the ranking entirely,
    # so an irrelevant query can honestly come back with no matches.
    min_semantic_score: float = 0.05
    max_context_chars: int = 6000
    max_context_tokens_estimate: Optional[int] = 1500
    excerpt_chars: int = 500
    recency_window_days: float = 30.0
    rank_weights: RankWeights = field(default_factory=RankWeights)
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# RAG Engine
# =============================================================================

class RAGEngine:
    """
    StarkOS Retrieval-Augmented Generation engine. See the module
    docstring's "Honesty about scope" section before treating
    `answer_text`/`confidence` as more than what they actually are.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[RAGEngineConfig] = None,
        generation_provider: Optional[GenerationProvider] = None,
        token_estimator: Optional[TokenEstimator] = None,
    ) -> None:
        self._services = services
        self._config = config or RAGEngineConfig()
        self._synthesizer: GenerationProvider = generation_provider or ExtractiveSynthesizer()
        self._token_estimator: TokenEstimator = token_estimator or HeuristicTokenEstimator()

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._obsidian_bridge: Optional[ObsidianBridge] = None
        self._synthesizer_available = True

        logger.info("RAGEngine constructed.", extra={"synthesis_method": self._synthesizer.synthesis_method})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rag_engine"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to RAGEngine.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to RAGEngine.")

    def bind_obsidian_bridge(self, obsidian_bridge: ObsidianBridge) -> None:
        self._obsidian_bridge = obsidian_bridge
        logger.debug("ObsidianBridge bound to RAGEngine.")

    def set_generation_provider(self, provider: GenerationProvider) -> None:
        self._synthesizer = provider
        logger.info("Generation provider configured.", extra={"synthesis_method": provider.synthesis_method})

    async def initialize(self) -> None:
        logger.info("Initializing RAGEngine.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._obsidian_bridge is None:
            self._obsidian_bridge = self._services.resolve_optional(ObsidianBridge)

        self._synthesizer_available = await self._probe(self._synthesizer)
        if not self._synthesizer_available:
            logger.warning("Configured generation provider is unavailable -- answer() will raise until fixed.")

        logger.info(
            "RAGEngine initialized.",
            extra={"knowledge_graph_bound": self._knowledge_graph is not None, "synthesis_method": self._synthesizer.synthesis_method},
        )

    async def start(self) -> None:
        logger.info("RAGEngine ready.")

    async def stop(self) -> None:
        logger.info("RAGEngine stopped.")

    async def _probe(self, provider: Any) -> bool:
        check = getattr(provider, "check_available", None)
        if check is None:
            return True
        try:
            return bool(await asyncio.to_thread(check))
        except Exception:
            logger.exception("Availability probe failed for a RAG provider.")
            return False

    # ------------------------------------------------------------------
    # Retrieval: hybrid (semantic + keyword) via Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _semantic_rank(self, query: str, top_k: int, node_type: Optional[str]) -> list[tuple[str, int]]:
        assert self._knowledge_graph is not None
        try:
            results = self._knowledge_graph.search(query, top_k=top_k, node_type=node_type)
        except InvalidQueryError:
            raise
        except Exception:
            logger.exception("Semantic retrieval failed -- continuing with keyword ranking alone.")
            return []
        # Filter by raw score BEFORE assigning rank positions -- RRF only
        # sees rank, not magnitude, so a floor here is what keeps a
        # genuinely irrelevant query from still "winning" a rank-0 slot.
        relevant = [result for result in results if result.score >= self._config.min_semantic_score]
        return [(result.node.id, rank) for rank, result in enumerate(relevant)]

    @staticmethod
    def _keyword_rank(query: str, candidates: Sequence[KnowledgeNode]) -> list[tuple[str, int]]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return []
        scored = []
        for node in candidates:
            content_tokens = set(re.findall(r"[a-z0-9]+", f"{node.label} {node.content}".lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap > 0:
                scored.append((node.id, overlap))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(node_id, rank) for rank, (node_id, _overlap) in enumerate(scored)]

    def _reciprocal_rank_fusion(self, *rankings: Sequence[tuple[str, int]]) -> dict[str, float]:
        """
        Combine multiple ranked lists into one score per item using
        Reciprocal Rank Fusion: score += 1/(rrf_k + rank + 1) for each
        ranking an item appears in. This sidesteps the problem of
        semantic cosine-similarity and keyword-overlap-count living on
        completely different scales -- only rank position matters, so
        nothing needs score normalization/calibration to be combined
        fairly. A well-established technique (used in, e.g.,
        Elasticsearch's own RRF support), not something invented here.
        """
        scores: dict[str, float] = {}
        for ranking in rankings:
            for node_id, rank in ranking:
                scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (self._config.rrf_k + rank + 1)
        return scores

    def _apply_hard_filters(self, nodes: Sequence[KnowledgeNode], rag_filter: RAGFilter) -> tuple[KnowledgeNode, ...]:
        filtered = list(nodes)

        if rag_filter.node_types:
            wanted_types = set(rag_filter.node_types)
            filtered = [node for node in filtered if node.node_type in wanted_types]

        if rag_filter.max_age_days is not None:
            now = datetime.utcnow()
            filtered = [
                node for node in filtered
                if (now - node.updated_at).total_seconds() / 86400.0 <= rag_filter.max_age_days
            ]

        if rag_filter.tags:
            wanted_tags = {tag.lower() for tag in rag_filter.tags}
            filtered = [node for node in filtered if wanted_tags & self._node_tags(node)]

        if rag_filter.require_active_project and rag_filter.active_project:
            project_lower = rag_filter.active_project.lower()
            filtered = [
                node for node in filtered
                if project_lower in self._node_tags(node) or project_lower in node.label.lower()
            ]

        return tuple(filtered)

    @staticmethod
    def _node_tags(node: KnowledgeNode) -> set[str]:
        raw_tags = node.attributes.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        try:
            return {str(tag).lower() for tag in raw_tags}
        except TypeError:
            return set()

    # ------------------------------------------------------------------
    # Ranking: transparent multi-factor scoring
    # ------------------------------------------------------------------

    def _rank(self, hits: Sequence[RetrievalHit], rag_filter: RAGFilter) -> tuple[RankedResult, ...]:
        weights = self._config.rank_weights
        ranked = []

        for hit in hits:
            factors: dict[str, float] = {"retrieval": hit.fused_score * weights.retrieval}

            age_days = max(0.0, (datetime.utcnow() - hit.node.updated_at).total_seconds() / 86400.0)
            window = max(self._config.recency_window_days, 1e-6)
            factors["recency"] = max(0.0, 1.0 - age_days / window) * weights.recency

            node_tags = self._node_tags(hit.node)
            if rag_filter.active_project:
                project_lower = rag_filter.active_project.lower()
                matches = project_lower in node_tags or project_lower in hit.node.label.lower()
                factors["active_project"] = weights.active_project if matches else 0.0
            else:
                factors["active_project"] = 0.0

            if rag_filter.tags:
                overlap = len(node_tags & {tag.lower() for tag in rag_filter.tags})
                factors["tag_match"] = weights.tag_match * overlap
            else:
                factors["tag_match"] = 0.0

            ranked.append(
                RankedResult(node=hit.node, retrieval_score=hit.fused_score, rank_score=sum(factors.values()), factors=factors)
            )

        ranked.sort(key=lambda result: result.rank_score, reverse=True)
        return tuple(ranked)

    # ------------------------------------------------------------------
    # Context builder (real size budget)
    # ------------------------------------------------------------------

    def build_context(
        self,
        ranked: Sequence[RankedResult],
        *,
        max_chars: Optional[int] = None,
        max_tokens_estimate: Optional[int] = None,
    ) -> tuple[str, tuple[SourceCitation, ...]]:
        """
        Assemble the top-ranked results into a plain-text context block
        under a hard character budget and an estimated token budget --
        see the module docstring's honesty note on what the token
        estimate actually is. Never exceeds either budget; stops adding
        results once the next one would.
        """
        effective_max_chars = max_chars if max_chars is not None else self._config.max_context_chars
        effective_max_tokens = max_tokens_estimate if max_tokens_estimate is not None else self._config.max_context_tokens_estimate

        lines: list[str] = []
        sources: list[SourceCitation] = []
        used_chars = 0

        for result in ranked:
            excerpt = result.node.content[: self._config.excerpt_chars]
            line = f"- [{result.node.node_type}] {result.node.label}: {excerpt}"
            projected_chars = used_chars + len(line) + 1

            if projected_chars > effective_max_chars:
                break
            if effective_max_tokens is not None and self._token_estimator.estimate("\n".join(lines + [line])) > effective_max_tokens:
                break

            lines.append(line)
            used_chars = projected_chars
            sources.append(
                SourceCitation(
                    node_id=result.node.id, label=result.node.label, node_type=result.node.node_type,
                    score=round(result.rank_score, 4), excerpt=excerpt,
                )
            )

        return "\n".join(lines), tuple(sources)

    # ------------------------------------------------------------------
    # Confidence (heuristic over real retrieval signal -- see honesty note)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_confidence(ranked: Sequence[RankedResult]) -> tuple[float, str]:
        if not ranked:
            return 0.0, "No candidates matched either the semantic or keyword retrieval pass."

        top = ranked[0].rank_score
        margin = top - ranked[1].rank_score if len(ranked) > 1 else top

        normalized_top = min(1.0, top / (top + 2.0)) if top > 0 else 0.0
        normalized_margin = min(1.0, margin / (abs(top) + 1e-6)) if top != 0 else 0.0
        confidence = round(0.7 * normalized_top + 0.3 * normalized_margin, 3)

        basis = (
            f"Derived from the top result's rank score ({top:.3f}) and its margin over the runner-up "
            f"({margin:.3f}) -- a heuristic over real retrieval signal, not a calibrated probability."
        )
        return confidence, basis

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def answer(self, query: str, *, filter: Optional[RAGFilter] = None, top_k: Optional[int] = None) -> RAGAnswer:
        if not query or not query.strip():
            raise InvalidRAGQueryError("Query cannot be empty.")
        if self._knowledge_graph is None:
            raise RAGEngineError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")
        if not self._synthesizer_available:
            raise RAGEngineError("Configured generation provider is unavailable.")

        effective_top_k = top_k if top_k is not None else self._config.default_top_k
        if effective_top_k <= 0:
            raise InvalidRAGQueryError("top_k must be positive.")

        active_filter = filter or RAGFilter(active_project=self._resolve_active_project())
        pool_size = effective_top_k * self._config.candidate_pool_multiplier

        node_type_hint = active_filter.node_types[0] if active_filter.node_types and len(active_filter.node_types) == 1 else None
        semantic_ranking = self._semantic_rank(query, pool_size, node_type_hint)

        try:
            all_nodes = self._apply_hard_filters(self._knowledge_graph.all_nodes(), active_filter)
        except Exception as exc:
            raise RetrievalError("Failed to enumerate candidate nodes.") from exc
        keyword_ranking = self._keyword_rank(query, all_nodes)

        fused_scores = self._reciprocal_rank_fusion(semantic_ranking, keyword_ranking)

        node_lookup = {node.id: node for node in all_nodes}
        hits = []
        for node_id, fused_score in fused_scores.items():
            node = node_lookup.get(node_id)
            if node is None:
                # Matched semantically but excluded by a hard filter (or
                # belongs to a different node_type pool) -- correctly skip it.
                continue
            hits.append(RetrievalHit(node=node, fused_score=fused_score))
        hits.sort(key=lambda hit: hit.fused_score, reverse=True)
        hits = hits[:pool_size]

        ranked = self._rank(hits, active_filter)[:effective_top_k]
        context, sources = self.build_context(ranked)
        confidence, confidence_basis = self._estimate_confidence(ranked)

        try:
            answer_text = self._synthesizer.generate(query, context, sources)
        except Exception as exc:
            logger.exception("Answer synthesis failed.")
            raise RAGEngineError("Generation provider raised while synthesizing an answer.") from exc

        result = RAGAnswer(
            query=query,
            answer_text=answer_text,
            synthesis_method=self._synthesizer.synthesis_method,
            confidence=confidence,
            confidence_basis=confidence_basis,
            sources=sources,
            context_used_chars=len(context),
            context_used_tokens_estimate=self._token_estimator.estimate(context),
        )

        logger.info(
            "RAG answer produced.",
            extra={"synthesis_method": result.synthesis_method, "confidence": result.confidence, "source_count": len(result.sources)},
        )
        return result

    async def answer_async(self, query: str, *, filter: Optional[RAGFilter] = None, top_k: Optional[int] = None) -> RAGAnswer:
        return await asyncio.to_thread(self.answer, query, filter=filter, top_k=top_k)

    def _resolve_active_project(self) -> Optional[str]:
        if self._obsidian_bridge is None:
            return None
        try:
            return self._obsidian_bridge.diagnostics().get("active_context")
        except Exception:
            logger.exception("Failed to read active context from ObsidianBridge.")
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "obsidian_bridge_bound": self._obsidian_bridge is not None,
            "synthesis_method": self._synthesizer.synthesis_method,
            "synthesizer_available": self._synthesizer_available,
            "default_top_k": self._config.default_top_k,
            "max_context_chars": self._config.max_context_chars,
        }