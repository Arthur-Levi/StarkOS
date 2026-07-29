"""
core/autonomous_engineering_loop.py
======================================

The Autonomous Engineering Loop: StarkOS's top-level coordinator for
turning a goal into verified, documented engineering work.

Responsibilities
----------------
Runs one full, iterative cycle:

    1. Goal            -- the human's (or another module's) request
    2. Interpretation   -- captured as a CognitiveEngine Goal + requirements
    3. Planning          -- CognitiveEngine builds a task plan
    4. Execution         -- CognitiveEngine runs the plan
    5. Verification      -- VerificationEngine critiques the result
    6. Optimization       -- acknowledges/reviews any AutoEngineer
                             optimization already performed within the plan
    7. Documentation      -- a real, structured record of the whole cycle,
                             optionally exported as Markdown

with the ability to step *backward* -- to Planning on a rejected
verification, or to Execution on a "needs revision"/low-confidence one
-- up to a hard cap (`max_revision_cycles`), past which the cycle stops
and asks for a human rather than looping forever.

- Every phase transition and step is recorded into DigitalThread's
  immutable, hash-chained ledger (if bound) -- full traceability of one
  goal's entire journey to its final, documented outcome.
- Claims produced along the way are classified via VerificationEngine's
  FACT/HYPOTHESIS/ESTIMATE/RECOMMENDATION taxonomy -- this module never
  blurs the distinction or invents its own.
- Human approval checkpoints (reusing CognitiveEngine's own
  `ApprovalDecision` type) can gate entry to Execution and/or
  finalization when verification found any issues.
- Publishes phase-change events on EventBus for any other module to
  observe.

Honesty about scope
--------------------
1. **This is an orchestrator, not a new source of engineering
   intelligence.** Every phase delegates to an already-built, already
   honest module: Interpretation/Planning/Execution to CognitiveEngine
   (rule-based goal interpretation -- see its own honesty note),
   Verification to VerificationEngine (rule-based checks -- see its
   own honesty note). This loop adds real, useful *coordination and
   bounded retry* logic on top -- it does not add semantic
   understanding of the goal or the engineering domain that those
   modules don't already (honestly) have.

2. **Stepping backward is a mechanical retry, not a learning process.**
   When verification rejects a result or asks for revision, the loop
   re-runs the *same* rule-based planner/interpreter and executor --
   it does not analyze *why* the previous attempt failed and adjust its
   strategy. This helps with transient failures and retryable actions;
   it will not rescue a request that is genuinely infeasible, and
   `max_revision_cycles` exists specifically so a structurally
   impossible goal fails fast and escalates to a human instead of
   spinning forever.

3. **The "Optimization" phase does not perform new optimization work of
   its own.** Real optimization is `core.auto_engineer.AutoEngineer`'s
   job, invoked as an ordinary plan task if the goal's metadata asked
   for one (see `core.cognitive_engine`'s `optimize_parameters`
   convention). This phase only reviews/records whether that happened
   -- it never re-runs or second-guesses an optimizer.

4. **Documentation is a structured compilation of real facts, not
   generated narrative.** `DocumentationArtifact` summarizes exactly
   what happened (plan size, task success counts, the verification
   verdict, the classified claims) in plain, templated sentences --
   nothing here is synthesized prose from a language model (StarkOS has
   none). The optional Markdown export reuses DigitalThread's own
   `export_markdown()`, which renders the real, hash-chained entries
   for the cycle's trace.

Design
------
`AutonomousEngineeringLoop` satisfies the `Module` protocol (name/
initialize/start/stop) and registers with the Kernel like any other
StarkOS module:

    loop = AutonomousEngineeringLoop(services=services)
    loop.bind_cognitive_engine(cognitive_engine)
    loop.bind_verification_engine(verification_engine)
    loop.bind_digital_thread(digital_thread)
    kernel.register_module(loop, name="autonomous_engineering_loop", priority=310)

    result = await loop.run_cycle(
        "Otimizar o suporte do motor e documentar o resultado",
        metadata={"optimize_parameters": {"spec": my_spec, "evaluator": my_evaluator}},
    )
    print(result.outcome, result.documentation.verification_summary)
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from core.cognitive_engine import ApprovalDecision, CognitiveEngine
from core.digital_thread import DigitalThread
from core.event_bus import Event, EventBus
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.service_container import ServiceContainer
from core.verification_engine import _CONFIDENCE_RANK, ConfidenceLevel, VerificationEngine, VerificationReport

logger = get_logger("autonomous_engineering_loop")

# =============================================================================
# Exceptions
# =============================================================================

class AutonomousEngineeringLoopError(Exception):
    """Base exception for AutonomousEngineeringLoop failures."""

class MissingDependencyError(AutonomousEngineeringLoopError):
    """Raised when run_cycle() needs a module that isn't bound."""

class InvalidCycleRequestError(AutonomousEngineeringLoopError):
    """Raised when run_cycle() arguments are malformed."""

# =============================================================================
# Phases
# =============================================================================

class Phase(Enum):
    GOAL = auto()
    INTERPRETATION = auto()
    PLANNING = auto()
    EXECUTION = auto()
    VERIFICATION = auto()
    OPTIMIZATION = auto()
    DOCUMENTATION = auto()
    COMPLETED = auto()
    FAILED = auto()

# =============================================================================
# Phase-level human approval (reuses CognitiveEngine's ApprovalDecision)
# =============================================================================

@runtime_checkable
class PhaseReviewProvider(Protocol):
    """Approves or rejects entry into a critical phase (distinct from
    CognitiveEngine's own per-*task* approvals -- this is a per-*cycle-
    phase* checkpoint)."""

    async def request_approval(self, cycle_id: str, phase: str, context: dict[str, Any]) -> ApprovalDecision: ...

class AutoApprovePhaseReviewer:
    """Default: approves every phase checkpoint immediately."""

    async def request_approval(self, cycle_id: str, phase: str, context: dict[str, Any]) -> ApprovalDecision:
        return ApprovalDecision(approved=True)

# =============================================================================
# Documentation artifact (real, structured -- see honesty note 4)
# =============================================================================

@dataclass(slots=True, frozen=True)
class DocumentationArtifact:
    cycle_id: str
    goal_description: str
    plan_summary: str
    execution_summary: str
    verification_summary: str
    claims_summary: tuple[str, ...]
    markdown_path: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Cycle state (internal working memory -- not itself an audit record;
# DigitalThread entries and VerificationReports are the real audit trail)
# =============================================================================

@dataclass(slots=True)
class CycleState:
    cycle_id: str
    goal_description: str
    phase: str
    goal: Any = None
    plan: Any = None
    execution_result: Any = None
    verification_report: Optional[VerificationReport] = None
    claims: tuple[Any, ...] = ()
    revision_count: int = 0
    trace_id: Optional[str] = None
    documentation: Optional[DocumentationArtifact] = None
    phase_history: list[str] = field(default_factory=list)

@dataclass(slots=True, frozen=True)
class CycleResult:
    cycle_id: str
    goal_description: str
    final_phase: str
    outcome: str  # "completed" | "aborted_needs_human" | "failed"
    verification_report: Optional[VerificationReport]
    documentation: Optional[DocumentationArtifact]
    revision_count: int
    phase_history: tuple[str, ...]
    trace_id: Optional[str]
    error: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class AutonomousEngineeringLoopConfig:
    max_revision_cycles: int = 3  # hard cap on backward transitions -- see honesty note 2
    minimum_acceptable_confidence: str = ConfidenceLevel.MEDIUM.name
    require_approval_before_execution: bool = False
    require_approval_before_finalizing: bool = True  # only actually asked if verification found any issues
    record_to_knowledge_graph: bool = True
    documentation_output_dir: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Autonomous Engineering Loop
# =============================================================================

class AutonomousEngineeringLoop:
    """
    StarkOS's goal-to-documented-engineering coordinator. See the
    module docstring's "Honesty about scope" section before treating
    this as more than bounded orchestration over already-honest modules.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(self, *, services: ServiceContainer, config: Optional[AutonomousEngineeringLoopConfig] = None) -> None:
        self._services = services
        self._config = config or AutonomousEngineeringLoopConfig()

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._digital_thread: Optional[DigitalThread] = None
        self._cognitive_engine: Optional[CognitiveEngine] = None
        self._verification_engine: Optional[VerificationEngine] = None
        self._rag_engine: Any = None
        self._event_bus: Optional[EventBus] = None
        self._review_provider: Optional[PhaseReviewProvider] = None

        logger.info("AutonomousEngineeringLoop constructed.", extra={"max_revision_cycles": self._config.max_revision_cycles})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "autonomous_engineering_loop"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to AutonomousEngineeringLoop.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to AutonomousEngineeringLoop.")

    def bind_digital_thread(self, digital_thread: DigitalThread) -> None:
        self._digital_thread = digital_thread
        logger.debug("DigitalThread bound to AutonomousEngineeringLoop.")

    def bind_cognitive_engine(self, cognitive_engine: CognitiveEngine) -> None:
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to AutonomousEngineeringLoop.")

    def bind_verification_engine(self, verification_engine: VerificationEngine) -> None:
        self._verification_engine = verification_engine
        logger.debug("VerificationEngine bound to AutonomousEngineeringLoop.")

    def bind_rag_engine(self, rag_engine: Any) -> None:
        self._rag_engine = rag_engine
        logger.debug("RAGEngine bound to AutonomousEngineeringLoop.")

    def bind_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        logger.debug("EventBus bound to AutonomousEngineeringLoop.")

    def set_review_provider(self, provider: PhaseReviewProvider) -> None:
        self._review_provider = provider
        logger.debug("Phase review provider set for AutonomousEngineeringLoop.")

    async def initialize(self) -> None:
        logger.info("Initializing AutonomousEngineeringLoop.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._digital_thread is None:
            self._digital_thread = self._services.resolve_optional(DigitalThread)
        if self._cognitive_engine is None:
            self._cognitive_engine = self._services.resolve_optional(CognitiveEngine)
        if self._verification_engine is None:
            self._verification_engine = self._services.resolve_optional(VerificationEngine)
        if self._event_bus is None:
            self._event_bus = self._services.resolve_optional(EventBus)

        logger.info(
            "AutonomousEngineeringLoop initialized.",
            extra={
                "cognitive_engine_bound": self._cognitive_engine is not None,
                "verification_engine_bound": self._verification_engine is not None,
                "digital_thread_bound": self._digital_thread is not None,
            },
        )

    async def start(self) -> None:
        logger.info("AutonomousEngineeringLoop ready.")

    async def stop(self) -> None:
        logger.info("AutonomousEngineeringLoop stopped.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_cycle(
        self,
        description: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> CycleResult:
        if not description or not description.strip():
            raise InvalidCycleRequestError("description cannot be empty.")
        if self._cognitive_engine is None:
            raise MissingDependencyError("No CognitiveEngine bound -- call bind_cognitive_engine() first.")
        if self._verification_engine is None:
            raise MissingDependencyError("No VerificationEngine bound -- call bind_verification_engine() first.")

        cycle_id = str(uuid.uuid4())
        trace_id = self._digital_thread.begin_trace(f"Engineering cycle: {description}", actor=actor) if self._digital_thread else None
        state = CycleState(cycle_id=cycle_id, goal_description=description, phase=Phase.GOAL.name, trace_id=trace_id)
        await self._publish_event("autonomous_loop.cycle_started", cycle_id=cycle_id, description=description)

        try:
            state = await self._phase_interpretation(state, metadata, actor)

            while True:
                state = await self._phase_planning(state, actor)

                if self._config.require_approval_before_execution:
                    if not await self._request_phase_approval(state, "execution", actor):
                        return self._finalize(state, outcome="aborted_needs_human", actor=actor)

                state = await self._phase_execution(state, actor)
                state = await self._phase_verification(state, actor)

                report = state.verification_report
                assert report is not None
                confidence_rank = _CONFIDENCE_RANK.get(report.confidence, 0)
                minimum_rank = _CONFIDENCE_RANK.get(self._config.minimum_acceptable_confidence, 2)

                if report.verdict == "accepted" and confidence_rank >= minimum_rank:
                    break

                if state.revision_count >= self._config.max_revision_cycles:
                    logger.warning(
                        "Max revision cycles exceeded -- escalating to a human instead of looping forever.",
                        extra={"cycle_id": cycle_id, "verdict": report.verdict, "revision_count": state.revision_count},
                    )
                    return self._finalize(state, outcome="aborted_needs_human", actor=actor)

                target_phase = Phase.PLANNING if report.verdict == "rejected" else Phase.EXECUTION
                state = self._record_backward_transition(state, from_phase=Phase.VERIFICATION, to_phase=target_phase, actor=actor)

            state = await self._phase_optimization(state, actor)

            if self._config.require_approval_before_finalizing and state.verification_report.issues:
                if not await self._request_phase_approval(state, "finalization", actor):
                    return self._finalize(state, outcome="aborted_needs_human", actor=actor)

            state = await self._phase_documentation(state, actor)
            return self._finalize(state, outcome="completed", actor=actor)

        except (MissingDependencyError, InvalidCycleRequestError):
            raise
        except Exception as exc:
            logger.exception("Engineering cycle failed unexpectedly.", extra={"cycle_id": cycle_id})
            state.phase = Phase.FAILED.name
            return self._finalize(state, outcome="failed", actor=actor, error=str(exc))

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    async def _phase_interpretation(self, state: CycleState, metadata: Optional[dict[str, Any]], actor: Optional[str]) -> CycleState:
        await self._transition(state, Phase.INTERPRETATION)
        assert self._cognitive_engine is not None
        goal = self._cognitive_engine.submit_goal(state.goal_description, metadata=metadata or {})
        state.goal = goal
        self._record_step(
            state, description="Goal submitted and requirements captured.", method="CognitiveEngine.submit_goal",
            inputs={"metadata_keys": sorted((metadata or {}).keys())}, result={"goal_id": goal.id}, actor=actor,
        )
        return state

    async def _phase_planning(self, state: CycleState, actor: Optional[str]) -> CycleState:
        await self._transition(state, Phase.PLANNING)
        assert self._cognitive_engine is not None
        plan = self._cognitive_engine.plan(state.goal)
        state.plan = plan
        self._record_step(
            state, description=f"Plan created with {len(plan.tasks)} task(s).", method="CognitiveEngine.plan",
            inputs={"goal_id": state.goal.id}, result={"plan_id": plan.id, "task_count": len(plan.tasks)}, actor=actor,
        )
        return state

    async def _phase_execution(self, state: CycleState, actor: Optional[str]) -> CycleState:
        await self._transition(state, Phase.EXECUTION)
        assert self._cognitive_engine is not None
        execution_result = await self._cognitive_engine.execute_plan(state.plan)
        state.execution_result = execution_result
        succeeded = sum(1 for r in execution_result.task_results if r.status.name == "SUCCEEDED")
        self._record_step(
            state, description=f"Plan executed: {succeeded}/{len(execution_result.task_results)} task(s) succeeded.",
            method="CognitiveEngine.execute_plan", inputs={"plan_id": state.plan.id},
            result={"succeeded": execution_result.succeeded}, actor=actor,
        )
        return state

    async def _phase_verification(self, state: CycleState, actor: Optional[str]) -> CycleState:
        await self._transition(state, Phase.VERIFICATION)
        assert self._verification_engine is not None
        report = self._verification_engine.verify_plan_execution(state.goal, state.execution_result, actor=actor)
        state.verification_report = report
        state.claims = report.claims
        self._record_validation_step(
            state, description=f"Verification verdict: {report.verdict}",
            validation={"verdict": report.verdict, "confidence": report.confidence},
            result={"issue_count": len(report.issues)}, actor=actor,
        )
        return state

    async def _phase_optimization(self, state: CycleState, actor: Optional[str]) -> CycleState:
        await self._transition(state, Phase.OPTIMIZATION)
        optimize_task_count = sum(1 for task in state.plan.tasks if task.action == "optimize") if state.plan else 0
        self._record_step(
            state,
            description="Optimization phase reviewed -- see plan tasks for any AutoEngineer optimization actually performed.",
            method="AutonomousEngineeringLoop.review_optimization", inputs={},
            result={"optimize_task_count": optimize_task_count}, actor=actor,
        )
        return state

    async def _phase_documentation(self, state: CycleState, actor: Optional[str]) -> CycleState:
        await self._transition(state, Phase.DOCUMENTATION)
        artifact = self._build_documentation(state)

        markdown_path = None
        if self._digital_thread is not None and state.trace_id is not None and self._config.documentation_output_dir is not None:
            path = self._config.documentation_output_dir / f"{state.cycle_id}.md"
            try:
                self._digital_thread.export_markdown(state.trace_id, path)
                markdown_path = str(path)
            except Exception:
                logger.exception("Failed to export cycle documentation to Markdown.")

        artifact = dataclasses.replace(artifact, markdown_path=markdown_path)
        state.documentation = artifact
        self._record_step(
            state, description="Documentation compiled.", method="AutonomousEngineeringLoop.document",
            inputs={}, result={"markdown_path": markdown_path}, actor=actor,
        )
        return state

    def _build_documentation(self, state: CycleState) -> DocumentationArtifact:
        plan_summary = f"{len(state.plan.tasks)} task(s) planned." if state.plan else "No plan was created."

        if state.execution_result:
            succeeded = sum(1 for r in state.execution_result.task_results if r.status.name == "SUCCEEDED")
            execution_summary = f"{succeeded}/{len(state.execution_result.task_results)} task(s) succeeded."
        else:
            execution_summary = "No execution occurred."

        if state.verification_report:
            verification_summary = (
                f"Verdict: {state.verification_report.verdict}, confidence: {state.verification_report.confidence}, "
                f"{len(state.verification_report.issues)} issue(s) found."
            )
        else:
            verification_summary = "No verification was performed."

        claims_summary = tuple(f"[{claim.claim_type}] {claim.text}" for claim in state.claims)

        return DocumentationArtifact(
            cycle_id=state.cycle_id, goal_description=state.goal_description, plan_summary=plan_summary,
            execution_summary=execution_summary, verification_summary=verification_summary, claims_summary=claims_summary,
        )

    # ------------------------------------------------------------------
    # Backward transitions (bounded -- see honesty note 2)
    # ------------------------------------------------------------------

    def _record_backward_transition(self, state: CycleState, *, from_phase: Phase, to_phase: Phase, actor: Optional[str]) -> CycleState:
        state.revision_count += 1
        logger.warning(
            "Engineering cycle stepping back.",
            extra={
                "cycle_id": state.cycle_id, "from_phase": from_phase.name, "to_phase": to_phase.name,
                "revision_count": state.revision_count,
            },
        )
        verdict = state.verification_report.verdict if state.verification_report else "unknown"
        self._record_step(
            state, description=f"Stepping back from {from_phase.name} to {to_phase.name} (revision {state.revision_count}, verdict: {verdict}).",
            method="AutonomousEngineeringLoop.revise", inputs={"verdict": verdict},
            result={"revision_count": state.revision_count}, actor=actor,
        )
        return state

    # ------------------------------------------------------------------
    # Human approval checkpoints
    # ------------------------------------------------------------------

    async def _request_phase_approval(self, state: CycleState, phase_label: str, actor: Optional[str]) -> bool:
        reviewer = self._review_provider or AutoApprovePhaseReviewer()
        context = {
            "goal_description": state.goal_description,
            "plan_task_count": len(state.plan.tasks) if state.plan else 0,
            "verification_verdict": state.verification_report.verdict if state.verification_report else None,
        }
        try:
            decision = await reviewer.request_approval(state.cycle_id, phase_label, context)
        except Exception:
            logger.exception("Phase review provider failed -- treating as not approved.")
            decision = ApprovalDecision(approved=False, notes="Review provider raised an exception.")

        self._record_step(
            state, description=f"Human approval requested for '{phase_label}': {'approved' if decision.approved else 'declined'}.",
            method="PhaseReviewProvider.request_approval", inputs=context,
            result={"approved": decision.approved, "notes": decision.notes}, actor=actor,
        )
        return decision.approved

    # ------------------------------------------------------------------
    # Phase transitions / EventBus
    # ------------------------------------------------------------------

    async def _transition(self, state: CycleState, phase: Phase) -> None:
        state.phase = phase.name
        state.phase_history.append(phase.name)
        logger.info("Engineering cycle phase transition.", extra={"cycle_id": state.cycle_id, "phase": phase.name})
        await self._publish_event("autonomous_loop.phase_changed", cycle_id=state.cycle_id, phase=phase.name)

    async def _publish_event(self, topic: str, **payload: Any) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.publish(Event(topic=topic, source="autonomous_engineering_loop", payload=payload))
        except Exception:
            logger.exception("Failed to publish event '%s'.", topic)

    # ------------------------------------------------------------------
    # DigitalThread / KnowledgeGraph recording
    # ------------------------------------------------------------------

    def _record_step(self, state: CycleState, *, description: str, method: str, inputs: dict[str, Any], result: Any, actor: Optional[str]) -> None:
        if self._digital_thread is None or state.trace_id is None:
            return
        try:
            self._digital_thread.record_action(
                trace_id=state.trace_id, description=description, inputs=inputs, method=method,
                parameters={"phase": state.phase}, result=result, actor=actor,
            )
        except Exception:
            logger.exception("Failed to record cycle step in DigitalThread.")

    def _record_validation_step(self, state: CycleState, *, description: str, validation: dict[str, Any], result: Any, actor: Optional[str]) -> None:
        if self._digital_thread is None or state.trace_id is None:
            return
        try:
            self._digital_thread.record_validation(
                trace_id=state.trace_id, description=description, validation=validation, result=result, actor=actor,
            )
        except Exception:
            logger.exception("Failed to record verification step in DigitalThread.")

    def _finalize(self, state: CycleState, *, outcome: str, actor: Optional[str], error: Optional[str] = None) -> CycleResult:
        if self._digital_thread is not None and state.trace_id is not None:
            try:
                self._digital_thread.record_result(
                    trace_id=state.trace_id, description=f"Cycle finished: {outcome}",
                    result={"outcome": outcome, "final_phase": state.phase, "error": error}, actor=actor,
                )
            except Exception:
                logger.exception("Failed to record final cycle result in DigitalThread.")

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            try:
                self._knowledge_graph.remember(
                    f"Engineering cycle '{state.goal_description}' finished: {outcome}",
                    node_type="engineering_cycle",
                    metadata={"cycle_id": state.cycle_id, "outcome": outcome, "revision_count": state.revision_count},
                    source="autonomous_engineering_loop",
                )
            except Exception:
                logger.exception("Failed to record cycle outcome in KnowledgeGraph.")

        logger.info(
            "Engineering cycle finished.",
            extra={"cycle_id": state.cycle_id, "outcome": outcome, "revision_count": state.revision_count, "final_phase": state.phase},
        )

        return CycleResult(
            cycle_id=state.cycle_id, goal_description=state.goal_description, final_phase=state.phase, outcome=outcome,
            verification_report=state.verification_report, documentation=state.documentation,
            revision_count=state.revision_count, phase_history=tuple(state.phase_history), trace_id=state.trace_id, error=error,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "cognitive_engine_bound": self._cognitive_engine is not None,
            "verification_engine_bound": self._verification_engine is not None,
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "digital_thread_bound": self._digital_thread is not None,
            "rag_engine_bound": self._rag_engine is not None,
            "event_bus_bound": self._event_bus is not None,
            "max_revision_cycles": self._config.max_revision_cycles,
        }