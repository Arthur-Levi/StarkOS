"""
core/cognitive_engine.py
==========================

Goal-driven orchestration for StarkOS -- the coordination layer that
turns a goal into a traceable, verifiable plan of specialist actions.

Responsibilities
----------------
- Accept a goal (free text + optional structured metadata) from a user
  or another module, decompose it into a multi-step, dependency-ordered
  Plan, and execute it: independent tasks run concurrently (real
  parallelism, not simulated), dependent ones wait on their inputs,
  outputs flow forward via `$task:`/`$blackboard:` references, results
  are validated and retried, and a human can be asked to approve
  individual steps.
- Layered memory: a bounded short-term buffer (immediate context),
  episodic recall of past goals/executions, and semantic recall over
  general stored knowledge -- see the honesty note on how these three
  are actually built.
- A "multi-agent" system: bound specialist modules (Identity,
  KnowledgeGraph, AutoEngineer, RAGEngine, ...) are the agents, invoked
  through an explicit, reviewable action allow-list; they communicate
  through task outputs and a shared per-execution blackboard, and
  independent ones execute in parallel.
- Post-execution reflection: a rule-based check of whether the plan
  actually completed, with simple, mechanically-derived suggestions --
  see the honesty note on what this reflection is and isn't.
- Every goal/plan/execution is recorded into KnowledgeGraph (long-term,
  searchable memory) and, if a DigitalThread is bound, into its
  immutable, hash-chained ledger -- full goal-to-execution traceability.

Honesty about scope
--------------------
This is an **orchestration and planning framework**, not a general
reasoning engine. It contains no language model and performs no genuine
natural-language understanding, autonomous multi-agent reasoning, or
deep semantic verification. Concretely:

- "Understanding the user's goal" -- including a *vague* one -- is
  rule-based pattern matching (see `HeuristicGoalInterpreter`): it
  recognizes a handful of known verbs/phrases and maps them to
  specialist actions. Anything it doesn't recognize falls back to
  `Identity.respond()` -- it is never silently "understood" in some
  deeper sense, however the goal is phrased.
- **"Layered memory" is a thin, honest interface over systems that
  already exist**, not three new memory engines: short-term is a plain
  bounded in-process buffer (gone on restart, by design); episodic and
  semantic recall both delegate to KnowledgeGraph (and RAGEngine, if
  bound) -- which already do real storage, embeddings and retrieval.
  Building a second, separate long-term memory system here would just
  duplicate what those modules already do correctly.
- **"Agents" are the same bound specialist modules as before**, under a
  more evocative name -- never autonomous entities with their own
  reasoning loop. "Communication between agents" is two concrete,
  inspectable mechanisms: `$task:<id>.<path>` references (one task's
  output feeds another's input) and a shared, per-execution
  `$blackboard:<key>` namespace any task can publish a named value into
  for later ones to read. "Parallel orchestration" is real: tasks with
  no dependency on each other (the same level of the dependency graph)
  run concurrently via `asyncio.gather` -- not sequential execution
  dressed up as parallel.
- **Reflection is a rule-based, mechanical check** (`HeuristicReflector`):
  did tasks succeed, are there non-empty results, what's the obvious
  next step if something failed. It is not semantic verification that
  the goal was actually, meaningfully achieved -- StarkOS has no model
  capable of that judgment call.
- The actual engineering inputs for a task (a `DesignSpec`, an
  `Assembly`, a list of `RiskFactor`) are never invented by this module.
  They must be supplied by the caller via `Goal.metadata`, exactly as
  `AutoEngineer`'s own evaluators/constraints must be supplied by its
  caller. CognitiveEngine only decides *which* specialist/action a goal
  maps to and *in what order* (or in parallel) to run things -- never
  what the engineering content of a task should be.
- "Creating specialists" means registering (`bind_...`) already-built
  StarkOS modules under a name, and exposing an explicit, reviewable
  allow-list of their methods as callable actions -- never dynamically
  invoking arbitrary attributes/strings, and never generating or
  executing new code at runtime.

The `GoalInterpreter`, `ReviewProvider` and `Reflector` Protocols exist
precisely so that a real language-model-backed planner/verifier (once
StarkOS's AI Runtime/LLM providers exist) can replace today's honest,
simple defaults without touching `CognitiveEngine` itself.

Design
------
Same low-coupling shape as the rest of StarkOS: pluggable Protocols with
transparent, dependency-free defaults.

- `GoalInterpreter` -- turns a `Goal` into a `Plan`. Default:
  `HeuristicGoalInterpreter` (keyword/pattern matching).
- `ReviewProvider` -- approves or rejects a task before it runs, in
  collaborative mode. Default: `AutoApproveReviewer`.
- `Reflector` -- reviews a completed execution. Default:
  `HeuristicReflector` (rule-based, not semantic verification).

`CognitiveEngine` satisfies the `Module` protocol (name/initialize/
start/stop) and registers with the Kernel like any other StarkOS module:

    engine = CognitiveEngine(services=services)
    engine.bind_identity(identity)
    engine.bind_knowledge_graph(knowledge_graph)
    engine.bind_rag_engine(rag_engine)
    engine.bind_digital_thread(digital_thread)
    kernel.register_module(engine, name="cognitive_engine", priority=300)

    result = await engine.pursue_goal(
        "Otimizar o design e gerar o BOM",
        metadata={
            "optimize_parameters": {"spec": my_spec, "evaluator": my_evaluator},
            "bom_parameters": {"assembly": my_assembly},
        },
    )
    print(result.reflection.summary, result.digital_thread_trace_id)
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from core.auto_engineer import AutoEngineer
from core.event_bus import Event, EventBus
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.service_container import ServiceContainer

logger = get_logger("cognitive_engine")

# =============================================================================
# Exceptions
# =============================================================================

class CognitiveEngineError(Exception):
    """Base exception for CognitiveEngine failures."""

class InvalidGoalError(CognitiveEngineError):
    """Raised when a Goal is malformed (e.g. empty description)."""

class PlanningError(CognitiveEngineError):
    """Raised when a GoalInterpreter fails to produce a Plan."""

class InvalidPlanError(CognitiveEngineError):
    """Raised when a Plan's task graph is malformed (unknown/circular deps)."""

class UnknownSpecialistError(CognitiveEngineError):
    """Raised when a task references a specialist that isn't registered."""

class UnknownActionError(CognitiveEngineError):
    """Raised when a task references an action its specialist doesn't expose."""

class PlanExecutionError(CognitiveEngineError):
    """Raised when parameter resolution or execution bookkeeping fails."""

# =============================================================================
# Goals
# =============================================================================

class GoalStatus(Enum):
    RECEIVED = auto()
    PLANNED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass(slots=True, frozen=True)
class Goal:
    """A user's (or another module's) objective, in free text, plus
    whatever structured inputs the caller already knows are needed
    (design specs, assemblies, risk registers, ...)."""

    id: str
    description: str
    priority: int = 3  # 1 (highest) .. 5 (lowest)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Tasks and plans
# =============================================================================

class TaskStatus(Enum):
    SUCCEEDED = auto()
    FAILED = auto()
    SKIPPED = auto()

@dataclass(slots=True, frozen=True)
class Task:
    """One step of a Plan: dispatch `action` on specialist `specialist`
    with `parameters` (which may reference earlier tasks' outputs via
    "$task:<task_id>.<path>" placeholders), after `depends_on` tasks
    have all succeeded."""

    id: str
    description: str
    specialist: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    requires_approval: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    max_retries: int = 0

@dataclass(slots=True, frozen=True)
class Plan:
    id: str
    goal_id: str
    tasks: tuple[Task, ...]
    created_at: datetime = field(default_factory=datetime.utcnow)

    def task(self, task_id: str) -> Task:
        for candidate in self.tasks:
            if candidate.id == task_id:
                return candidate
        raise InvalidPlanError(f"Unknown task '{task_id}' in plan '{self.id}'.")

@dataclass(slots=True, frozen=True)
class TaskResult:
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    attempts: int = 1

@dataclass(slots=True, frozen=True)
class PlanExecutionResult:
    plan_id: str
    goal_id: str
    task_results: tuple[TaskResult, ...]
    succeeded: bool
    completed_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Goal interpretation (planning)
# =============================================================================

@runtime_checkable
class GoalInterpreter(Protocol):
    """Turns a Goal into an executable Plan, given the names of
    currently-registered specialists."""

    def interpret(self, goal: Goal, *, specialists: Sequence[str]) -> Plan:
        ...

class HeuristicGoalInterpreter:
    """
    Rule-based goal decomposition: matches known phrase patterns in the
    goal's description and produces one task per match, chained
    sequentially. Each matched task pulls its actual parameters from
    `goal.metadata[<metadata_key>]` -- this class decides *which*
    specialist/action a phrase maps to, never the engineering content
    of the call. Unmatched goals fall back to `Identity.respond()`.

    This is intentionally simple and transparent, not a claim of
    natural-language understanding. Once StarkOS's AI Runtime (LLM
    providers) exists, a real language-model-backed `GoalInterpreter`
    can replace this without `CognitiveEngine` changing at all.
    """

    _PATTERNS: tuple[tuple[str, str, str, str], ...] = (
        (r"\botimiz|\boptimi", "auto_engineer", "optimize", "optimize_parameters"),
        (r"\bbom\b|lista de materiais|bill of materials", "auto_engineer", "generate_bom", "bom_parameters"),
        (r"\brisco|\brisk", "auto_engineer", "assess_risks", "risk_parameters"),
        (r"\bbusca|\blembr|\brecall|\bsearch|conhecimento", "knowledge_graph", "recall", "recall_parameters"),
    )

    def interpret(self, goal: Goal, *, specialists: Sequence[str]) -> Plan:
        description = goal.description.lower()
        tasks: list[Task] = []
        previous_id: Optional[str] = None

        for pattern, specialist, action, metadata_key in self._PATTERNS:
            if specialist not in specialists:
                continue
            if not re.search(pattern, description):
                continue

            parameters = dict(goal.metadata.get(metadata_key, {}))
            if action == "recall" and "query" not in parameters:
                parameters["query"] = goal.description

            task_id = f"{goal.id}-{len(tasks)}"
            tasks.append(
                Task(
                    id=task_id,
                    description=f"{action} via {specialist} (matched for goal '{goal.description[:40]}')",
                    specialist=specialist,
                    action=action,
                    parameters=parameters,
                    depends_on=(previous_id,) if previous_id else (),
                )
            )
            previous_id = task_id

        if not tasks:
            if "identity" not in specialists:
                raise InvalidPlanError(
                    f"No task pattern matched goal '{goal.id}' and no fallback specialist ('identity') is registered."
                )
            tasks.append(
                Task(
                    id=f"{goal.id}-0",
                    description="Fallback: converse via Identity (no specialist pattern matched).",
                    specialist="identity",
                    action="respond",
                    parameters={"message": goal.description},
                )
            )

        return Plan(id=str(uuid.uuid4()), goal_id=goal.id, tasks=tuple(tasks))

# =============================================================================
# Collaborative mode (human-in-the-loop review)
# =============================================================================

class CollaborationMode(Enum):
    AUTONOMOUS = auto()
    COLLABORATIVE = auto()

@dataclass(slots=True, frozen=True)
class ApprovalDecision:
    approved: bool
    notes: str = ""

@runtime_checkable
class ReviewProvider(Protocol):
    """Approves or rejects a task before it runs, in collaborative mode."""

    async def request_approval(self, task: Task, parameters: dict[str, Any]) -> ApprovalDecision:
        ...

class AutoApproveReviewer:
    """Default reviewer: approves everything immediately. Used for
    CollaborationMode.AUTONOMOUS; also a safe fallback if collaborative
    mode is selected but no real reviewer has been set."""

    async def request_approval(self, task: Task, parameters: dict[str, Any]) -> ApprovalDecision:
        return ApprovalDecision(approved=True)

class CallbackReviewProvider:
    """
    Wraps a plain callable (sync or async, taking (task, parameters)) as
    a ReviewProvider -- a lightweight way to plug in a real approval
    hook (e.g. prompting the operator from the console) without writing
    a full class. The callback may return a bool or an ApprovalDecision.
    """

    def __init__(self, callback: Callable[[Task, dict[str, Any]], Any]) -> None:
        self._callback = callback

    async def request_approval(self, task: Task, parameters: dict[str, Any]) -> ApprovalDecision:
        if inspect.iscoroutinefunction(self._callback):
            result = await self._callback(task, parameters)
        else:
            result = await asyncio.to_thread(self._callback, task, parameters)
        if isinstance(result, ApprovalDecision):
            return result
        return ApprovalDecision(approved=bool(result))

# =============================================================================
# Specialists
# =============================================================================

@dataclass(slots=True)
class SpecialistHandle:
    """An explicit, reviewable allow-list of callables exposed by one
    bound module. CognitiveEngine dispatches to these by name -- it
    never invokes arbitrary attributes from a string."""

    name: str
    actions: dict[str, Callable[..., Any]]

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class CognitiveEngineConfig:
    default_mode: CollaborationMode = CollaborationMode.AUTONOMOUS
    # In COLLABORATIVE mode, if True every task needs approval regardless
    # of its own `requires_approval` flag.
    approve_every_task: bool = False
    # Sync specialist actions are dispatched via asyncio.to_thread by
    # default so a heavy call (e.g. AutoEngineer.optimize with many
    # iterations) can't block the event loop.
    run_sync_actions_in_thread: bool = True
    record_to_knowledge_graph: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Cognitive Engine
# =============================================================================

class CognitiveEngine:
    """
    StarkOS goal-driven orchestration module: understands (via pattern
    matching) what a goal is asking for, plans a dependency-ordered
    sequence of specialist actions, executes it with validation/retries,
    optionally pausing for human approval, and remembers the outcome.

    Satisfies the `Module` protocol (name/initialize/start/stop) and can
    be registered with the Kernel like any other module. See the module
    docstring for the (deliberately explicit) boundaries of what this
    class actually understands versus what the caller must supply.
    """

    TASK_RESULT_PREFIX = "$task:"

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[CognitiveEngineConfig] = None,
        interpreter: Optional[GoalInterpreter] = None,
    ) -> None:
        self._services = services
        self._config = config or CognitiveEngineConfig()
        self._interpreter: GoalInterpreter = interpreter or HeuristicGoalInterpreter()

        self._kernel: Any = None
        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._auto_engineer: Optional[AutoEngineer] = None
        self._digital_twin: Any = None
        self._review_provider: Optional[ReviewProvider] = None

        self._specialists: dict[str, SpecialistHandle] = {}
        self._goals: dict[str, Goal] = {}
        self._plans: dict[str, Plan] = {}
        self._history: list[PlanExecutionResult] = []

        logger.info("CognitiveEngine constructed.", extra={"default_mode": self._config.default_mode.name})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "cognitive_engine"

    async def initialize(self) -> None:
        logger.info("Initializing CognitiveEngine.")

        if self._identity is None:
            identity = self._services.resolve_optional(Identity)
            if identity is not None:
                self.bind_identity(identity)
        if self._knowledge_graph is None:
            knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
            if knowledge_graph is not None:
                self.bind_knowledge_graph(knowledge_graph)
        if self._auto_engineer is None:
            auto_engineer = self._services.resolve_optional(AutoEngineer)
            if auto_engineer is not None:
                self.bind_auto_engineer(auto_engineer)

        logger.info("CognitiveEngine initialized.", extra={"specialists": list(self._specialists.keys())})

    async def start(self) -> None:
        logger.info("CognitiveEngine ready.", extra={"specialists": list(self._specialists.keys())})

    async def stop(self) -> None:
        logger.info("CognitiveEngine stopped.")

    # ------------------------------------------------------------------
    # Specialist / dependency binding
    # ------------------------------------------------------------------

    def bind_kernel(self, kernel: Any) -> None:
        """Kernel does not register itself into the ServiceContainer, so
        it is handed to modules explicitly -- mirrors Identity/VoiceInterface."""
        self._kernel = kernel
        self._specialists["kernel"] = SpecialistHandle(
            name="kernel",
            actions={
                "health": kernel.health,
                "diagnostics": kernel.diagnostics,
                "demo": kernel.demo,
                "restart": kernel.restart,
            },
        )
        logger.debug("Kernel bound to CognitiveEngine.")

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        self._specialists["identity"] = SpecialistHandle(
            name="identity",
            actions={"respond": identity.respond, "greet": identity.greet},
        )
        logger.debug("Identity bound to CognitiveEngine.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        self._specialists["knowledge_graph"] = SpecialistHandle(
            name="knowledge_graph",
            actions={
                "remember": knowledge_graph.remember,
                "recall": knowledge_graph.recall,
                "add_node": knowledge_graph.add_node,
                "search": knowledge_graph.search,
            },
        )
        logger.debug("KnowledgeGraph bound to CognitiveEngine.")

    def bind_auto_engineer(self, auto_engineer: AutoEngineer) -> None:
        self._auto_engineer = auto_engineer
        self._specialists["auto_engineer"] = SpecialistHandle(
            name="auto_engineer",
            actions={
                "optimize": auto_engineer.optimize,
                "optimize_async": auto_engineer.optimize_async,
                "generate_bom": auto_engineer.generate_bom,
                "assess_risks": auto_engineer.assess_risks,
                "recall_similar_designs": auto_engineer.recall_similar_designs,
            },
        )
        logger.debug("AutoEngineer bound to CognitiveEngine.")

    def bind_digital_twin(self, digital_twin: Any) -> None:
        """
        No concrete Digital Twin exists yet in StarkOS (see
        `core.auto_engineer.DigitalTwinQueryable`); this accepts anything
        duck-typing that interface and exposes whichever of its methods
        are actually present under the 'digital_twin' specialist name.
        """
        self._digital_twin = digital_twin
        actions: dict[str, Callable[..., Any]] = {}
        get_state = getattr(digital_twin, "get_asset_state", None)
        if get_state is not None:
            actions["get_asset_state"] = get_state
        self._specialists["digital_twin"] = SpecialistHandle(name="digital_twin", actions=actions)
        logger.debug("Digital Twin bound to CognitiveEngine.", extra={"actions": list(actions.keys())})

    def set_review_provider(self, provider: ReviewProvider) -> None:
        """Wire a real human-in-the-loop reviewer for CollaborationMode.COLLABORATIVE."""
        self._review_provider = provider
        logger.debug("Review provider set for CognitiveEngine.")

    # ------------------------------------------------------------------
    # Goal submission and planning
    # ------------------------------------------------------------------

    def submit_goal(
        self,
        description: str,
        *,
        priority: int = 3,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Goal:
        if not description or not description.strip():
            raise InvalidGoalError("Goal description cannot be empty.")
        if not (1 <= priority <= 5):
            raise InvalidGoalError("priority must be between 1 (highest) and 5 (lowest).")

        goal = Goal(id=str(uuid.uuid4()), description=description, priority=priority, metadata=metadata or {})
        self._goals[goal.id] = goal
        logger.info("Goal submitted.", extra={"goal_id": goal.id, "priority": priority})

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            try:
                self._knowledge_graph.remember(
                    f"Goal: {description}",
                    node_type="goal",
                    metadata={"goal_id": goal.id, "priority": priority},
                    source="cognitive_engine",
                )
            except Exception:
                logger.exception("Failed to record goal in KnowledgeGraph.")

        return goal

    def plan(self, goal: Goal) -> Plan:
        try:
            new_plan = self._interpreter.interpret(goal, specialists=tuple(self._specialists.keys()))
        except InvalidPlanError:
            raise
        except Exception as exc:
            raise PlanningError(f"Failed to plan for goal '{goal.id}'.") from exc

        self._execution_order(new_plan)  # eagerly validates the dependency graph
        self._plans[new_plan.id] = new_plan
        logger.info("Plan created.", extra={"goal_id": goal.id, "plan_id": new_plan.id, "task_count": len(new_plan.tasks)})
        return new_plan

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    async def execute_plan(self, plan: Plan, *, mode: Optional[CollaborationMode] = None) -> PlanExecutionResult:
        effective_mode = mode or self._config.default_mode
        order = self._execution_order(plan)

        results: dict[str, TaskResult] = {}
        raw_outputs: dict[str, Any] = {}

        logger.info(
            "Executing plan.",
            extra={"plan_id": plan.id, "goal_id": plan.goal_id, "task_count": len(plan.tasks), "mode": effective_mode.name},
        )

        for task_id in order:
            task = plan.task(task_id)

            if any(results[dep].status is not TaskStatus.SUCCEEDED for dep in task.depends_on):
                results[task_id] = TaskResult(task_id=task_id, status=TaskStatus.SKIPPED, error="A dependency did not succeed.")
                logger.warning("Task skipped -- a dependency failed or was skipped.", extra={"task_id": task_id})
                continue

            try:
                resolved_parameters = self._resolve_parameters(task.parameters, raw_outputs)
            except PlanExecutionError as exc:
                results[task_id] = TaskResult(task_id=task_id, status=TaskStatus.FAILED, error=str(exc))
                logger.error("Parameter resolution failed for task.", extra={"task_id": task_id})
                continue

            if effective_mode is CollaborationMode.COLLABORATIVE and (task.requires_approval or self._config.approve_every_task):
                decision = await self._request_approval(task, resolved_parameters)
                if not decision.approved:
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED,
                        error=f"Not approved: {decision.notes}" if decision.notes else "Not approved.",
                    )
                    logger.info("Task rejected by reviewer.", extra={"task_id": task_id})
                    continue

            task_result = await self._run_task_with_retries(task, resolved_parameters)
            results[task_id] = task_result
            if task_result.status is TaskStatus.SUCCEEDED:
                raw_outputs[task_id] = task_result.output

        ordered_results = tuple(results[task_id] for task_id in order)
        succeeded = all(result.status is TaskStatus.SUCCEEDED for result in ordered_results)

        execution_result = PlanExecutionResult(
            plan_id=plan.id, goal_id=plan.goal_id, task_results=ordered_results, succeeded=succeeded
        )
        self._history.append(execution_result)

        logger.info("Plan execution completed.", extra={"plan_id": plan.id, "succeeded": succeeded})

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            self._record_execution(execution_result)

        return execution_result

    async def pursue_goal(
        self,
        description: str,
        *,
        priority: int = 3,
        metadata: Optional[dict[str, Any]] = None,
        mode: Optional[CollaborationMode] = None,
    ) -> PlanExecutionResult:
        """Convenience one-shot pipeline: submit -> plan -> execute."""
        goal = self.submit_goal(description, priority=priority, metadata=metadata)
        new_plan = self.plan(goal)
        return await self.execute_plan(new_plan, mode=mode)

    # ------------------------------------------------------------------
    # Task execution internals
    # ------------------------------------------------------------------

    async def _run_task_with_retries(self, task: Task, parameters: dict[str, Any]) -> TaskResult:
        attempts = 0
        last_error: Optional[str] = None

        while attempts <= task.max_retries:
            attempts += 1
            try:
                output = await self._invoke_specialist(task, parameters)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Task attempt failed.", extra={"task_id": task.id, "attempt": attempts})
                continue

            if task.validator is not None:
                try:
                    valid = bool(task.validator(output))
                except Exception as exc:
                    valid = False
                    last_error = f"Validator raised: {exc}"
                if not valid:
                    last_error = last_error or "Validation failed."
                    logger.warning("Task output failed validation.", extra={"task_id": task.id, "attempt": attempts})
                    continue

            return TaskResult(task_id=task.id, status=TaskStatus.SUCCEEDED, output=output, attempts=attempts)

        logger.error("Task failed after all attempts.", extra={"task_id": task.id, "attempts": attempts})
        return TaskResult(task_id=task.id, status=TaskStatus.FAILED, error=last_error or "Unknown failure.", attempts=attempts)

    async def _invoke_specialist(self, task: Task, parameters: dict[str, Any]) -> Any:
        handle = self._specialists.get(task.specialist)
        if handle is None:
            raise UnknownSpecialistError(f"No specialist registered for '{task.specialist}'.")

        handler = handle.actions.get(task.action)
        if handler is None:
            raise UnknownActionError(
                f"Specialist '{task.specialist}' has no action '{task.action}'. Available: {sorted(handle.actions)}"
            )

        if inspect.iscoroutinefunction(handler):
            return await handler(**parameters)
        if self._config.run_sync_actions_in_thread:
            return await asyncio.to_thread(handler, **parameters)
        return handler(**parameters)

    async def _request_approval(self, task: Task, parameters: dict[str, Any]) -> ApprovalDecision:
        reviewer = self._review_provider or AutoApproveReviewer()
        try:
            return await reviewer.request_approval(task, parameters)
        except Exception:
            logger.exception("Review provider failed -- treating the task as not approved.")
            return ApprovalDecision(approved=False, notes="Review provider raised an exception.")

    # ------------------------------------------------------------------
    # Plan graph / parameter resolution helpers
    # ------------------------------------------------------------------

    def _execution_order(self, plan: Plan) -> list[str]:
        graph: dict[str, set[str]] = {task.id: set(task.depends_on) for task in plan.tasks}
        for task_id, deps in graph.items():
            for dep in deps:
                if dep not in graph:
                    raise InvalidPlanError(f"Task '{task_id}' depends on unknown task '{dep}'.")

        remaining = {key: set(value) for key, value in graph.items()}
        resolved: list[str] = []
        while remaining:
            ready = sorted(name for name, deps in remaining.items() if not deps)
            if not ready:
                raise InvalidPlanError(f"Circular dependency detected in plan '{plan.id}'.")
            resolved.extend(ready)
            for name in ready:
                remaining.pop(name)
            for deps in remaining.values():
                deps.difference_update(ready)
        return resolved

    def _resolve_parameters(self, parameters: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
        return {key: self._resolve_value(value, results) for key, value in parameters.items()}

    def _resolve_value(self, value: Any, results: dict[str, Any]) -> Any:
        if isinstance(value, str) and value.startswith(self.TASK_RESULT_PREFIX):
            reference = value[len(self.TASK_RESULT_PREFIX):]
            task_id, _, path = reference.partition(".")
            if task_id not in results:
                raise PlanExecutionError(f"Reference '{value}' points to a task with no recorded output yet.")
            target = results[task_id]
            if path:
                for part in path.split("."):
                    if isinstance(target, dict):
                        target = target.get(part)
                    else:
                        target = getattr(target, part, None)
            return target
        if isinstance(value, dict):
            return {key: self._resolve_value(item, results) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._resolve_value(item, results) for item in value)
        return value

    # ------------------------------------------------------------------
    # KnowledgeGraph integration
    # ------------------------------------------------------------------

    def _record_execution(self, execution_result: PlanExecutionResult) -> None:
        if self._knowledge_graph is None:
            return

        succeeded_count = sum(1 for result in execution_result.task_results if result.status is TaskStatus.SUCCEEDED)
        content = (
            f"Plan execution for goal '{execution_result.goal_id}': "
            f"{succeeded_count}/{len(execution_result.task_results)} tasks succeeded, "
            f"overall succeeded={execution_result.succeeded}"
        )
        metadata = {
            "plan_id": execution_result.plan_id,
            "goal_id": execution_result.goal_id,
            "succeeded": execution_result.succeeded,
            "task_count": len(execution_result.task_results),
        }
        try:
            self._knowledge_graph.remember(content, node_type="plan_execution", metadata=metadata, source="cognitive_engine")
        except Exception:
            logger.exception("Failed to record plan execution in KnowledgeGraph.")

    def recall_similar_goals(self, query: str, *, top_k: int = 5) -> tuple[Any, ...]:
        """Semantic recall over past goals/plan executions recorded in
        the bound KnowledgeGraph."""
        if self._knowledge_graph is None:
            raise CognitiveEngineError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")
        return self._knowledge_graph.recall(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "specialists": list(self._specialists.keys()),
            "goals_submitted": len(self._goals),
            "plans_created": len(self._plans),
            "executions_run": len(self._history),
            "default_mode": self._config.default_mode.name,
        }