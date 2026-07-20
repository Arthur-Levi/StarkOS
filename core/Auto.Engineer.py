"""
core/auto_engineer.py
======================

Engineering design orchestration for StarkOS.

Responsibilities
----------------
- Iterative design optimization over a caller-defined parameter space,
  against caller-defined objectives and constraints.
- Bill of Materials (BOM) generation from a hierarchical assembly tree.
- Generic project risk assessment and constraint-based feasibility
  checking.
- Bridge to KnowledgeGraph: record every optimization run, BOM and risk
  assessment as searchable long-term memory, and recall similar past
  designs.
- Bridge to a (future) Digital Twin: seed a design candidate from an
  asset's real, currently-deployed state.

Honesty about scope
--------------------
This module is an **orchestrator**, not a physics engine. It contains no
embedded structural, thermal, electrical or manufacturing domain
knowledge, and it never fabricates an engineering safety verdict. Every
piece of actual engineering judgment is supplied by the caller:

- Optimization objectives are scored by metric functions *you* provide
  (`WeightedSumEvaluator(metric_functions=...)`, or your own
  `DesignEvaluator` -- e.g. wrapping a real FEA/thermal/cost model).
- Feasibility is plain constraint checking (`>=`, `<=`, `==`, `between`)
  against bounds *you* declare in a `DesignSpec` -- not a certified
  engineering sign-off.
- Risk scoring uses a generic likelihood x impact matrix (in the spirit
  of ISO 31000), applied to risk factors *you* enumerate -- it has no
  opinion on what those risks should be for your specific project.

Design
------
As with `core.voice_interface` (TTS/STT) and `core.knowledge_graph`
(storage/embeddings), the parts that need real domain expertise are
pushed behind small Protocols so they can be supplied or swapped without
touching this module:

- `DesignEvaluator` -- scores a candidate. Default: `WeightedSumEvaluator`,
  a generic multi-objective aggregator with zero embedded engineering
  knowledge (see docstring above).
- `OptimizationStrategy` -- proposes the next candidate to try. Two
  transparent, dependency-free defaults are included: `RandomSearchStrategy`
  and `HillClimbingStrategy`. Neither claims to be state-of-the-art; a
  real solver (scipy.optimize, a genetic algorithm, Bayesian optimization)
  can be wired in via the same Protocol.
- `DigitalTwinQueryable` -- the interface AutoEngineer needs from a
  Digital Twin module. No concrete Digital Twin exists in StarkOS yet
  (still on the roadmap); this Protocol lets AutoEngineer be written,
  tested and used today against a stand-in, and start working unmodified
  the day a real one implementing it is wired in via `bind_digital_twin()`.

`AutoEngineer` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    engineer = AutoEngineer(services=services)
    engineer.bind_knowledge_graph(knowledge_graph)
    kernel.register_module(engineer, name="auto_engineer", priority=250)

`bind_kernel()`/`bind_identity()`/`bind_knowledge_graph()`/
`bind_digital_twin()` mirror the pattern used by `Identity` and
`VoiceInterface`: dependencies that aren't -- or, for Digital Twin,
can't yet be -- resolved from the ServiceContainer are handed to the
module explicitly by the composition root.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Protocol, Sequence, Union, runtime_checkable

from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.service_container import ServiceContainer

logger = logging.getLogger("starkos.auto_engineer")

# =============================================================================
# Exceptions
# =============================================================================

class AutoEngineerError(Exception):
    """Base exception for AutoEngineer failures."""

class InvalidDesignSpecError(AutoEngineerError):
    """Raised when a DesignSpec, parameter, objective or constraint is malformed."""

class OptimizationError(AutoEngineerError):
    """Raised when a design evaluator fails during optimization."""

class BOMValidationError(AutoEngineerError):
    """Raised when an assembly/component tree is invalid (bad data, cycles)."""

class RiskAssessmentError(AutoEngineerError):
    """Raised when a risk assessment cannot be computed."""

class DigitalTwinUnavailableError(AutoEngineerError):
    """Raised when a Digital Twin operation is requested but none is bound
    (or the bound object doesn't implement the expected interface)."""

# =============================================================================
# Design Space: parameters, objectives, constraints
# =============================================================================

@dataclass(slots=True, frozen=True)
class DesignParameter:
    """One tunable dimension of the design (a bounded continuous value)."""

    name: str
    min_value: float
    max_value: float
    unit: str = ""

    def __post_init__(self) -> None:
        if self.min_value > self.max_value:
            raise InvalidDesignSpecError(f"Parameter '{self.name}': min_value cannot exceed max_value.")

    def clamp(self, value: float) -> float:
        return min(max(value, self.min_value), self.max_value)

    def midpoint(self) -> float:
        return (self.min_value + self.max_value) / 2.0

@dataclass(slots=True, frozen=True)
class Objective:
    """One metric to optimize, with a direction and relative weight."""

    name: str
    weight: float = 1.0
    direction: str = "maximize"  # "maximize" | "minimize"

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise InvalidDesignSpecError(f"Objective '{self.name}': direction must be 'maximize' or 'minimize'.")
        if self.weight <= 0:
            raise InvalidDesignSpecError(f"Objective '{self.name}': weight must be positive.")

@dataclass(slots=True, frozen=True)
class Constraint:
    """A bound on a parameter or a computed metric."""

    name: str
    target: str  # a DesignParameter name or an Objective/metric name
    operator: str  # ">=" | "<=" | "==" | "between"
    bound: Union[float, tuple[float, float]]

    def __post_init__(self) -> None:
        if self.operator == "between":
            if not (isinstance(self.bound, tuple) and len(self.bound) == 2):
                raise InvalidDesignSpecError(f"Constraint '{self.name}': 'between' requires a (low, high) tuple bound.")
        elif self.operator in (">=", "<=", "=="):
            if isinstance(self.bound, tuple):
                raise InvalidDesignSpecError(f"Constraint '{self.name}': operator '{self.operator}' requires a scalar bound.")
        else:
            raise InvalidDesignSpecError(f"Constraint '{self.name}': unknown operator '{self.operator}'.")

    def is_satisfied(self, value: float) -> bool:
        if self.operator == ">=":
            return value >= self.bound
        if self.operator == "<=":
            return value <= self.bound
        if self.operator == "==":
            return math.isclose(value, self.bound, rel_tol=1e-9)
        low, high = self.bound
        return low <= value <= high

@dataclass(slots=True, frozen=True)
class DesignSpec:
    """The full search space + evaluation criteria for one design problem."""

    name: str
    parameters: tuple[DesignParameter, ...]
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.parameters:
            raise InvalidDesignSpecError(f"DesignSpec '{self.name}' must declare at least one parameter.")
        if not self.objectives:
            raise InvalidDesignSpecError(f"DesignSpec '{self.name}' must declare at least one objective.")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise InvalidDesignSpecError(f"DesignSpec '{self.name}' has duplicate parameter names.")

    def parameter(self, name: str) -> DesignParameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise InvalidDesignSpecError(f"Unknown parameter '{name}' in spec '{self.name}'.")

# =============================================================================
# Candidates and evaluation results
# =============================================================================

@dataclass(slots=True, frozen=True)
class DesignCandidate:
    """One point in the design space -- a concrete assignment of parameter values."""

    id: str
    parameters: dict[str, float]
    iteration: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True, frozen=True)
class EvaluationResult:
    """The outcome of scoring one DesignCandidate against a DesignSpec."""

    candidate: DesignCandidate
    metrics: dict[str, float]
    score: float  # always "higher is better" by evaluator convention
    feasible: bool
    violated_constraints: tuple[str, ...] = ()

@dataclass(slots=True, frozen=True)
class DesignIterationResult:
    """The outcome of a full optimization run: the best candidate found
    plus the complete evaluation history."""

    spec_name: str
    best_evaluation: Optional[EvaluationResult]
    history: tuple[EvaluationResult, ...]
    iterations_run: int
    feasible_found: bool

# =============================================================================
# Shared constraint-checking helper
# =============================================================================

def _constraint_violations(spec: DesignSpec, values: dict[str, float]) -> list[str]:
    violated: list[str] = []
    for constraint in spec.constraints:
        value = values.get(constraint.target)
        if value is None:
            violated.append(f"{constraint.name}: target '{constraint.target}' not found among parameters/metrics.")
            continue
        if not constraint.is_satisfied(value):
            violated.append(f"{constraint.name}: {constraint.target}={value:.4g} violates {constraint.operator} {constraint.bound}")
    return violated

# =============================================================================
# Evaluator Protocol + default implementation
# =============================================================================

@runtime_checkable
class DesignEvaluator(Protocol):
    """Scores a DesignCandidate against a DesignSpec."""

    def evaluate(self, spec: DesignSpec, candidate: DesignCandidate) -> EvaluationResult:
        ...

class WeightedSumEvaluator:
    """
    Generic multi-objective evaluator with **no embedded engineering
    domain knowledge**: you supply one metric function per objective (a
    plain function of a DesignCandidate -- wrapping a real stress,
    thermal or cost formula, or a placeholder for testing). This class
    only aggregates those metrics into a single "higher is better" score
    using each objective's weight and direction, and checks the spec's
    constraints against both the candidate's parameters and the computed
    metrics.
    """

    def __init__(self, metric_functions: dict[str, Callable[[DesignCandidate], float]]) -> None:
        self._metric_functions = dict(metric_functions)

    def evaluate(self, spec: DesignSpec, candidate: DesignCandidate) -> EvaluationResult:
        metrics: dict[str, float] = {}
        for objective in spec.objectives:
            metric_fn = self._metric_functions.get(objective.name)
            if metric_fn is None:
                raise InvalidDesignSpecError(f"No metric function supplied for objective '{objective.name}'.")
            try:
                metrics[objective.name] = float(metric_fn(candidate))
            except Exception as exc:
                raise OptimizationError(f"Metric function for objective '{objective.name}' failed.") from exc

        score = 0.0
        for objective in spec.objectives:
            value = metrics[objective.name]
            contribution = value if objective.direction == "maximize" else -value
            score += objective.weight * contribution

        values = {**candidate.parameters, **metrics}
        violated = _constraint_violations(spec, values)

        return EvaluationResult(
            candidate=candidate,
            metrics=metrics,
            score=score,
            feasible=not violated,
            violated_constraints=tuple(violated),
        )

# =============================================================================
# Optimization strategies
# =============================================================================

@runtime_checkable
class OptimizationStrategy(Protocol):
    """Proposes the next DesignCandidate to evaluate, given the spec and
    the evaluation history so far."""

    def propose(
        self,
        spec: DesignSpec,
        history: Sequence[EvaluationResult],
        rng: random.Random,
    ) -> DesignCandidate:
        ...

class RandomSearchStrategy:
    """Samples each parameter uniformly at random within its bounds.
    No memory of history -- a simple, unbiased baseline."""

    def propose(self, spec: DesignSpec, history: Sequence[EvaluationResult], rng: random.Random) -> DesignCandidate:
        parameters = {parameter.name: rng.uniform(parameter.min_value, parameter.max_value) for parameter in spec.parameters}
        return DesignCandidate(id=str(uuid.uuid4()), parameters=parameters, iteration=len(history))

class HillClimbingStrategy:
    """
    Simple stochastic local search: perturbs the best feasible candidate
    found so far by a random step (scaled to a fraction of each
    parameter's range), clamped to bounds. Falls back to a uniform-random
    proposal when there is no (feasible) history yet, so it always starts
    from a valid point in the search space.

    This is a transparent, dependency-free default, not a claim of
    state-of-the-art optimization. For demanding real engineering
    problems, a proper solver (scipy.optimize, a genetic algorithm,
    Bayesian optimization) should be wired in via `OptimizationStrategy`.
    """

    def __init__(self, *, step_fraction: float = 0.1) -> None:
        if not (0.0 < step_fraction <= 1.0):
            raise ValueError("step_fraction must be in (0, 1].")
        self._step_fraction = step_fraction

    def propose(self, spec: DesignSpec, history: Sequence[EvaluationResult], rng: random.Random) -> DesignCandidate:
        feasible_history = [evaluation for evaluation in history if evaluation.feasible]
        pool = feasible_history or list(history)

        if not pool:
            parameters = {parameter.name: rng.uniform(parameter.min_value, parameter.max_value) for parameter in spec.parameters}
        else:
            base = max(pool, key=lambda evaluation: evaluation.score).candidate
            parameters = {}
            for parameter in spec.parameters:
                span = parameter.max_value - parameter.min_value
                step = rng.uniform(-1.0, 1.0) * span * self._step_fraction
                parameters[parameter.name] = parameter.clamp(base.parameters[parameter.name] + step)

        return DesignCandidate(id=str(uuid.uuid4()), parameters=parameters, iteration=len(history))

# =============================================================================
# Bill of Materials
# =============================================================================

@dataclass(slots=True, frozen=True)
class Component:
    """A single purchasable/manufacturable part used in an assembly."""

    part_number: str
    description: str
    quantity: float
    unit_cost: float
    unit: str = "unit"
    category: str = "general"
    supplier: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise BOMValidationError(f"Component '{self.part_number}': quantity must be positive.")
        if self.unit_cost < 0:
            raise BOMValidationError(f"Component '{self.part_number}': unit_cost cannot be negative.")

@dataclass(slots=True)
class Assembly:
    """A node in the BOM tree: its own components plus nested sub-assemblies."""

    id: str
    name: str
    components: list[Component] = field(default_factory=list)
    sub_assemblies: list["AssemblyReference"] = field(default_factory=list)

@dataclass(slots=True, frozen=True)
class AssemblyReference:
    """A sub-assembly used `quantity` times within its parent."""

    assembly: Assembly
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise BOMValidationError(f"Sub-assembly reference to '{self.assembly.name}': quantity must be positive.")

@dataclass(slots=True, frozen=True)
class BOMLineItem:
    """One flattened, aggregated row of a generated Bill of Materials."""

    part_number: str
    description: str
    category: str
    total_quantity: float
    unit: str
    unit_cost: float
    total_cost: float

@dataclass(slots=True, frozen=True)
class BillOfMaterials:
    project_name: str
    line_items: tuple[BOMLineItem, ...]
    total_cost: float
    currency: str = "USD"
    generated_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Risk assessment
# =============================================================================

@dataclass(slots=True, frozen=True)
class RiskFactor:
    """One entry in a project risk register, scored on a standard 1-5 x
    1-5 likelihood/impact matrix."""

    name: str
    category: str
    description: str
    likelihood: int  # 1 (rare) - 5 (near certain)
    impact: int  # 1 (negligible) - 5 (severe)
    mitigation: Optional[str] = None

    def __post_init__(self) -> None:
        if not (1 <= self.likelihood <= 5):
            raise RiskAssessmentError(f"Risk '{self.name}': likelihood must be between 1 and 5.")
        if not (1 <= self.impact <= 5):
            raise RiskAssessmentError(f"Risk '{self.name}': impact must be between 1 and 5.")

    @property
    def score(self) -> int:
        return self.likelihood * self.impact

@dataclass(slots=True, frozen=True)
class RiskAssessment:
    project_name: str
    risks: tuple[RiskFactor, ...]
    overall_score: float
    overall_level: str  # "low" | "medium" | "high" | "critical"
    generated_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Digital Twin integration Protocol (no concrete implementation exists yet)
# =============================================================================

@runtime_checkable
class DigitalTwinQueryable(Protocol):
    """
    Structural contract AutoEngineer needs from a Digital Twin module.
    No concrete Digital Twin exists yet in StarkOS (still on the
    roadmap) -- this Protocol lets AutoEngineer be written and tested
    against a stand-in today, and start working unmodified the day a
    real Digital Twin implementing it is wired in via
    `bind_digital_twin()`.
    """

    def get_asset_state(self, asset_id: str) -> dict[str, Any]:
        ...

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class AutoEngineerConfig:
    default_iterations: int = 50
    random_seed: Optional[int] = None
    # Boundaries between low/medium, medium/high, high/critical on the
    # 1-25 (1-5 x 1-5) risk-matrix scale.
    risk_thresholds: tuple[float, float, float] = (5.0, 10.0, 20.0)
    record_to_knowledge_graph: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# AutoEngineer
# =============================================================================

class AutoEngineer:
    """
    StarkOS engineering design orchestration module.

    Satisfies the `Module` protocol (name/initialize/start/stop) and can
    be registered with the Kernel like any other module. See the module
    docstring for the (deliberately limited, honest) scope of what this
    class does and does not know about real engineering.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[AutoEngineerConfig] = None,
    ) -> None:
        self._services = services
        self._config = config or AutoEngineerConfig()

        self._kernel: Any = None
        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._digital_twin: Any = None

        logger.info("AutoEngineer constructed.", extra={"default_iterations": self._config.default_iterations})

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "auto_engineer"

    def bind_kernel(self, kernel: Any) -> None:
        """Kernel does not register itself into the ServiceContainer, so
        it is handed to modules explicitly -- mirrors Identity/VoiceInterface."""
        self._kernel = kernel
        logger.debug("Kernel bound to AutoEngineer.")

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to AutoEngineer.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to AutoEngineer.")

    def bind_digital_twin(self, digital_twin: Any) -> None:
        self._digital_twin = digital_twin
        logger.debug("Digital Twin bound to AutoEngineer.")

    async def initialize(self) -> None:
        logger.info("Initializing AutoEngineer.")
        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        logger.info(
            "AutoEngineer initialized.",
            extra={"knowledge_graph_bound": self._knowledge_graph is not None, "identity_bound": self._identity is not None},
        )

    async def start(self) -> None:
        logger.info("AutoEngineer ready.")

    async def stop(self) -> None:
        logger.info("AutoEngineer stopped.")

    # ------------------------------------------------------------------
    # Design optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        spec: DesignSpec,
        evaluator: DesignEvaluator,
        *,
        strategy: Optional[OptimizationStrategy] = None,
        iterations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DesignIterationResult:
        if not isinstance(spec, DesignSpec):
            raise InvalidDesignSpecError("spec must be a DesignSpec instance.")

        strategy = strategy or HillClimbingStrategy()
        total_iterations = iterations if iterations is not None else self._config.default_iterations
        if total_iterations <= 0:
            raise InvalidDesignSpecError("iterations must be positive.")

        rng = random.Random(seed if seed is not None else self._config.random_seed)

        history: list[EvaluationResult] = []
        logger.info("Starting design optimization.", extra={"spec_name": spec.name, "iterations": total_iterations})

        for _ in range(total_iterations):
            candidate = strategy.propose(spec, history, rng)
            try:
                evaluation = evaluator.evaluate(spec, candidate)
            except AutoEngineerError:
                raise
            except Exception as exc:
                raise OptimizationError(f"Evaluator failed for spec '{spec.name}'.") from exc
            history.append(evaluation)

        feasible_evaluations = [evaluation for evaluation in history if evaluation.feasible]
        if feasible_evaluations:
            best: Optional[EvaluationResult] = max(feasible_evaluations, key=lambda evaluation: evaluation.score)
            feasible_found = True
        elif history:
            best = max(history, key=lambda evaluation: evaluation.score)
            feasible_found = False
            logger.warning(
                "No feasible design found within constraints -- returning the best infeasible candidate.",
                extra={"spec_name": spec.name},
            )
        else:
            best = None
            feasible_found = False

        result = DesignIterationResult(
            spec_name=spec.name,
            best_evaluation=best,
            history=tuple(history),
            iterations_run=total_iterations,
            feasible_found=feasible_found,
        )

        logger.info(
            "Design optimization completed.",
            extra={"spec_name": spec.name, "feasible_found": feasible_found, "best_score": best.score if best else None},
        )

        if self._config.record_to_knowledge_graph:
            self.record_design_iteration(spec, result)

        return result

    async def optimize_async(
        self,
        spec: DesignSpec,
        evaluator: DesignEvaluator,
        *,
        strategy: Optional[OptimizationStrategy] = None,
        iterations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> DesignIterationResult:
        """Async wrapper for `optimize()` -- useful when the evaluator (a
        real simulation, perhaps) is heavy enough to want off the event
        loop thread."""
        return await asyncio.to_thread(
            self.optimize, spec, evaluator, strategy=strategy, iterations=iterations, seed=seed
        )

    def check_feasibility(
        self,
        spec: DesignSpec,
        candidate: DesignCandidate,
        metrics: Optional[dict[str, float]] = None,
    ) -> tuple[bool, tuple[str, ...]]:
        """
        Check a candidate (and optionally its already-computed metrics)
        against the spec's constraints directly, without running a full
        evaluator. Returns (feasible, violated_constraint_descriptions).
        """
        values = {**candidate.parameters, **(metrics or {})}
        violated = _constraint_violations(spec, values)
        return (not violated, tuple(violated))

    # ------------------------------------------------------------------
    # Bill of Materials
    # ------------------------------------------------------------------

    def generate_bom(self, assembly: Assembly, *, currency: str = "USD") -> BillOfMaterials:
        if not isinstance(assembly, Assembly):
            raise BOMValidationError("assembly must be an Assembly instance.")

        aggregated: dict[str, dict[str, Any]] = {}

        def _walk(node: Assembly, multiplier: float, visited_path: tuple[str, ...]) -> None:
            if node.id in visited_path:
                raise BOMValidationError(f"Cyclic assembly reference detected involving '{node.id}'.")

            for component in node.components:
                effective_quantity = component.quantity * multiplier
                entry = aggregated.get(component.part_number)
                if entry is None:
                    aggregated[component.part_number] = {
                        "description": component.description,
                        "category": component.category,
                        "unit": component.unit,
                        "unit_cost": component.unit_cost,
                        "total_quantity": effective_quantity,
                    }
                else:
                    if not math.isclose(entry["unit_cost"], component.unit_cost, rel_tol=1e-9):
                        logger.warning(
                            "Part number '%s' has inconsistent unit cost across the assembly "
                            "(%.4f vs %.4f) -- keeping the first value seen.",
                            component.part_number, entry["unit_cost"], component.unit_cost,
                        )
                    entry["total_quantity"] += effective_quantity

            for reference in node.sub_assemblies:
                _walk(reference.assembly, multiplier * reference.quantity, visited_path + (node.id,))

        _walk(assembly, 1.0, ())

        line_items = tuple(
            BOMLineItem(
                part_number=part_number,
                description=data["description"],
                category=data["category"],
                total_quantity=data["total_quantity"],
                unit=data["unit"],
                unit_cost=data["unit_cost"],
                total_cost=round(data["total_quantity"] * data["unit_cost"], 2),
            )
            for part_number, data in sorted(aggregated.items())
        )
        total_cost = round(sum(item.total_cost for item in line_items), 2)

        bom = BillOfMaterials(project_name=assembly.name, line_items=line_items, total_cost=total_cost, currency=currency)
        logger.info(
            "BOM generated.",
            extra={"project_name": assembly.name, "line_items": len(line_items), "total_cost": total_cost},
        )

        if self._config.record_to_knowledge_graph:
            self.record_bom(bom)

        return bom

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def assess_risks(self, project_name: str, risks: Sequence[RiskFactor]) -> RiskAssessment:
        """
        Aggregate a risk register into an overall score using a standard
        likelihood x impact risk-matrix convention (each 1-5, so each
        risk scores 1-25; overall = 50% worst single risk + 50% average
        risk). This is a generic project-risk technique (in the spirit
        of ISO 31000 risk matrices), not a certified engineering safety
        analysis -- domain-specific safety margins (structural, thermal,
        electrical) must come from your own evaluators/constraints.
        """
        if not risks:
            raise RiskAssessmentError("At least one risk factor is required for an assessment.")

        scores = [risk.score for risk in risks]
        overall_score = round(max(scores) * 0.5 + (sum(scores) / len(scores)) * 0.5, 2)
        overall_level = self._classify_risk(overall_score)

        assessment = RiskAssessment(
            project_name=project_name,
            risks=tuple(risks),
            overall_score=overall_score,
            overall_level=overall_level,
        )

        logger.info(
            "Risk assessment completed.",
            extra={"project_name": project_name, "overall_level": overall_level, "risk_count": len(risks)},
        )

        if self._config.record_to_knowledge_graph:
            self.record_risk_assessment(assessment)

        return assessment

    def _classify_risk(self, score: float) -> str:
        low, medium, high = self._config.risk_thresholds
        if score < low:
            return "low"
        if score < medium:
            return "medium"
        if score < high:
            return "high"
        return "critical"

    # ------------------------------------------------------------------
    # Digital Twin integration
    # ------------------------------------------------------------------

    def seed_candidate_from_digital_twin(self, spec: DesignSpec, asset_id: str) -> DesignCandidate:
        """
        Pull an asset's current state from the bound Digital Twin and use
        whichever fields match `spec` parameter names as the initial
        values for a design candidate (useful for "optimize starting from
        the real, currently deployed configuration" workflows). Parameters
        with no matching field in the twin's state -- or a non-numeric
        value -- fall back to the midpoint of their declared bounds.
        """
        if self._digital_twin is None:
            raise DigitalTwinUnavailableError("No Digital Twin bound -- call bind_digital_twin() first.")

        get_state = getattr(self._digital_twin, "get_asset_state", None)
        if get_state is None:
            raise DigitalTwinUnavailableError("Bound Digital Twin does not implement get_asset_state().")

        try:
            state = get_state(asset_id)
        except Exception as exc:
            raise DigitalTwinUnavailableError(f"Failed to read state for asset '{asset_id}'.") from exc

        parameters: dict[str, float] = {}
        for parameter in spec.parameters:
            raw_value = state.get(parameter.name) if isinstance(state, dict) else None
            if raw_value is None:
                parameters[parameter.name] = parameter.midpoint()
                continue
            try:
                parameters[parameter.name] = parameter.clamp(float(raw_value))
            except (TypeError, ValueError):
                logger.warning(
                    "Digital Twin value for parameter '%s' is not numeric -- using midpoint instead.",
                    parameter.name,
                )
                parameters[parameter.name] = parameter.midpoint()

        logger.info("Design candidate seeded from Digital Twin.", extra={"asset_id": asset_id, "spec_name": spec.name})
        return DesignCandidate(id=str(uuid.uuid4()), parameters=parameters, iteration=0)

    # ------------------------------------------------------------------
    # KnowledgeGraph integration
    # ------------------------------------------------------------------

    def record_design_iteration(self, spec: DesignSpec, result: DesignIterationResult) -> Optional[Any]:
        if self._knowledge_graph is None:
            logger.debug("record_design_iteration skipped -- no KnowledgeGraph bound.")
            return None

        best = result.best_evaluation
        content = (
            f"Design optimization for '{spec.name}': {result.iterations_run} iterations, "
            f"feasible_found={result.feasible_found}, best_score={best.score if best else 'n/a'}"
        )
        metadata = {
            "spec_name": spec.name,
            "iterations_run": result.iterations_run,
            "feasible_found": result.feasible_found,
            "best_parameters": best.candidate.parameters if best else None,
            "best_metrics": best.metrics if best else None,
        }
        try:
            return self._knowledge_graph.remember(content, node_type="design_iteration", metadata=metadata, source="auto_engineer")
        except Exception:
            logger.exception("Failed to record design iteration in KnowledgeGraph.")
            return None

    def record_bom(self, bom: BillOfMaterials) -> Optional[Any]:
        if self._knowledge_graph is None:
            logger.debug("record_bom skipped -- no KnowledgeGraph bound.")
            return None

        content = f"BOM for '{bom.project_name}': {len(bom.line_items)} line items, total cost {bom.total_cost} {bom.currency}"
        metadata = {
            "project_name": bom.project_name,
            "total_cost": bom.total_cost,
            "currency": bom.currency,
            "line_item_count": len(bom.line_items),
        }
        try:
            return self._knowledge_graph.remember(content, node_type="bom", metadata=metadata, source="auto_engineer")
        except Exception:
            logger.exception("Failed to record BOM in KnowledgeGraph.")
            return None

    def record_risk_assessment(self, assessment: RiskAssessment) -> Optional[Any]:
        if self._knowledge_graph is None:
            logger.debug("record_risk_assessment skipped -- no KnowledgeGraph bound.")
            return None

        content = f"Risk assessment for '{assessment.project_name}': level={assessment.overall_level}, score={assessment.overall_score}"
        metadata = {
            "project_name": assessment.project_name,
            "overall_score": assessment.overall_score,
            "overall_level": assessment.overall_level,
            "risk_count": len(assessment.risks),
        }
        try:
            return self._knowledge_graph.remember(content, node_type="risk_assessment", metadata=metadata, source="auto_engineer")
        except Exception:
            logger.exception("Failed to record risk assessment in KnowledgeGraph.")
            return None

    def recall_similar_designs(self, query: str, *, top_k: int = 5) -> tuple[Any, ...]:
        """Semantic recall over past design iterations/BOMs/risk
        assessments recorded in the bound KnowledgeGraph."""
        if self._knowledge_graph is None:
            raise AutoEngineerError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")
        return self._knowledge_graph.recall(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "digital_twin_bound": self._digital_twin is not None,
            "kernel_bound": self._kernel is not None,
            "identity_bound": self._identity is not None,
            "default_iterations": self._config.default_iterations,
        }