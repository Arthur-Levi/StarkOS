"""
core/simulation_orchestrator.py
==================================

Simulation orchestration for StarkOS: runs, interprets and validates
technical simulations (structural, thermal, fluid, dynamic, fatigue,
...) against requirements -- without implementing any physics itself.

Responsibilities
----------------
- Provide a standardized `SimulationConnector` interface for external
  simulators (FEA, CFD, multibody dynamics, fatigue tools, ...) to plug
  into, regardless of simulation type or vendor.
- Run a simulation request through whichever connector is registered
  for its type, with the same execution isolation (timeout, exception
  containment) used elsewhere in StarkOS.
- Interpret raw simulator output into named quantities with real units,
  and compare them against requirements (reusing
  `core.auto_engineer.Constraint` directly) to compute pass/fail and
  numeric safety margins.
- Detect failures, insufficient margins and inconsistencies, and
  package the outcome as explicitly-classified `Claim`s (reusing
  `core.verification_engine.Claim`/`ClaimType`) verified through
  `VerificationEngine` -- never a bespoke, parallel classification.
- Close the full validation loop: Requirements -> Design -> Simulation
  -> Interpretation -> Verification -> Adjustment recommendation,
  recording every stage into DigitalThread.
- Integrate with CognitiveEngine and AutonomousEngineeringLoop as a
  bindable capability (see the honesty note on what "integration" means
  here, since neither is imported concretely to avoid the dependency
  cycles already present between StarkOS's cognitive-stack modules).

Honesty about scope
--------------------
1. **This module implements no physics.** It ships no FEA/CFD/fatigue
   solver and never fabricates a simulation result. Every numeric
   output in a `SimulationRunResult` comes from a real
   `SimulationConnector` you register and connect to an actual
   simulation tool; the default state (no connector registered/reachable
   for a given `simulation_type`) is `SimulationUnavailableError`, never
   an invented number.

2. **"Interpretation" is unit-aware bookkeeping and margin arithmetic,
   not engineering judgment.** Comparing a simulated stress against an
   allowable and computing `(limit - actual) / limit` is honest,
   deterministic math. Deciding whether a resulting margin is
   *acceptable* for your application is an engineering judgment call
   this module does not make -- it reports the number and lets
   `VerificationEngine`'s constraint checking (which you configure)
   determine pass/fail.

3. **"Recommendation of adjustment" is mechanically derived, generic
   guidance** ("output X exceeds requirement Y by Z%; the design
   parameter(s) tied to X may need revisiting"), tagged
   `ClaimType.RECOMMENDATION` and never presented as a specific,
   validated engineering fix. It is not a substitute for an engineer
   reviewing the actual result.

4. **Assumptions are exactly what you declare them to be.** Every
   `SimulationRequest.assumptions` entry becomes an explicit
   `ClaimType.HYPOTHESIS` claim -- this module does not infer hidden
   assumptions a simulation might be making.

Design
------
Same shape as the rest of StarkOS: a `SimulationConnector` Protocol
with zero shipped physics implementations (only a `NullSimulationConnector`
that honestly reports unavailable), reusing already-built, already-
tested components (`auto_engineer.Constraint`, `verification_engine.Claim`
and the `VerificationEngine` itself) instead of duplicating them.
`CognitiveEngine`/`AutonomousEngineeringLoop` are bound via `Any` (not
imported concretely) -- consistent with how `Kernel` itself is handled
everywhere else in StarkOS, and here specifically avoiding real import
cycles already present among the cognitive-stack modules.

`SimulationOrchestrator` satisfies the `Module` protocol (name/
initialize/start/stop) and registers with the Kernel like any other
StarkOS module:

    orchestrator = SimulationOrchestrator(services=services)
    orchestrator.bind_verification_engine(verification_engine)
    orchestrator.bind_digital_thread(digital_thread)
    orchestrator.register_connector(MyFEAConnector())
    kernel.register_module(orchestrator, name="simulation_orchestrator", priority=320)

    interpretation = orchestrator.run_simulation(
        SimulationRequest(
            id="sim-1", simulation_type=SimulationType.STRUCTURAL, subject="motor_mount",
            parameters=(SimulationParameter(name="load", value=500.0, units="N"),),
            assumptions=("Load is static, not cyclic.",),
        ),
        requirements=(Constraint(name="max_stress", target="von_mises_stress", operator="<=", bound=250.0),),
    )
    print(interpretation.verdict, [m.margin for m in interpretation.margins])
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from core.auto_engineer import Constraint
from core.digital_thread import DigitalThread
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.service_container import ServiceContainer
from core.verification_engine import (
    Claim,
    ClaimType,
    EvidenceReference,
    VerificationEngine,
    VerificationReport,
)

logger = get_logger("simulation_orchestrator")

# =============================================================================
# Exceptions
# =============================================================================

class SimulationOrchestratorError(Exception):
    """Base exception for SimulationOrchestrator failures."""

class SimulationUnavailableError(SimulationOrchestratorError):
    """Raised when no connector is registered/reachable for a requested
    simulation type."""

class SimulationExecutionError(SimulationOrchestratorError):
    """Raised when a connector raises during a run."""

class InvalidSimulationRequestError(SimulationOrchestratorError):
    """Raised when a SimulationRequest/requirement set is malformed."""

# =============================================================================
# Simulation type vocabulary
# =============================================================================

class SimulationType:
    """Well-known `simulation_type` values. Stays a free string --
    connectors declare their own; a custom simulator can use any name."""

    STRUCTURAL = "structural"
    THERMAL = "thermal"
    FLUID = "fluid"
    DYNAMIC = "dynamic"
    FATIGUE = "fatigue"
    MODAL = "modal"
    ELECTROMAGNETIC = "electromagnetic"

# =============================================================================
# Data models
# =============================================================================

@dataclass(slots=True, frozen=True)
class SimulationParameter:
    name: str
    value: float
    units: Optional[str] = None

@dataclass(slots=True, frozen=True)
class SimulationRequest:
    id: str
    simulation_type: str
    subject: str  # e.g. a component name or a KnowledgeGraph node id
    parameters: tuple[SimulationParameter, ...]
    assumptions: tuple[str, ...] = ()  # each becomes an explicit HYPOTHESIS claim
    connector_name: Optional[str] = None  # force a specific connector instead of type-based lookup
    requested_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidSimulationRequestError("SimulationRequest.id is required.")
        if not self.subject:
            raise InvalidSimulationRequestError("SimulationRequest.subject is required.")

@dataclass(slots=True, frozen=True)
class SimulationOutput:
    """One raw, uninterpreted output quantity from a simulation run."""

    name: str
    value: float
    units: Optional[str] = None

@dataclass(slots=True, frozen=True)
class SimulationRunResult:
    """What a connector returns. Interpretation/comparison/verification
    all happen afterward, in SimulationOrchestrator -- never inside the
    connector itself."""

    request_id: str
    connector_name: str
    outputs: tuple[SimulationOutput, ...]
    converged: Optional[bool] = None  # did the solver report convergence, if it reports this at all
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: Optional[float] = None

@dataclass(slots=True, frozen=True)
class MarginResult:
    """A single requirement-vs-output comparison, with a real numeric
    safety margin -- not a pass/fail label alone."""

    requirement_name: str
    output_name: str
    actual_value: float
    limit: Any  # the Constraint's own bound (float, or a (low, high) tuple for "between")
    operator: str
    margin: Optional[float]  # (limit - actual) / limit as a fraction, when a single scalar bound applies; None for "between"/"=="
    passes: bool

@dataclass(slots=True, frozen=True)
class SimulationInterpretation:
    """The fully-interpreted, verified outcome of one simulation run --
    the actual deliverable of this module."""

    request: SimulationRequest
    run_result: SimulationRunResult
    margins: tuple[MarginResult, ...]
    claims: tuple[Claim, ...]
    verification_report: Optional[VerificationReport]
    verdict: str  # mirrors verification_report.verdict, or "unverified" if no VerificationEngine is bound
    digital_thread_trace_id: Optional[str] = None
    interpreted_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True, frozen=True)
class ValidationCycleResult:
    """The outcome of close_validation_cycle(): the named Requirements
    -> Design -> Simulation -> Interpretation -> Verification ->
    Adjustment-recommendation sequence, fully traced."""

    subject: str
    trace_id: Optional[str]
    interpretation: SimulationInterpretation
    adjustment_recommendations: tuple[Claim, ...]
    completed_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Connector Protocol (the standardized simulator interface)
# =============================================================================

@runtime_checkable
class SimulationConnector(Protocol):
    """
    The interface any external simulator plugs into. Implementations
    are entirely the caller's responsibility -- SimulationOrchestrator
    ships none beyond `NullSimulationConnector` (see honesty note 1).
    """

    name: str
    simulation_type: str

    def check_available(self) -> bool: ...
    def run(self, request: SimulationRequest) -> SimulationRunResult: ...

class NullSimulationConnector:
    """Honest default/placeholder: always reports unavailable and never
    fabricates a result. Registering a real connector for a given
    `simulation_type` is what actually enables `run_simulation()` for it."""

    def __init__(self, simulation_type: str) -> None:
        self.name = f"null_{simulation_type}"
        self.simulation_type = simulation_type

    def check_available(self) -> bool:
        return False

    def run(self, request: SimulationRequest) -> SimulationRunResult:
        raise SimulationUnavailableError(
            f"No real simulator is connected for type '{self.simulation_type}'. StarkOS ships no simulation "
            "engine of its own -- register a real SimulationConnector via register_connector()."
        )

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class SimulationOrchestratorConfig:
    default_timeout_seconds: float = 120.0
    record_to_knowledge_graph: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Simulation Orchestrator
# =============================================================================

class SimulationOrchestrator:
    """
    StarkOS's simulation orchestration module. See the module
    docstring's "Honesty about scope" section -- especially that it
    implements no physics of its own -- before relying on it.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(self, *, services: ServiceContainer, config: Optional[SimulationOrchestratorConfig] = None) -> None:
        self._services = services
        self._config = config or SimulationOrchestratorConfig()
        self._connectors: dict[str, SimulationConnector] = {}  # keyed by connector name
        self._connectors_by_type: dict[str, list[str]] = {}  # simulation_type -> [connector names]

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._digital_thread: Optional[DigitalThread] = None
        self._verification_engine: Optional[VerificationEngine] = None
        self._cognitive_engine: Any = None
        self._autonomous_loop: Any = None

        logger.info("SimulationOrchestrator constructed.")

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "simulation_orchestrator"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to SimulationOrchestrator.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to SimulationOrchestrator.")

    def bind_digital_thread(self, digital_thread: DigitalThread) -> None:
        self._digital_thread = digital_thread
        logger.debug("DigitalThread bound to SimulationOrchestrator.")

    def bind_verification_engine(self, verification_engine: VerificationEngine) -> None:
        self._verification_engine = verification_engine
        logger.debug("VerificationEngine bound to SimulationOrchestrator.")

    def bind_cognitive_engine(self, cognitive_engine: Any) -> None:
        """Stored for CognitiveEngine (or a future specialist wiring) to
        call into run_simulation()/close_validation_cycle() -- this
        module never calls back into CognitiveEngine itself."""
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to SimulationOrchestrator.")

    def bind_autonomous_loop(self, autonomous_loop: Any) -> None:
        """Stored for AutonomousEngineeringLoop to use as a stage within
        its own cycle -- see the module docstring's honesty note on
        why this is Any-typed rather than a concrete import."""
        self._autonomous_loop = autonomous_loop
        logger.debug("AutonomousEngineeringLoop bound to SimulationOrchestrator.")

    async def initialize(self) -> None:
        logger.info("Initializing SimulationOrchestrator.")
        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._digital_thread is None:
            self._digital_thread = self._services.resolve_optional(DigitalThread)
        if self._verification_engine is None:
            self._verification_engine = self._services.resolve_optional(VerificationEngine)

        if not self._connectors:
            logger.warning("No simulation connectors registered yet -- run_simulation() will raise SimulationUnavailableError until register_connector() is called.")

        logger.info("SimulationOrchestrator initialized.", extra={"connector_count": len(self._connectors)})

    async def start(self) -> None:
        logger.info("SimulationOrchestrator ready.", extra={"connectors": list(self._connectors.keys())})

    async def stop(self) -> None:
        logger.info("SimulationOrchestrator stopped.")

    # ------------------------------------------------------------------
    # Connector registration
    # ------------------------------------------------------------------

    def register_connector(self, connector: SimulationConnector) -> None:
        self._connectors[connector.name] = connector
        self._connectors_by_type.setdefault(connector.simulation_type, [])
        if connector.name not in self._connectors_by_type[connector.simulation_type]:
            self._connectors_by_type[connector.simulation_type].append(connector.name)
        logger.info("Simulation connector registered.", extra={"connector": connector.name, "simulation_type": connector.simulation_type})

    def unregister_connector(self, name: str) -> bool:
        connector = self._connectors.pop(name, None)
        if connector is None:
            return False
        names = self._connectors_by_type.get(connector.simulation_type, [])
        if name in names:
            names.remove(name)
        logger.info("Simulation connector unregistered.", extra={"connector": name})
        return True

    def list_connectors(self) -> tuple[SimulationConnector, ...]:
        return tuple(self._connectors.values())

    def _resolve_connector(self, request: SimulationRequest) -> SimulationConnector:
        if request.connector_name is not None:
            connector = self._connectors.get(request.connector_name)
            if connector is None:
                raise SimulationUnavailableError(f"No connector named '{request.connector_name}' is registered.")
            return connector

        candidate_names = self._connectors_by_type.get(request.simulation_type, [])
        for candidate_name in candidate_names:
            connector = self._connectors[candidate_name]
            try:
                if connector.check_available():
                    return connector
            except Exception:
                logger.exception("Availability check raised for connector '%s'.", candidate_name)

        raise SimulationUnavailableError(
            f"No available connector for simulation_type '{request.simulation_type}'. StarkOS ships no "
            "simulation engine of its own -- register one via register_connector()."
        )

    # ------------------------------------------------------------------
    # Execution (isolated, honest failure)
    # ------------------------------------------------------------------

    async def _run_with_isolation(self, connector: SimulationConnector, request: SimulationRequest) -> SimulationRunResult:
        def _call() -> SimulationRunResult:
            return connector.run(request)

        try:
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=self._config.default_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise SimulationExecutionError(
                f"Simulation '{request.id}' via connector '{connector.name}' timed out after {self._config.default_timeout_seconds}s."
            ) from exc
        except SimulationOrchestratorError:
            raise
        except Exception as exc:
            raise SimulationExecutionError(f"Connector '{connector.name}' raised while running '{request.id}': {exc}") from exc

    # ------------------------------------------------------------------
    # Interpretation: margins, comparisons, claim classification
    # ------------------------------------------------------------------

    def _compute_margin(self, requirement: Constraint, output: SimulationOutput) -> MarginResult:
        passes = requirement.is_satisfied(output.value)
        margin: Optional[float] = None
        if requirement.operator in ("<=", ">=") and isinstance(requirement.bound, (int, float)) and requirement.bound != 0:
            if requirement.operator == "<=":
                margin = (requirement.bound - output.value) / abs(requirement.bound)
            else:  # ">="
                margin = (output.value - requirement.bound) / abs(requirement.bound)
        return MarginResult(
            requirement_name=requirement.name, output_name=output.name, actual_value=output.value,
            limit=requirement.bound, operator=requirement.operator, margin=margin, passes=passes,
        )

    def _build_claims(
        self, request: SimulationRequest, run_result: SimulationRunResult, margins: Sequence[MarginResult],
    ) -> list[Claim]:
        claims: list[Claim] = []

        # Assumptions -> explicit HYPOTHESIS claims (never inferred, always what the caller declared).
        for index, assumption in enumerate(request.assumptions):
            claims.append(Claim(
                id=f"{request.id}-assumption-{index}", text=assumption, claim_type=ClaimType.HYPOTHESIS.name,
                metadata={"quantity": None, "role": "assumption"},
            ))

        # Raw simulator outputs -> FACT claims, evidenced by the run itself.
        for output in run_result.outputs:
            claims.append(Claim(
                id=f"{request.id}-output-{output.name}", text=f"Simulated {output.name} = {output.value} {output.units or ''}".strip(),
                claim_type=ClaimType.FACT.name,
                evidence=(EvidenceReference(source_type="measurement", reference=f"{run_result.connector_name}:{request.id}", description="Simulation run output"),),
                units=output.units, numeric_value=output.value,
                metadata={"quantity": output.name, "role": "simulation_output"},
            ))

        # Margin comparisons -> ESTIMATE claims (a real number, but every
        # simulation result carries numerical/modeling uncertainty).
        for margin in margins:
            text = (
                f"{margin.output_name} = {margin.actual_value} vs requirement '{margin.requirement_name}' "
                f"({margin.operator} {margin.limit}): {'PASS' if margin.passes else 'FAIL'}"
                + (f", margin={margin.margin:.1%}" if margin.margin is not None else "")
            )
            claims.append(Claim(
                id=f"{request.id}-margin-{margin.requirement_name}", text=text, claim_type=ClaimType.ESTIMATE.name,
                evidence=(EvidenceReference(source_type="measurement", reference=f"{run_result.connector_name}:{request.id}", description="Margin computed from simulation output"),),
                numeric_value=margin.actual_value, metadata={"quantity": margin.output_name, "role": "margin", "passes": margin.passes},
            ))

            # Mechanically-derived, generic adjustment guidance -- see
            # honesty note 3: never a specific validated fix.
            if not margin.passes:
                claims.append(Claim(
                    id=f"{request.id}-recommendation-{margin.requirement_name}",
                    text=(
                        f"'{margin.output_name}' fails requirement '{margin.requirement_name}' "
                        f"({margin.actual_value} {margin.operator} {margin.limit} not satisfied) -- the design "
                        f"parameter(s) driving '{margin.output_name}' likely need revisiting."
                    ),
                    claim_type=ClaimType.RECOMMENDATION.name,
                    metadata={"quantity": margin.output_name, "role": "adjustment_recommendation"},
                ))

        return claims

    # ------------------------------------------------------------------
    # Main entry point: run + interpret + verify one simulation
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        request: SimulationRequest,
        *,
        requirements: Sequence[Constraint] = (),
        actor: Optional[str] = None,
    ) -> SimulationInterpretation:
        """Synchronous convenience wrapper -- see run_simulation_async()
        for the real (I/O-bound) execution path. This blocks the calling
        thread; prefer the async version from async code."""
        return asyncio.get_event_loop().run_until_complete(self.run_simulation_async(request, requirements=requirements, actor=actor))

    async def run_simulation_async(
        self,
        request: SimulationRequest,
        *,
        requirements: Sequence[Constraint] = (),
        actor: Optional[str] = None,
    ) -> SimulationInterpretation:
        connector = self._resolve_connector(request)
        run_result = await self._run_with_isolation(connector, request)

        output_by_name = {output.name: output for output in run_result.outputs}
        margins: list[MarginResult] = []
        for requirement in requirements:
            output = output_by_name.get(requirement.target)
            if output is None:
                logger.warning("Requirement '%s' targets output '%s', which the simulation did not produce.", requirement.name, requirement.target)
                continue
            margins.append(self._compute_margin(requirement, output))

        claims = self._build_claims(request, run_result, margins)

        verification_report: Optional[VerificationReport] = None
        verdict = "unverified"
        if self._verification_engine is not None:
            constraint_values = {output.name: output.value for output in run_result.outputs}
            verification_report = self._verification_engine.verify(
                subject=f"Simulation '{request.id}' ({request.simulation_type}) on '{request.subject}'",
                claims=claims, constraints=list(requirements), constraint_values=constraint_values, actor=actor,
            )
            verdict = verification_report.verdict
        else:
            logger.warning("No VerificationEngine bound -- simulation claims were built but not formally verified.")

        trace_id = self._record_to_digital_thread(request, run_result, margins, verification_report, actor)

        interpretation = SimulationInterpretation(
            request=request, run_result=run_result, margins=tuple(margins), claims=tuple(claims),
            verification_report=verification_report, verdict=verdict, digital_thread_trace_id=trace_id,
        )

        logger.info(
            "Simulation interpreted.",
            extra={"request_id": request.id, "simulation_type": request.simulation_type, "verdict": verdict, "margin_count": len(margins)},
        )

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            self._mirror_to_knowledge_graph(interpretation)

        return interpretation

    # ------------------------------------------------------------------
    # Full validation-cycle closure: Requirements -> Design -> Simulation
    # -> Interpretation -> Verification -> Adjustment recommendation
    # ------------------------------------------------------------------

    async def close_validation_cycle(
        self,
        *,
        subject: str,
        request: SimulationRequest,
        requirements: Sequence[Constraint],
        actor: Optional[str] = None,
    ) -> ValidationCycleResult:
        """
        Runs the named stages explicitly and records each into
        DigitalThread under one shared trace: Requirements (what must
        hold) -> Design (the subject being checked) -> Simulation (the
        run) -> Interpretation (margins) -> Verification (the formal
        verdict) -> Adjustment recommendation (mechanically-derived
        guidance for anything that failed).
        """
        trace_id = None
        parent_id = None
        if self._digital_thread is not None:
            trace_id = self._digital_thread.begin_trace(f"Validation cycle: {subject}", actor=actor)
            parent_id = self._append_stage(trace_id, "requirements", f"{len(requirements)} requirement(s) declared for '{subject}'.", parent_id, actor, {"requirement_names": [r.name for r in requirements]})
            parent_id = self._append_stage(trace_id, "design", f"Design subject: '{request.subject}'.", parent_id, actor, {"subject": request.subject})

        interpretation = await self.run_simulation_async(request, requirements=requirements, actor=actor)

        if self._digital_thread is not None:
            parent_id = self._append_stage(trace_id, "simulation", f"Simulation '{request.id}' completed via '{interpretation.run_result.connector_name}'.", parent_id, actor, {"outputs": {o.name: o.value for o in interpretation.run_result.outputs}})
            parent_id = self._append_stage(trace_id, "interpretation", f"{len(interpretation.margins)} margin(s) computed.", parent_id, actor, {"margins": {m.requirement_name: m.margin for m in interpretation.margins}})
            parent_id = self._append_stage(trace_id, "verification", f"Verdict: {interpretation.verdict}.", parent_id, actor, {"verdict": interpretation.verdict})

        adjustment_claims = tuple(claim for claim in interpretation.claims if claim.claim_type == ClaimType.RECOMMENDATION.name)

        if self._digital_thread is not None:
            self._append_stage(
                trace_id, "adjustment_recommendation",
                f"{len(adjustment_claims)} adjustment recommendation(s)." if adjustment_claims else "No adjustments recommended.",
                parent_id, actor, {"recommendations": [claim.text for claim in adjustment_claims]},
            )

        return ValidationCycleResult(subject=subject, trace_id=trace_id, interpretation=interpretation, adjustment_recommendations=adjustment_claims)

    def _append_stage(
        self, trace_id: Optional[str], stage_name: str, description: str, parent_entry_id: Optional[str],
        actor: Optional[str], data: dict[str, Any],
    ) -> Optional[str]:
        if self._digital_thread is None or trace_id is None:
            return parent_entry_id
        try:
            entry = self._digital_thread.record_action(
                trace_id=trace_id, description=description, inputs=data,
                method=f"SimulationOrchestrator.{stage_name}", parameters={}, result=data,
                actor=actor, parent_entry_id=parent_entry_id,
            )
            return entry.id
        except Exception:
            logger.exception("Failed to record validation-cycle stage '%s' in DigitalThread.", stage_name)
            return parent_entry_id

    # ------------------------------------------------------------------
    # DigitalThread / KnowledgeGraph integration
    # ------------------------------------------------------------------

    def _record_to_digital_thread(
        self, request: SimulationRequest, run_result: SimulationRunResult, margins: Sequence[MarginResult],
        verification_report: Optional[VerificationReport], actor: Optional[str],
    ) -> Optional[str]:
        if self._digital_thread is None:
            return None
        try:
            trace_id = self._digital_thread.begin_trace(f"Simulation: {request.id} ({request.simulation_type})", actor=actor)
            self._digital_thread.record_decision(
                trace_id=trace_id, description=f"Ran '{request.simulation_type}' simulation on '{request.subject}'.",
                inputs={
                    "parameters": {p.name: p.value for p in request.parameters},
                    "assumptions": list(request.assumptions),
                },
                method=f"SimulationOrchestrator via {run_result.connector_name}",
                parameters={"connector": run_result.connector_name, "converged": run_result.converged},
                validation={"verdict": verification_report.verdict} if verification_report else None,
                result={
                    "outputs": {o.name: o.value for o in run_result.outputs},
                    "margins": {m.requirement_name: m.margin for m in margins},
                },
                actor=actor,
            )
            return trace_id
        except Exception:
            logger.exception("Failed to record simulation in DigitalThread.")
            return None

    def _mirror_to_knowledge_graph(self, interpretation: SimulationInterpretation) -> None:
        if self._knowledge_graph is None:
            return
        request = interpretation.request
        content = (
            f"Simulation '{request.id}' ({request.simulation_type}) on '{request.subject}': "
            f"verdict={interpretation.verdict}, {len(interpretation.margins)} margin(s) checked."
        )
        metadata = {
            "request_id": request.id, "simulation_type": request.simulation_type, "subject": request.subject,
            "verdict": interpretation.verdict,
            "margins": {m.requirement_name: m.margin for m in interpretation.margins},
        }
        try:
            self._knowledge_graph.remember(content, node_type="simulation_result", metadata=metadata, source="simulation_orchestrator")
        except Exception:
            logger.exception("Failed to record simulation interpretation in KnowledgeGraph.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "connectors": {name: connector.simulation_type for name, connector in self._connectors.items()},
            "supported_types": sorted(self._connectors_by_type.keys()),
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "digital_thread_bound": self._digital_thread is not None,
            "verification_engine_bound": self._verification_engine is not None,
            "cognitive_engine_bound": self._cognitive_engine is not None,
            "autonomous_loop_bound": self._autonomous_loop is not None,
        }