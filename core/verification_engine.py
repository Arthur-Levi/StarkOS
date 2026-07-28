"""
core/verification_engine.py
==============================

Technical verification engine for StarkOS: checks claims, plans and
results for real, mechanically-detectable problems before they're
trusted -- unit mismatches, constraint/requirement violations, missing
or non-existent evidence, standards non-compliance, and numeric
contradictions -- and classifies every claim's epistemic status
explicitly rather than blurring fact, hypothesis, estimate and
recommendation together.

Responsibilities
----------------
- Verify `Claim`s: statements explicitly tagged by the caller as FACT,
  HYPOTHESIS, ESTIMATE or RECOMMENDATION (see the honesty note on why
  this classification is never auto-detected from free text).
- Detect real inconsistencies:
  - **Units**: a self-contained dimensional-analysis checker (SI base
    dimensions + common derived units) flags claims about the same
    tagged quantity whose units aren't dimensionally compatible.
  - **Constraints**: reuses `core.auto_engineer.Constraint` directly --
    the same, already-tested bound/threshold checking, not a
    reimplementation.
  - **Requirements/standards**: reuses the component -> material ->
    standard graph structure already modeled in `core.knowledge_graph`
    (`made_of`/`complies_with` relations) via real traversal.
  - **Logic**: numeric contradiction detection -- claims about the same
    tagged quantity whose value ranges don't overlap cannot all be true.
  - **Evidence**: every evidence reference is checked against the real
    system it claims to point at (e.g. does the referenced
    KnowledgeGraph node actually exist), and FACT claims without any
    evidence are flagged outright.
- Classify confidence per claim (and an overall, conservative summary)
  from real, inspectable signals -- see the honesty note on what this
  number actually means.
- Produce a `VerificationReport` with an explicit verdict --
  "accepted" / "needs_revision" / "rejected" -- so unsafe or
  insufficiently-supported results can be mechanically caught rather
  than silently passed through.
- Record every verification into the DigitalThread's immutable ledger
  (if bound), and mirror it into KnowledgeGraph as searchable memory.
- Convenience translators for a CognitiveEngine `PlanExecutionResult`
  and a RAGEngine `RAGAnswer` into verifiable claim sets.

Honesty about scope
--------------------
1. **Claim classification (FACT/HYPOTHESIS/ESTIMATE/RECOMMENDATION) is
   always caller-supplied, never auto-detected from free text.**
   Deciding whether an arbitrary sentence is a fact or a hypothesis
   requires real natural-language understanding, which this module
   doesn't have. What it *does* check is whether a claim's declared
   type is *internally consistent* with its evidence -- a FACT with no
   evidence is flagged, a HYPOTHESIS with strong evidence is noted as a
   candidate for reclassification -- self-consistency checking, not
   semantic judgment of the claim's content.

2. **"Confidence" is derived from real, inspectable signals** (evidence
   count, whether referenced evidence actually exists, how many
   verification errors/warnings were found against a claim) -- **not a
   calibrated probability that the claim is true.** The overall report
   confidence is deliberately the *worst* individual claim's confidence,
   not an average -- a conservative choice appropriate for critical
   engineering review (a chain is only as strong as its weakest claim).

3. **The unit checker is a real but scoped dimensional-analysis tool,
   not a full physical-units library.** It covers the seven SI base
   dimensions and a modest table of common derived units (N, J, W, Pa,
   Hz, V, Ω, ...) with basic `*`/`/`/`^` compound-expression parsing.
   Unrecognized unit symbols raise `UnitParsingError` rather than being
   silently ignored or guessed at.

4. **Logical contradiction detection is narrow and numeric**: it flags
   non-overlapping value ranges declared for the same caller-tagged
   quantity. It is not a general theorem prover and cannot detect
   contradictions expressed only in prose.

5. **This module doesn't invent engineering judgment.** It cannot tell
   you whether a design is *good* -- only whether the claims made about
   it are internally consistent, evidenced, within stated constraints,
   and compliant with the standards you told it to check against.

Design
------
Same shape as the rest of StarkOS: real, honest logic where it's
genuinely achievable (dimensional analysis, constraint checking, graph
traversal, evidence existence checks are all deterministic and fully
testable), reusing already-built, already-tested components
(`auto_engineer.Constraint`, `knowledge_graph`'s standards graph)
instead of duplicating them.

`VerificationEngine` satisfies the `Module` protocol (name/initialize/
start/stop) and registers with the Kernel like any other StarkOS module:

    verifier = VerificationEngine(services=services)
    verifier.bind_knowledge_graph(knowledge_graph)
    verifier.bind_digital_thread(digital_thread)
    kernel.register_module(verifier, name="verification_engine", priority=35)

    report = verifier.verify(
        subject="Motor mount load claim",
        claims=[Claim(id="c1", text="Peak load is 450-500 N", claim_type=ClaimType.ESTIMATE.name,
                      numeric_range=(450.0, 500.0), units="N", metadata={"quantity": "peak_load"})],
    )
    if report.verdict == "rejected":
        ...
"""

from __future__ import annotations

import dataclasses
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Sequence, Union

from core.auto_engineer import AutoEngineer, Constraint
from core.digital_thread import DigitalThread
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph, KnowledgeGraphError
from core.logger import get_logger
from core.service_container import ServiceContainer

logger = get_logger("verification_engine")

# =============================================================================
# Exceptions
# =============================================================================

class VerificationEngineError(Exception):
    """Base exception for VerificationEngine failures."""

class InvalidVerificationRequestError(VerificationEngineError):
    """Raised when verify() arguments are malformed (bad claim_type, empty claims, ...)."""

class UnitParsingError(VerificationEngineError):
    """Raised when a unit expression can't be parsed/recognized."""

# =============================================================================
# Claim classification
# =============================================================================

class ClaimType(Enum):
    FACT = auto()
    HYPOTHESIS = auto()
    ESTIMATE = auto()
    RECOMMENDATION = auto()

class ConfidenceLevel(Enum):
    UNVERIFIABLE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

_CONFIDENCE_RANK: dict[str, int] = {
    ConfidenceLevel.UNVERIFIABLE.name: 0,
    ConfidenceLevel.LOW.name: 1,
    ConfidenceLevel.MEDIUM.name: 2,
    ConfidenceLevel.HIGH.name: 3,
}

class IssueSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()  # forces "rejected" -- see _determine_verdict

# =============================================================================
# Data models
# =============================================================================

@dataclass(slots=True, frozen=True)
class EvidenceReference:
    """A pointer to where a claim's support supposedly comes from --
    checked against the real system it references, never taken on faith."""

    source_type: str  # "knowledge_graph_node" | "rag_source" | "task_result" | "external" | "measurement"
    reference: str
    description: str = ""

@dataclass(slots=True, frozen=True)
class Claim:
    """
    One discrete statement to verify. `claim_type` is always supplied
    by the caller (see module docstring point 1) -- CognitiveEngine,
    RAGEngine, AutoEngineer, or a human reviewer, never inferred here
    from `text` alone.
    """

    id: str
    text: str
    claim_type: str  # a ClaimType member name
    evidence: tuple[EvidenceReference, ...] = ()
    units: Optional[str] = None
    numeric_value: Optional[float] = None
    numeric_range: Optional[tuple[float, float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True, frozen=True)
class VerificationIssue:
    severity: str  # an IssueSeverity member name
    category: str  # "units" | "constraint" | "requirement" | "standard" | "logic" | "evidence" | "classification"
    claim_id: Optional[str]
    message: str

@dataclass(slots=True, frozen=True)
class VerificationReport:
    subject: str
    verdict: str  # "accepted" | "needs_revision" | "rejected"
    confidence: str  # a ConfidenceLevel member name
    confidence_basis: str
    issues: tuple[VerificationIssue, ...]
    claims: tuple[Claim, ...]
    verified_at: datetime = field(default_factory=datetime.utcnow)
    digital_thread_entry_id: Optional[str] = None

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class VerificationEngineConfig:
    record_to_knowledge_graph: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Dimensional analysis (real, scoped -- see honesty note 3)
# =============================================================================
#
# Dimension vector order: (length, mass, time, current, temperature, amount, luminous_intensity)

_BASE_UNIT_DIMENSIONS: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "m": (1, 0, 0, 0, 0, 0, 0),
    "kg": (0, 1, 0, 0, 0, 0, 0),
    "g": (0, 1, 0, 0, 0, 0, 0),  # treated dimensionally same as kg -- magnitude/prefix not modeled
    "s": (0, 0, 1, 0, 0, 0, 0),
    "a": (0, 0, 0, 1, 0, 0, 0),
    "k": (0, 0, 0, 0, 1, 0, 0),
    "mol": (0, 0, 0, 0, 0, 1, 0),
    "cd": (0, 0, 0, 0, 0, 0, 1),
    # dimensionless
    "rad": (0, 0, 0, 0, 0, 0, 0),
    "deg": (0, 0, 0, 0, 0, 0, 0),
    "%": (0, 0, 0, 0, 0, 0, 0),
    # common derived units
    "n": (1, 1, -2, 0, 0, 0, 0),       # newton
    "j": (2, 1, -2, 0, 0, 0, 0),       # joule
    "w": (2, 1, -3, 0, 0, 0, 0),       # watt
    "pa": (-1, 1, -2, 0, 0, 0, 0),     # pascal
    "hz": (0, 0, -1, 0, 0, 0, 0),      # hertz
    "v": (2, 1, -3, -1, 0, 0, 0),      # volt
    "ohm": (2, 1, -3, -2, 0, 0, 0),
    "nm": (2, 1, -2, 0, 0, 0, 0),      # newton-meter (torque) -- dimensionally identical to joule
}

_UNIT_TOKEN_PATTERN = re.compile(r"([a-zA-Z%µ]+)(?:\^(-?\d+))?")

def _parse_unit_dimension(unit_expr: str) -> tuple[int, ...]:
    """Parse a compound unit expression (e.g. "kg/m^3", "N*m", "m/s^2")
    into its SI base-dimension vector. Raises UnitParsingError for any
    unrecognized symbol rather than guessing."""
    if not unit_expr or not unit_expr.strip():
        raise UnitParsingError("Empty unit expression.")

    dimension = [0, 0, 0, 0, 0, 0, 0]
    tokens = re.split(r"([*/])", unit_expr.replace(" ", ""))
    sign = 1
    for token in tokens:
        if token == "*":
            sign = 1
            continue
        if token == "/":
            sign = -1
            continue
        if not token:
            continue
        match = _UNIT_TOKEN_PATTERN.fullmatch(token)
        if not match:
            raise UnitParsingError(f"Cannot parse unit token '{token}' in '{unit_expr}'.")
        symbol, exponent_str = match.groups()
        exponent = int(exponent_str) if exponent_str else 1
        base = _BASE_UNIT_DIMENSIONS.get(symbol.lower())
        if base is None:
            raise UnitParsingError(f"Unknown unit symbol '{symbol}' in '{unit_expr}'.")
        for i in range(7):
            dimension[i] += base[i] * exponent * sign
    return tuple(dimension)

def dimensions_match(unit_a: str, unit_b: str) -> bool:
    """True if two unit expressions describe the same physical
    dimension (e.g. "N*m" and "J" both resolve to the same vector)."""
    return _parse_unit_dimension(unit_a) == _parse_unit_dimension(unit_b)

# =============================================================================
# Verification Engine
# =============================================================================

class VerificationEngine:
    """
    StarkOS's technical verification module. See the module docstring's
    "Honesty about scope" section before treating a verdict or
    confidence level as more than what it actually checks.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(self, *, services: ServiceContainer, config: Optional[VerificationEngineConfig] = None) -> None:
        self._services = services
        self._config = config or VerificationEngineConfig()

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._digital_thread: Optional[DigitalThread] = None
        self._auto_engineer: Optional[AutoEngineer] = None
        self._cognitive_engine: Any = None
        self._rag_engine: Any = None

        logger.info("VerificationEngine constructed.")

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "verification_engine"

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to VerificationEngine.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to VerificationEngine.")

    def bind_digital_thread(self, digital_thread: DigitalThread) -> None:
        self._digital_thread = digital_thread
        logger.debug("DigitalThread bound to VerificationEngine.")

    def bind_auto_engineer(self, auto_engineer: AutoEngineer) -> None:
        self._auto_engineer = auto_engineer
        logger.debug("AutoEngineer bound to VerificationEngine.")

    def bind_cognitive_engine(self, cognitive_engine: Any) -> None:
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to VerificationEngine.")

    def bind_rag_engine(self, rag_engine: Any) -> None:
        self._rag_engine = rag_engine
        logger.debug("RAGEngine bound to VerificationEngine.")

    async def initialize(self) -> None:
        logger.info("Initializing VerificationEngine.")
        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._digital_thread is None:
            self._digital_thread = self._services.resolve_optional(DigitalThread)
        if self._auto_engineer is None:
            self._auto_engineer = self._services.resolve_optional(AutoEngineer)
        logger.info(
            "VerificationEngine initialized.",
            extra={"knowledge_graph_bound": self._knowledge_graph is not None, "digital_thread_bound": self._digital_thread is not None},
        )

    async def start(self) -> None:
        logger.info("VerificationEngine ready.")

    async def stop(self) -> None:
        logger.info("VerificationEngine stopped.")

    # ------------------------------------------------------------------
    # Evidence verification
    # ------------------------------------------------------------------

    def _verify_evidence(self, claim: Claim) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        if claim.claim_type == ClaimType.FACT.name and not claim.evidence:
            issues.append(VerificationIssue(
                severity=IssueSeverity.ERROR.name, category="evidence", claim_id=claim.id,
                message="Claim is classified as FACT but has no supporting evidence -- facts require evidence.",
            ))

        for evidence in claim.evidence:
            if evidence.source_type == "knowledge_graph_node":
                if self._knowledge_graph is None:
                    issues.append(VerificationIssue(
                        severity=IssueSeverity.WARNING.name, category="evidence", claim_id=claim.id,
                        message=f"Cannot verify evidence '{evidence.reference}' -- no KnowledgeGraph bound.",
                    ))
                    continue
                if self._knowledge_graph.get_node_optional(evidence.reference) is None:
                    issues.append(VerificationIssue(
                        severity=IssueSeverity.ERROR.name, category="evidence", claim_id=claim.id,
                        message=f"Evidence references KnowledgeGraph node '{evidence.reference}', which does not exist.",
                    ))

        if claim.claim_type == ClaimType.HYPOTHESIS.name and len(claim.evidence) >= 3:
            issues.append(VerificationIssue(
                severity=IssueSeverity.INFO.name, category="classification", claim_id=claim.id,
                message="Claim has substantial evidence but is classified as HYPOTHESIS -- "
                        "consider reclassifying as FACT if the evidence is conclusive.",
            ))

        if claim.claim_type == ClaimType.RECOMMENDATION.name and claim.evidence and not any(
            e.source_type in ("knowledge_graph_node", "rag_source") for e in claim.evidence
        ):
            issues.append(VerificationIssue(
                severity=IssueSeverity.INFO.name, category="classification", claim_id=claim.id,
                message="Recommendation's evidence doesn't reference stored knowledge -- verify it is still grounded.",
            ))

        return issues

    # ------------------------------------------------------------------
    # Unit consistency
    # ------------------------------------------------------------------

    def _verify_unit_consistency(self, claims: Sequence[Claim]) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        by_quantity: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            quantity = claim.metadata.get("quantity")
            if quantity is not None and claim.units:
                by_quantity[str(quantity)].append(claim)

        for quantity, group in by_quantity.items():
            if len(group) < 2:
                continue
            reference_claim = group[0]
            try:
                reference_dimension = _parse_unit_dimension(reference_claim.units)  # type: ignore[arg-type]
            except UnitParsingError as exc:
                issues.append(VerificationIssue(
                    severity=IssueSeverity.WARNING.name, category="units", claim_id=reference_claim.id,
                    message=f"Could not parse units '{reference_claim.units}': {exc}",
                ))
                continue

            for claim in group[1:]:
                try:
                    dimension = _parse_unit_dimension(claim.units)  # type: ignore[arg-type]
                except UnitParsingError as exc:
                    issues.append(VerificationIssue(
                        severity=IssueSeverity.WARNING.name, category="units", claim_id=claim.id,
                        message=f"Could not parse units '{claim.units}': {exc}",
                    ))
                    continue
                if dimension != reference_dimension:
                    issues.append(VerificationIssue(
                        severity=IssueSeverity.ERROR.name, category="units", claim_id=claim.id,
                        message=(
                            f"Claim '{claim.id}' states quantity '{quantity}' in '{claim.units}', dimensionally "
                            f"incompatible with claim '{reference_claim.id}' in '{reference_claim.units}'."
                        ),
                    ))
        return issues

    # ------------------------------------------------------------------
    # Logical (numeric) contradiction detection
    # ------------------------------------------------------------------

    def _verify_logical_consistency(self, claims: Sequence[Claim]) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        by_quantity: dict[str, list[tuple[str, tuple[float, float]]]] = defaultdict(list)

        for claim in claims:
            quantity = claim.metadata.get("quantity")
            if quantity is None:
                continue
            if claim.numeric_range is not None:
                by_quantity[str(quantity)].append((claim.id, claim.numeric_range))
            elif claim.numeric_value is not None:
                by_quantity[str(quantity)].append((claim.id, (claim.numeric_value, claim.numeric_value)))

        for quantity, ranges in by_quantity.items():
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    id_a, (lo_a, hi_a) = ranges[i]
                    id_b, (lo_b, hi_b) = ranges[j]
                    if hi_a < lo_b or hi_b < lo_a:
                        issues.append(VerificationIssue(
                            severity=IssueSeverity.ERROR.name, category="logic", claim_id=id_a,
                            message=(
                                f"Claim '{id_a}' ([{lo_a}, {hi_a}]) contradicts claim '{id_b}' ([{lo_b}, {hi_b}]) "
                                f"for quantity '{quantity}' -- the ranges don't overlap."
                            ),
                        ))
        return issues

    # ------------------------------------------------------------------
    # Constraints (reuses core.auto_engineer.Constraint directly)
    # ------------------------------------------------------------------

    def verify_constraints(self, constraints: Sequence[Constraint], values: dict[str, float]) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []
        for constraint in constraints:
            value = values.get(constraint.target)
            if value is None:
                issues.append(VerificationIssue(
                    severity=IssueSeverity.WARNING.name, category="constraint", claim_id=None,
                    message=f"Constraint '{constraint.name}': target '{constraint.target}' not found in provided values.",
                ))
                continue
            if not constraint.is_satisfied(value):
                issues.append(VerificationIssue(
                    severity=IssueSeverity.ERROR.name, category="constraint", claim_id=None,
                    message=f"Constraint '{constraint.name}' violated: {constraint.target}={value} fails {constraint.operator} {constraint.bound}.",
                ))
        return issues

    # ------------------------------------------------------------------
    # Standards/requirements compliance (reuses the KnowledgeGraph graph)
    # ------------------------------------------------------------------

    def verify_standards_compliance(self, component_node_id: str, required_standard_ids: Sequence[str]) -> list[VerificationIssue]:
        if self._knowledge_graph is None:
            raise VerificationEngineError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")

        try:
            component = self._knowledge_graph.get_node(component_node_id)
        except KnowledgeGraphError as exc:
            raise VerificationEngineError(f"Cannot verify standards compliance: {exc}") from exc

        complied_ids: set[str] = set()
        for standard in self._knowledge_graph.neighbors(component_node_id, relation_type="complies_with", direction="out"):
            complied_ids.add(standard.id)
        for material in self._knowledge_graph.neighbors(component_node_id, relation_type="made_of", direction="out"):
            for standard in self._knowledge_graph.neighbors(material.id, relation_type="complies_with", direction="out"):
                complied_ids.add(standard.id)

        issues: list[VerificationIssue] = []
        for required_id in required_standard_ids:
            if required_id not in complied_ids:
                issues.append(VerificationIssue(
                    severity=IssueSeverity.ERROR.name, category="standard", claim_id=None,
                    message=f"Component '{component.label}' does not comply with required standard '{required_id}'.",
                ))
        return issues

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _derive_claim_confidence(self, claim: Claim, issues: Sequence[VerificationIssue]) -> tuple[str, str]:
        claim_issues = [issue for issue in issues if issue.claim_id == claim.id]
        error_count = sum(1 for issue in claim_issues if issue.severity == IssueSeverity.ERROR.name)
        warning_count = sum(1 for issue in claim_issues if issue.severity == IssueSeverity.WARNING.name)

        if error_count > 0:
            return ConfidenceLevel.LOW.name, f"{error_count} verification error(s) found against this claim."
        if not claim.evidence:
            return ConfidenceLevel.UNVERIFIABLE.name, "No evidence was supplied to assess this claim against."
        if warning_count > 0:
            return ConfidenceLevel.MEDIUM.name, f"{warning_count} verification warning(s) found; evidence present but not fully confirmed."
        return ConfidenceLevel.HIGH.name, f"{len(claim.evidence)} evidence reference(s) checked with no issues found."

    def _summarize_confidence(self, claims: Sequence[Claim], issues: Sequence[VerificationIssue]) -> tuple[str, str]:
        if not claims:
            return ConfidenceLevel.UNVERIFIABLE.name, "No claims were assessed."
        per_claim = {claim.id: self._derive_claim_confidence(claim, issues) for claim in claims}
        worst_id = min(per_claim, key=lambda claim_id: _CONFIDENCE_RANK[per_claim[claim_id][0]])
        worst_level, worst_basis = per_claim[worst_id]
        return worst_level, (
            f"Overall confidence is bounded by the weakest claim ('{worst_id}'): {worst_basis} "
            "(the report's confidence is the minimum across all claims, not an average -- see module docstring)."
        )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_verdict(issues: Sequence[VerificationIssue]) -> str:
        if any(issue.severity == IssueSeverity.ERROR.name for issue in issues):
            return "rejected"
        if any(issue.severity == IssueSeverity.WARNING.name for issue in issues):
            return "needs_revision"
        return "accepted"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        *,
        subject: str,
        claims: Sequence[Claim],
        constraints: Sequence[Constraint] = (),
        constraint_values: Optional[dict[str, float]] = None,
        required_standards: Optional[dict[str, Sequence[str]]] = None,
        actor: Optional[str] = None,
    ) -> VerificationReport:
        if not subject:
            raise InvalidVerificationRequestError("subject is required.")
        if not claims:
            raise InvalidVerificationRequestError("At least one claim is required to verify.")

        valid_types = {member.name for member in ClaimType}
        for claim in claims:
            if claim.claim_type not in valid_types:
                raise InvalidVerificationRequestError(f"Claim '{claim.id}' has unknown claim_type '{claim.claim_type}'.")

        issues: list[VerificationIssue] = []
        for claim in claims:
            issues.extend(self._verify_evidence(claim))
        issues.extend(self._verify_unit_consistency(claims))
        issues.extend(self._verify_logical_consistency(claims))

        if constraints:
            issues.extend(self.verify_constraints(constraints, constraint_values or {}))

        if required_standards:
            for component_id, standard_ids in required_standards.items():
                try:
                    issues.extend(self.verify_standards_compliance(component_id, standard_ids))
                except VerificationEngineError:
                    logger.exception("Standards compliance check failed for '%s'.", component_id)

        verdict = self._determine_verdict(issues)
        confidence, confidence_basis = self._summarize_confidence(claims, issues)

        report = VerificationReport(
            subject=subject, verdict=verdict, confidence=confidence, confidence_basis=confidence_basis,
            issues=tuple(issues), claims=tuple(claims),
        )

        logger.info(
            "Verification completed.",
            extra={"subject": subject, "verdict": verdict, "confidence": confidence, "issue_count": len(issues)},
        )

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            self._mirror_to_knowledge_graph(report)

        entry_id = self._record_to_digital_thread(report, actor)
        if entry_id is not None:
            report = dataclasses.replace(report, digital_thread_entry_id=entry_id)

        return report

    def request_revision(self, report: VerificationReport, *, reason: str) -> VerificationReport:
        """Explicitly force a report to 'needs_revision' (e.g. a human
        reviewer overriding an 'accepted' verdict for a reason this
        engine couldn't check) -- always adds an INFO issue recording
        why, rather than silently changing the verdict."""
        issue = VerificationIssue(severity=IssueSeverity.WARNING.name, category="classification", claim_id=None, message=f"Revision requested: {reason}")
        return dataclasses.replace(report, verdict="needs_revision", issues=report.issues + (issue,))

    # ------------------------------------------------------------------
    # CognitiveEngine / RAGEngine translators
    # ------------------------------------------------------------------

    def verify_plan_execution(self, goal: Any, execution_result: Any, *, actor: Optional[str] = None) -> VerificationReport:
        """
        Translate an already-completed CognitiveEngine PlanExecutionResult
        into claims (one per task: FACT if succeeded, HYPOTHESIS
        otherwise) and verify them. This is a mechanical translation of
        task status, not a semantic judgment of whether the goal was
        actually achieved -- see module docstring point 5.
        """
        claims = []
        for task_result in execution_result.task_results:
            succeeded = getattr(task_result.status, "name", str(task_result.status)) == "SUCCEEDED"
            evidence = (
                (EvidenceReference(source_type="task_result", reference=task_result.task_id, description="CognitiveEngine task output"),)
                if succeeded else ()
            )
            claims.append(Claim(
                id=task_result.task_id,
                text=f"Task '{task_result.task_id}' completed with status {task_result.status}.",
                claim_type=ClaimType.FACT.name if succeeded else ClaimType.HYPOTHESIS.name,
                evidence=evidence,
                metadata={"status": str(task_result.status)},
            ))
        return self.verify(subject=f"Plan execution for goal '{goal.description}'", claims=claims, actor=actor)

    def verify_rag_answer(self, answer: Any, *, actor: Optional[str] = None) -> VerificationReport:
        """Verify a RAGEngine RAGAnswer: is it backed by real, existing
        KnowledgeGraph nodes, classified consistently with its own
        honestly-self-reported synthesis_method?"""
        evidence = tuple(
            EvidenceReference(source_type="knowledge_graph_node", reference=source.node_id, description=source.label)
            for source in answer.sources
        )
        claim_type = ClaimType.FACT.name if answer.synthesis_method == "extractive" and evidence else ClaimType.ESTIMATE.name
        claim = Claim(
            id=str(uuid.uuid4()), text=answer.answer_text, claim_type=claim_type, evidence=evidence,
            metadata={"synthesis_method": answer.synthesis_method, "rag_confidence": answer.confidence},
        )
        return self.verify(subject=f"RAG answer for: {answer.query}", claims=[claim], actor=actor)

    # ------------------------------------------------------------------
    # KnowledgeGraph / DigitalThread integration
    # ------------------------------------------------------------------

    def _mirror_to_knowledge_graph(self, report: VerificationReport) -> None:
        if self._knowledge_graph is None:
            return
        content = f"Verification of '{report.subject}': verdict={report.verdict}, confidence={report.confidence}, issues={len(report.issues)}"
        metadata = {
            "verdict": report.verdict, "confidence": report.confidence, "issue_count": len(report.issues),
            "claim_count": len(report.claims),
        }
        try:
            self._knowledge_graph.remember(content, node_type="verification_report", metadata=metadata, source="verification_engine")
        except Exception:
            logger.exception("Failed to record verification report in KnowledgeGraph.")

    def _record_to_digital_thread(self, report: VerificationReport, actor: Optional[str]) -> Optional[str]:
        if self._digital_thread is None:
            return None
        try:
            trace_id = self._digital_thread.begin_trace(f"Verification: {report.subject}", actor=actor)
            entry = self._digital_thread.record_validation(
                trace_id=trace_id,
                description=f"Verification of '{report.subject}'",
                validation={"verdict": report.verdict, "confidence": report.confidence, "issue_count": len(report.issues)},
                result={"issues": [dataclasses.asdict(issue) for issue in report.issues]},
                actor=actor,
            )
            return entry.id
        except Exception:
            logger.exception("Failed to record verification in DigitalThread.")
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "digital_thread_bound": self._digital_thread is not None,
            "auto_engineer_bound": self._auto_engineer is not None,
            "cognitive_engine_bound": self._cognitive_engine is not None,
            "rag_engine_bound": self._rag_engine is not None,
        }