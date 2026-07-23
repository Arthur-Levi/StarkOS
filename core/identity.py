"""
core/identity.py
===============

Identity and personality subsystem for StarkOS.

Responsibilities
----------------
* Define the runtime identity.
* Manage personality traits.
* Enforce security clearance.
* Provide contextual behavior for user interaction.
"""

from __future__ import annotations

import logging
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import Any, Sequence

from core.logger import get_logger
from core.service_container import ServiceContainer

logger = get_logger("identity")

# Light, honest heuristics for varying tone -- not natural-language
# understanding. A greeting/question is recognized by shape, not meaning.
_GREETING_PATTERN = re.compile(r"^\s*(oi|ol[aá]|hello|hi|hey|e\s*a[ií])\b", re.IGNORECASE)
_QUESTION_PATTERN = re.compile(r"\?\s*$")

# =============================================================================
# Exceptions
# =============================================================================

class IdentityError(Exception):
    """Base exception for Identity."""

class ClearanceError(IdentityError):
    """Raised when an operation exceeds the user's clearance."""

class PersonalityError(IdentityError):
    """Raised when personality configuration is invalid."""

# =============================================================================
# Clearance Levels
# =============================================================================

class ClearanceLevel(IntEnum):
    GUEST = 10
    USER = 20
    OPERATOR = 40
    ADMIN = 70
    FOUNDER = 100

# =============================================================================
# Personality Traits
# =============================================================================

class PersonalityTrait(Enum):
    PROFESSIONAL = auto()
    ELEGANT = auto()
    LOYAL = auto()
    PROACTIVE = auto()
    SARCASTIC = auto()
    CALM = auto()
    ANALYTICAL = auto()

# =============================================================================
# Runtime Mode
# =============================================================================

class IdentityMode(Enum):
    NORMAL = "normal"
    ASSISTANT = "assistant"
    DIAGNOSTIC = "diagnostic"
    SECURITY = "security"

# =============================================================================
# Persona
# =============================================================================

@dataclass(slots=True, frozen=True)
class Persona:
    name: str
    version: str
    description: str
    traits: tuple[PersonalityTrait, ...]
    humor_level: float = 0.30
    sarcasm_level: float = 0.15
    proactivity: float = 0.90
    formality: float = 0.85

# =============================================================================
# Founder Context
# =============================================================================

@dataclass(slots=True)
class FounderContext:
    identifier: str = "founder"
    display_name: str = "Founder"
    clearance: ClearanceLevel = ClearanceLevel.FOUNDER
    authenticated: bool = False
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connected_since: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Identity Snapshot
# =============================================================================

@dataclass(slots=True, frozen=True)
class IdentitySnapshot:
    mode: IdentityMode
    persona: Persona
    clearance: ClearanceLevel
    authenticated: bool
    session_id: str

# =============================================================================
# Conversation Models
# =============================================================================

@dataclass(slots=True, frozen=True)
class ConversationTurn:
    timestamp: datetime
    speaker: str
    message: str

@dataclass(slots=True, frozen=True)
class IdentityResponse:
    text: str
    suggestions: tuple[str, ...]
    mood: str = "neutral"
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Identity
# =============================================================================

class Identity:
    """
    StarkOS runtime identity.

    This service centralizes personality,
    security clearance and contextual behavior.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
    ) -> None:
        self._services = services
        self._mode = IdentityMode.NORMAL
        self._persona = self._build_default_persona()
        self._founder = FounderContext()
        self._history: list[ConversationTurn] = []
        self._kernel = None
        self._event_bus = None

        logger.info("Identity initialized.")

    def _build_default_persona(self) -> Persona:
        return Persona(
            name="STARK",
            version="0.4",
            description="Elegant, analytical and loyal digital assistant.",
            traits=(
                PersonalityTrait.ELEGANT,
                PersonalityTrait.PROACTIVE,
                PersonalityTrait.LOYAL,
                PersonalityTrait.ANALYTICAL,
                PersonalityTrait.PROFESSIONAL,
            ),
        )

    @property
    def persona(self) -> Persona:
        return self._persona

    @property
    def clearance(self) -> ClearanceLevel:
        return self._founder.clearance

    @property
    def mode(self) -> IdentityMode:
        return self._mode

    def snapshot(self) -> IdentitySnapshot:
        return IdentitySnapshot(
            mode=self._mode,
            persona=self._persona,
            clearance=self.clearance,
            authenticated=self._founder.authenticated,
            session_id=self._founder.session_id,
        )

    def has_clearance(self, required: ClearanceLevel) -> bool:
        return self.clearance >= required

    def require_clearance(self, required: ClearanceLevel) -> None:
        if not self.has_clearance(required):
            logger.warning("Clearance denied.")
            raise ClearanceError(f"Required clearance: {required.name}")

    def remember(self, speaker: str, message: str) -> None:
        self._history.append(
            ConversationTurn(
                timestamp=datetime.utcnow(),
                speaker=speaker,
                message=message,
            )
        )

    def session_history(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._history)

    def respond(self, message: str) -> IdentityResponse:
        """
        Compose a reply. Tone varies by the *shape* of the message
        (greeting-like, question-like, or a plain statement/request) via
        the light regex heuristics above -- this is elegant phrasing
        variation, not natural-language understanding. For anything
        requiring real comprehension, StarkOS routes through
        CognitiveEngine/KnowledgeGraph instead; Identity is the voice,
        not the reasoning.
        """
        logger.info("Generating contextual response.")
        self.remember("user", message)

        text = self._compose_response_text(message)
        response = IdentityResponse(
            text=text,
            suggestions=self.proactive_suggestions(),
            mood="professional",
            metadata={
                "history": len(self._history),
                "mode": self.mode.value,
            },
        )

        self.remember(self.persona.name, response.text)
        return response

    def _compose_response_text(self, message: str) -> str:
        stripped = message.strip()
        if not stripped:
            return "I'm listening whenever you're ready."

        if _GREETING_PATTERN.match(stripped):
            return self._greeting_text()

        if _QUESTION_PATTERN.search(stripped):
            return (
                f"A fair question. Let me weigh '{stripped}' against the current "
                "runtime context before I commit to an answer."
            )

        return (
            f"Understood. I'll treat '{stripped}' as the priority and keep you "
            "posted as it's handled."
        )

    def _greeting_text(self) -> str:
        hour = datetime.now().hour
        if hour < 12:
            salutation = "Good morning"
        elif hour < 18:
            salutation = "Good afternoon"
        else:
            salutation = "Good evening"

        if self._founder.authenticated:
            return f"{salutation}, {self._founder.display_name}. Everything is exactly as you left it."
        return f"{salutation}. Systems are green and standing by."

    def proactive_suggestions(self) -> tuple[str, ...]:
        """
        Suggestions tailored to the bound Kernel's actual state where
        possible -- e.g. "start it" when stopped, "check health" when
        running -- rather than a fixed list regardless of context.
        Falls back to generic suggestions if no Kernel is bound yet.
        """
        if self._kernel is None:
            return (
                "Bind me to the Kernel so I can watch over the whole system.",
                "Review what modules are registered.",
            )

        state_name = getattr(getattr(self._kernel, "state", None), "name", None)

        if state_name in ("CREATED", "STOPPED"):
            return ("Start the Kernel when you're ready.", "Review the current configuration.")
        if state_name == "RUNNING":
            return ("Review system health.", "Run a full diagnostics sweep.", "Ask me to run the demonstration.")
        if state_name == "FAILED":
            return ("Something needs your attention -- the last operation failed.", "Check the logs before retrying.")
        return ("Inspect registered modules.", "Review system health.")

    def greet(self) -> IdentityResponse:
        """A dedicated greeting -- distinct from `respond()` so a greeting
        never gets recorded as if the user had typed the word "greeting"."""
        text = self._greeting_text()
        response = IdentityResponse(
            text=text,
            suggestions=self.proactive_suggestions(),
            mood="professional",
            metadata={"history": len(self._history), "mode": self.mode.value},
        )
        self.remember(self.persona.name, response.text)
        return response

    def metadata(self) -> dict[str, Any]:
        return {
            "persona": self.persona.name,
            "version": self.persona.version,
            "mode": self.mode.value,
            "clearance": self.clearance.name,
            "authenticated": self._founder.authenticated,
            "session": self._founder.session_id,
            "traits": tuple(trait.name for trait in self.persona.traits),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "persona": self.persona.name,
            "version": self.persona.version,
            "mode": self.mode.value,
            "authenticated": self._founder.authenticated,
            "clearance": self.clearance.name,
            "history": len(self._history),
            "kernel": self.kernel_context(),
            "metadata": self.metadata(),
        }

    def kernel_context(self) -> dict[str, Any]:
        if self._kernel is None:
            return {"connected": False}
        try:
            return {
                "connected": True,
                "state": getattr(self._kernel, "state", None),
            }
        except Exception:
            logger.exception("Unable to query Kernel.")
            return {"connected": False}

    def bind_kernel(self, kernel: Any) -> None:
        self._kernel = kernel
        logger.debug("Kernel bound to Identity.")

    def bind_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        logger.debug("EventBus attached to Identity.")