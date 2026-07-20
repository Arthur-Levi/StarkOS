"""
core/voice_interface.py
=======================

Voice interaction subsystem for StarkOS.

Responsibilities
----------------
- Text-to-speech (TTS) synthesis of spoken responses.
- Speech-to-text (STT) capture and transcription of spoken commands.
- Bridge spoken input to the Identity subsystem for open-ended conversation.
- Bridge spoken input to the Kernel for system-level voice commands
  (status, health, restart, stop, start, diagnostics, demo).
- Degrade gracefully when no microphone/speaker/engine is available --
  a missing audio stack must never take the Kernel down with it.

Design
------
TTS and STT are accessed through small Protocols (`TextToSpeechProvider`,
`SpeechToTextProvider`) rather than importing a specific engine directly.
Two default, best-effort providers are included (`Pyttsx3TextToSpeech`,
`SpeechRecognitionSTT`), each lazily importing its optional dependency so
that VoiceInterface can be constructed, registered and even initialized
on a machine that has neither `pyttsx3` nor `speech_recognition` installed
-- it will simply report those capabilities as unavailable instead of
raising. Swapping in a cloud provider (e.g. a hosted STT/TTS API) later
means writing a new class that satisfies the same Protocol; VoiceInterface
itself never changes.

VoiceInterface implements the `core.module_registry.Module` protocol
(name / initialize / start / stop) so it can be registered exactly like
any other StarkOS module:

    voice = VoiceInterface(services=services)
    voice.bind_kernel(kernel)
    kernel.register_module(voice, name="voice_interface", priority=200)

`bind_kernel()` mirrors the pattern already used by `Identity` -- Kernel
does not register itself into the ServiceContainer, so modules that need
it receive it via an explicit bind call from the composition root, not
via `services.resolve(...)`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable

from core.identity import Identity, IdentityResponse
from core.service_container import ServiceContainer

logger = logging.getLogger("starkos.voice_interface")

# =============================================================================
# Exceptions
# =============================================================================

class VoiceInterfaceError(Exception):
    """Base exception for VoiceInterface failures."""

class ProviderUnavailableError(VoiceInterfaceError):
    """Raised when the configured TTS or STT provider cannot be used."""

class SynthesisError(VoiceInterfaceError):
    """Raised when text-to-speech synthesis fails."""

class RecognitionError(VoiceInterfaceError):
    """Raised when speech capture or transcription fails."""

# =============================================================================
# State
# =============================================================================

class VoiceState(Enum):
    DISABLED = auto()
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    FAILED = auto()

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class VoiceInterfaceConfig:
    """Runtime configuration for VoiceInterface."""

    enabled: bool = True
    # If True, start() spawns a background task that continuously listens
    # for speech. If False (default), the module runs push-to-talk style:
    # callers invoke listen()/handle_command() explicitly (e.g. from a CLI
    # "voice" command or a hotkey handler).
    continuous_listening: bool = False
    # Naive substring wake-word gate (not a real wake-word engine). Set to
    # None or "" to treat every captured utterance as an addressed command.
    wake_word: Optional[str] = "stark"
    language: str = "en-US"
    speech_rate: int = 175
    volume: float = 1.0
    voice_id: Optional[str] = None
    listen_timeout: float = 5.0
    phrase_time_limit: float = 10.0
    energy_threshold: int = 300
    speak_responses: bool = True
    max_history: int = 200
    # Pause applied in the continuous-listening loop after a failed/empty
    # capture, before retrying. Without this, a persistently failing
    # microphone (disconnected, permission denied, provider bug) turns the
    # loop into a 100%-CPU busy-loop that also floods the logs.
    listen_retry_backoff: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Data Models
# =============================================================================

@dataclass(slots=True, frozen=True)
class VoiceCommand:
    """A single captured utterance, normalized for command matching."""

    raw_text: str
    normalized_text: str
    wake_word_detected: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True, frozen=True)
class VoiceInteractionResult:
    """Outcome of routing one VoiceCommand through Kernel/Identity."""

    command: VoiceCommand
    response_text: Optional[str]
    identity_response: Optional[IdentityResponse]
    spoken: bool
    handled_by: str  # "kernel" | "identity" | "none"

# =============================================================================
# Provider Protocols
# =============================================================================

@runtime_checkable
class TextToSpeechProvider(Protocol):
    """
    Blocking TTS contract. Implementations perform actual audio I/O and are
    always invoked by VoiceInterface via `asyncio.to_thread`, never awaited
    directly -- keep implementations synchronous.
    """

    def synthesize(
        self,
        text: str,
        *,
        rate: int,
        volume: float,
        voice_id: Optional[str],
    ) -> None:
        ...

    def check_available(self) -> bool:
        ...

@runtime_checkable
class SpeechToTextProvider(Protocol):
    """Blocking STT contract. See TextToSpeechProvider for the threading note."""

    def listen(
        self,
        *,
        timeout: float,
        phrase_time_limit: float,
        energy_threshold: int,
    ) -> str:
        ...

    def check_available(self) -> bool:
        ...

# =============================================================================
# Default Providers (optional dependencies, imported lazily)
# =============================================================================

class Pyttsx3TextToSpeech:
    """
    Offline TTS provider backed by the `pyttsx3` package. The dependency is
    imported lazily so VoiceInterface can be constructed and even
    initialized without it installed -- `check_available()` simply returns
    False and VoiceInterface falls back to text-only responses.
    """

    def __init__(self) -> None:
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            import pyttsx3
        except ImportError as exc:
            raise ProviderUnavailableError("pyttsx3 is not installed.") from exc
        try:
            self._engine = pyttsx3.init()
        except Exception as exc:
            raise ProviderUnavailableError("Unable to initialize the pyttsx3 engine.") from exc
        return self._engine

    def check_available(self) -> bool:
        try:
            self._ensure_engine()
            return True
        except ProviderUnavailableError:
            return False

    def synthesize(
        self,
        text: str,
        *,
        rate: int,
        volume: float,
        voice_id: Optional[str],
    ) -> None:
        engine = self._ensure_engine()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        if voice_id:
            engine.setProperty("voice", voice_id)
        engine.say(text)
        engine.runAndWait()

class SpeechRecognitionSTT:
    """
    Microphone-based STT provider backed by the `speech_recognition`
    package, using its default Google Web Speech recognizer (requires
    network connectivity and no API key). Swap in a fully offline engine
    (e.g. Vosk, local Whisper) by implementing `SpeechToTextProvider`
    instead -- VoiceInterface only depends on the Protocol.
    """

    def __init__(self) -> None:
        self._recognizer: Any = None
        self._microphone_factory: Any = None

    def _ensure_ready(self) -> None:
        if self._recognizer is not None:
            return
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise ProviderUnavailableError("speech_recognition is not installed.") from exc
        try:
            self._recognizer = sr.Recognizer()
            self._microphone_factory = sr.Microphone
            self._sr = sr
        except Exception as exc:
            raise ProviderUnavailableError("Unable to initialize speech_recognition.") from exc

    def check_available(self) -> bool:
        try:
            self._ensure_ready()
            # A working microphone is still not guaranteed here (headless
            # machines, containers) -- this only confirms the library and
            # its recognizer/microphone bindings are importable.
            return True
        except ProviderUnavailableError:
            return False

    def listen(
        self,
        *,
        timeout: float,
        phrase_time_limit: float,
        energy_threshold: int,
    ) -> str:
        self._ensure_ready()
        self._recognizer.energy_threshold = energy_threshold
        try:
            with self._microphone_factory() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
        except Exception as exc:
            raise RecognitionError("Unable to capture audio from the microphone.") from exc

        try:
            return self._recognizer.recognize_google(audio)
        except Exception as exc:
            raise RecognitionError("Unable to transcribe the captured audio.") from exc

# =============================================================================
# Voice Interface
# =============================================================================

class VoiceInterface:
    """
    StarkOS voice interaction module.

    Satisfies the `Module` protocol (name/initialize/start/stop) and can be
    registered with the Kernel like any other module. Routes spoken (or,
    via `handle_text_command`, plain text) input to either a Kernel-level
    action (status, health, restart, start, stop, diagnostics, demo) or,
    failing a match, to `Identity.respond()` for open-ended conversation --
    then speaks the resulting text back, if a TTS provider is available.
    """

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: VoiceInterfaceConfig | None = None,
        tts_provider: TextToSpeechProvider | None = None,
        stt_provider: SpeechToTextProvider | None = None,
    ) -> None:
        self._services = services
        self._config = config or VoiceInterfaceConfig()
        self._tts: TextToSpeechProvider = tts_provider or Pyttsx3TextToSpeech()
        self._stt: SpeechToTextProvider = stt_provider or SpeechRecognitionSTT()

        self._state = VoiceState.IDLE if self._config.enabled else VoiceState.DISABLED
        self._kernel: Any = None
        self._identity: Identity | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._tts_available = False
        self._stt_available = False
        self._history: list[VoiceInteractionResult] = []
        self._command_table: dict[str, str] = self._build_command_table()

        logger.info("VoiceInterface constructed.", extra={"enabled": self._config.enabled})

    # ------------------------------------------------------------------
    # Identity / Kernel wiring
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "voice_interface"

    @property
    def state(self) -> VoiceState:
        return self._state

    def bind_kernel(self, kernel: Any) -> None:
        """
        Kernel does not register itself into the ServiceContainer, so it
        is handed to modules explicitly -- mirrors Identity.bind_kernel().
        """
        self._kernel = kernel
        logger.debug("Kernel bound to VoiceInterface.")

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to VoiceInterface.")

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        logger.info("Initializing VoiceInterface.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)

        if not self._config.enabled:
            self._state = VoiceState.DISABLED
            logger.warning("VoiceInterface disabled via configuration.")
            return

        self._tts_available = await self._probe(self._tts, "TTS")
        self._stt_available = await self._probe(self._stt, "STT")

        if not self._tts_available:
            logger.warning("TTS provider unavailable -- responses will be text-only.")
        if not self._stt_available:
            logger.warning(
                "STT provider unavailable -- voice capture disabled; "
                "handle_text_command() still works for text-driven input."
            )

        self._state = VoiceState.IDLE
        logger.info(
            "VoiceInterface initialized.",
            extra={"tts_available": self._tts_available, "stt_available": self._stt_available},
        )

    async def start(self) -> None:
        if self._state is VoiceState.DISABLED:
            logger.info("VoiceInterface start skipped (disabled).")
            return

        if self._config.continuous_listening and self._stt_available:
            self._listen_task = asyncio.create_task(self._listen_loop(), name="voice-listen-loop")
            logger.info("Continuous listening loop started.")
        else:
            logger.info("VoiceInterface ready (push-to-talk mode).")

    async def stop(self) -> None:
        logger.info("Stopping VoiceInterface.")

        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Listening loop raised during shutdown.")
            finally:
                self._listen_task = None

        self._state = VoiceState.DISABLED
        logger.info("VoiceInterface stopped.")

    # ------------------------------------------------------------------
    # Availability probing
    # ------------------------------------------------------------------

    async def _probe(self, provider: Any, label: str) -> bool:
        check = getattr(provider, "check_available", None)
        if check is None:
            return True
        try:
            return bool(await asyncio.to_thread(check))
        except Exception:
            logger.exception("%s availability probe failed.", label)
            return False

    # ------------------------------------------------------------------
    # Speech synthesis
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> bool:
        """
        Synthesize and play `text`. Returns True if audio was actually
        produced, False if skipped (disabled, no TTS provider) or failed.
        Never raises for an unavailable provider -- callers can always
        rely on the text-based response being delivered by other means
        (console, log, whatever surfaced the original request).
        """
        if not text:
            return False
        if self._state is VoiceState.DISABLED or not self._config.speak_responses:
            logger.debug("Speak skipped (voice disabled or speak_responses=False).")
            return False
        if not self._tts_available:
            logger.debug("Speak skipped -- TTS provider unavailable.")
            return False

        previous_state = self._state
        self._state = VoiceState.SPEAKING
        try:
            await asyncio.to_thread(
                self._tts.synthesize,
                text,
                rate=self._config.speech_rate,
                volume=self._config.volume,
                voice_id=self._config.voice_id,
            )
            return True
        except Exception as exc:
            logger.exception("Speech synthesis failed.")
            raise SynthesisError("Unable to synthesize speech.") from exc
        finally:
            self._state = previous_state

    # ------------------------------------------------------------------
    # Speech capture
    # ------------------------------------------------------------------

    async def listen(self) -> VoiceCommand:
        """Capture and transcribe one utterance from the microphone."""
        if not self._stt_available:
            raise ProviderUnavailableError("Speech-to-text provider is not available.")

        previous_state = self._state
        self._state = VoiceState.LISTENING
        try:
            raw_text = await asyncio.to_thread(
                self._stt.listen,
                timeout=self._config.listen_timeout,
                phrase_time_limit=self._config.phrase_time_limit,
                energy_threshold=self._config.energy_threshold,
            )
        except VoiceInterfaceError:
            raise
        except Exception as exc:
            # In continuous-listening mode this branch fires routinely
            # (nobody spoke before `listen_timeout` elapsed) -- that is
            # normal operation, not a fault, so it's logged at DEBUG
            # rather than as an ERROR with a full stack trace.
            logger.debug("No speech captured/transcribed: %s", exc)
            raise RecognitionError("Unable to capture or transcribe speech.") from exc
        finally:
            self._state = previous_state

        return self._build_command(raw_text)

    def _build_command(self, raw_text: str) -> VoiceCommand:
        normalized = raw_text.strip().lower()
        wake_word = (self._config.wake_word or "").strip().lower()

        if not wake_word:
            # No wake word configured -- every utterance counts as addressed.
            return VoiceCommand(raw_text=raw_text, normalized_text=normalized, wake_word_detected=True)

        detected = wake_word in normalized
        if detected:
            normalized = normalized.replace(wake_word, "", 1).strip()

        return VoiceCommand(raw_text=raw_text, normalized_text=normalized, wake_word_detected=detected)

    # ------------------------------------------------------------------
    # Command routing (Kernel + Identity integration)
    # ------------------------------------------------------------------

    async def handle_text_command(self, text: str) -> VoiceInteractionResult:
        """
        Route plain text through the exact same pipeline as a spoken
        command, without touching STT. Useful for a CLI "voice" command,
        tests, or any caller that already has text (e.g. a transcript from
        a different STT engine).
        """
        return await self.handle_command(self._build_command(text))

    async def handle_command(self, command: VoiceCommand) -> VoiceInteractionResult:
        if not command.wake_word_detected:
            logger.debug("Wake word not detected -- ignoring utterance.")
            result = VoiceInteractionResult(
                command=command, response_text=None, identity_response=None, spoken=False, handled_by="none"
            )
            self._remember(result)
            return result

        previous_state = self._state
        self._state = VoiceState.PROCESSING
        try:
            action = self._match_kernel_action(command.normalized_text)

            if action is not None and self._kernel is not None:
                response_text = await self._execute_kernel_action(action)
                spoken = await self._speak_safely(response_text)
                result = VoiceInteractionResult(
                    command=command,
                    response_text=response_text,
                    identity_response=None,
                    spoken=spoken,
                    handled_by="kernel",
                )
                self._remember(result)
                return result

            if self._identity is not None:
                identity_response = self._identity.respond(command.raw_text)
                spoken = await self._speak_safely(identity_response.text)
                result = VoiceInteractionResult(
                    command=command,
                    response_text=identity_response.text,
                    identity_response=identity_response,
                    spoken=spoken,
                    handled_by="identity",
                )
                self._remember(result)
                return result

            logger.warning("No Kernel action matched and Identity is unavailable -- command dropped.")
            result = VoiceInteractionResult(
                command=command, response_text=None, identity_response=None, spoken=False, handled_by="none"
            )
            self._remember(result)
            return result
        finally:
            self._state = previous_state

    async def _speak_safely(self, text: str) -> bool:
        try:
            return await self.speak(text)
        except SynthesisError:
            # Synthesis failing must never break command handling -- the
            # text response has already been produced and returned to the
            # caller regardless of whether it could be spoken aloud.
            return False

    def _match_kernel_action(self, normalized_text: str) -> str | None:
        for keyword, action in self._command_table.items():
            if keyword in normalized_text:
                return action
        return None

    def _build_command_table(self) -> dict[str, str]:
        # Naive keyword -> Kernel action mapping (substring match, not
        # NLU). Good enough for short imperative voice commands; a real
        # intent classifier can replace this table without touching the
        # rest of VoiceInterface.
        return {
            "diagnostics": "diagnostics",
            "diagnostic": "diagnostics",
            "demonstration": "demo",
            "demo": "demo",
            "restart": "restart",
            "reboot": "restart",
            "shutdown": "stop",
            "stop": "stop",
            "status": "status",
            "health": "health",
            "start": "start",
        }

    async def _execute_kernel_action(self, action: str) -> str:
        assert self._kernel is not None
        try:
            if action == "status":
                return f"Current kernel state is {self._kernel.state.name}."
            if action == "health":
                report = await self._kernel.health()
                status = report.get("monitor", {}).get("status", "unknown")
                return f"System health status is {status}."
            if action == "diagnostics":
                self._kernel.diagnostics()
                return "Diagnostics completed. Full details are available on screen."
            if action == "restart":
                await self._kernel.restart()
                return "Restart completed successfully."
            if action == "stop":
                await self._kernel.stop()
                return "Shutdown sequence initiated."
            if action == "start":
                await self._kernel.start()
                return "All primary systems are now online."
            if action == "demo":
                await self._kernel.demo()
                return "Demonstration completed successfully."
        except Exception:
            logger.exception("Kernel action '%s' failed.", action)
            return "I was unable to complete that action. Please check the system logs."

        return "Action acknowledged."

    # ------------------------------------------------------------------
    # Continuous listening loop (optional)
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        logger.info("Voice listen loop running.")
        try:
            while True:
                try:
                    command = await self.listen()
                except ProviderUnavailableError:
                    logger.error("STT provider became unavailable -- stopping listen loop.")
                    return
                except RecognitionError:
                    # Expected on silence/ambient noise/timeout -- brief
                    # pause before retrying. Without it, a persistently
                    # failing microphone spins this loop at 100% CPU and
                    # floods the logs instead of backing off.
                    await asyncio.sleep(self._config.listen_retry_backoff)
                    continue

                try:
                    await self.handle_command(command)
                except Exception:
                    logger.exception("Failed to handle voice command.")
        except asyncio.CancelledError:
            logger.info("Voice listen loop cancelled.")
            raise

    # ------------------------------------------------------------------
    # Diagnostics / history
    # ------------------------------------------------------------------

    def _remember(self, result: VoiceInteractionResult) -> None:
        self._history.append(result)
        if len(self._history) > self._config.max_history:
            del self._history[: len(self._history) - self._config.max_history]

    def history(self) -> tuple[VoiceInteractionResult, ...]:
        return tuple(self._history)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "state": self._state.name,
            "enabled": self._config.enabled,
            "tts_available": self._tts_available,
            "stt_available": self._stt_available,
            "continuous_listening": self._config.continuous_listening,
            "wake_word": self._config.wake_word,
            "history_size": len(self._history),
        }