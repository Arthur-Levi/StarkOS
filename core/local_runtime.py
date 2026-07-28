"""
core/local_runtime.py
========================

100% local, air-gapped model runtime for StarkOS: orchestrates locally-
running LLM backends (Ollama, llama.cpp), with dynamic model selection,
execution isolation, and honest fallback when nothing is available.

Responsibilities
----------------
- Run entirely on local compute, with zero required external API calls
  for inference -- see the honesty note on what "air-gapped" actually
  means here.
- Real integration with two common local-inference backends: `Ollama`
  (an HTTP server on localhost, the most common way to run Llama/
  Mistral/Phi/etc. locally) via the standard library's `urllib` (no
  extra dependency), and `llama-cpp-python` for loading a local GGUF
  file directly, both optional and gracefully degrading if not
  installed/running.
- Dynamic model selection based on real, caller-declared model
  characteristics (context window, GPU acceleration, a relative compute-
  cost weight) plus real, measured latency from actual prior calls --
  never fabricated numbers.
- Best-effort local GPU detection (via `torch.cuda` if installed, else
  a `nvidia-smi` presence check), used to prefer GPU-accelerated models
  when a GPU is actually found.
- Execution isolation: every generation call runs in a worker thread
  under a hard timeout, with exceptions from one candidate model never
  propagating past a clean fallback to the next.
- Honest fallback: if no local backend is reachable, `generate()` raises
  a clear `ModelUnavailableError` -- it never fabricates a response.
- Integration with KnowledgeGraph (generations recorded as memory),
  DigitalThread (recorded as thread entries, if bound), RAGEngine (a
  `LocalRuntimeGenerationProvider` adapter -- the concrete fulfillment
  of RAGEngine's own documented "wire in a real LLM" hook), Identity
  (actor attribution) and CognitiveEngine (a bindable reference, for a
  future LLM-backed GoalInterpreter to be layered on top of `generate()`
  -- not built here, see the honesty note).

Honesty about scope
--------------------
1. **StarkOS ships no model of its own, and neither does this module.**
   `LocalRuntime` is real, correct orchestration code for talking to a
   model *you* install and run (Ollama, or a GGUF file via llama.cpp).
   With nothing registered/reachable, every code path here is designed
   to say so honestly (`ModelUnavailableError`, `has_available_model()
   == False`) rather than pretend a model responded.

2. **"Air-gapped" is a configuration-level safeguard, not OS-level
   network sandboxing.** With `LocalRuntimeConfig.air_gapped=True` (the
   default), `register_model()` refuses any provider whose declared
   `endpoint_host` isn't in `allowed_hosts` (localhost by default) --
   this catches accidental/careless misconfiguration of the providers
   shipped here. It cannot stop a deliberately malicious custom
   provider from making its own network calls; genuine air-gapping for
   a hostile-code threat model needs OS/network-level controls (a
   firewalled host, no outbound route at all), which are out of scope
   for a pure-Python module -- the same honesty boundary already drawn
   for `core.security_core.PluginSandbox`.

3. **"Cost" is a caller-declared relative compute-cost weight
   (`ModelDescriptor.relative_cost_weight`), not a dollar figure.**
   Local models running on your own hardware don't have a per-token
   API price; this number is meant to encode "how expensive is this to
   run" (bigger/slower model = higher weight) for the selection scorer,
   and is only as accurate as what you declare when registering a model.

4. **Latency figures are real measurements, never fabricated.** Before
   a model has actually been called at least once, selection uses only
   its declared static characteristics; `LatencyStats` only reflects
   actual completed calls, accumulated over the runtime's lifetime.

5. **GPU detection is best-effort.** Absence of a detected GPU means
   this module couldn't confirm one (no `torch`, no `nvidia-smi`, or
   genuinely no GPU) -- not a certified absence.

Design
------
Same shape as the rest of StarkOS: a `LocalModelProvider` Protocol with
real backend implementations (`OllamaProvider`, `LlamaCppProvider`) and
an honest default (`UnavailableModelProvider` is what `select_model()`
effectively falls back to when nothing else is registered/reachable --
see `generate()`'s fallback chain).

`LocalRuntime` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    runtime = LocalRuntime(services=services)
    runtime.register_model(
        ModelDescriptor(name="llama3.1:8b", backend="ollama", context_window=8192, tags=frozenset({"general"})),
        OllamaProvider(),
    )
    kernel.register_module(runtime, name="local_runtime", priority=25)

    result = await runtime.generate("Summarize the last design review.")

    # Fulfilling RAGEngine's own documented hook for a real generator:
    rag_engine.set_generation_provider(LocalRuntimeGenerationProvider(runtime))
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from core.digital_thread import DigitalThread
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.logger import get_logger
from core.rag_engine import HeuristicTokenEstimator, SourceCitation
from core.service_container import ServiceContainer

logger = get_logger("local_runtime")

# =============================================================================
# Exceptions
# =============================================================================

class LocalRuntimeError(Exception):
    """Base exception for LocalRuntime failures."""

class ModelUnavailableError(LocalRuntimeError):
    """Raised when no candidate local model is reachable/available."""

class InvalidPromptError(LocalRuntimeError):
    """Raised when a prompt argument is malformed (e.g. empty)."""

class AirGappedViolationError(LocalRuntimeError):
    """Raised when register_model() would violate air_gapped=True."""

# =============================================================================
# GPU detection (best-effort, honest -- see module docstring point 5)
# =============================================================================

def _detect_gpu() -> dict[str, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return {"gpu_available": True, "backend": "cuda_via_torch", "detail": torch.cuda.get_device_name(0)}
    except ImportError:
        pass
    except Exception:
        logger.exception("GPU detection via torch raised unexpectedly.")

    if shutil.which("nvidia-smi") is not None:
        return {"gpu_available": True, "backend": "nvidia_smi_detected", "detail": "nvidia-smi found on PATH; exact device not queried (torch not installed)."}

    return {"gpu_available": False, "backend": None, "detail": "No GPU detection library available or no GPU found."}

# =============================================================================
# Model descriptor and generation result
# =============================================================================

@dataclass(slots=True, frozen=True)
class ModelDescriptor:
    """Real, caller-declared characteristics of one locally-runnable
    model -- never fabricated performance numbers."""

    name: str  # whatever the backend calls it, e.g. "llama3.1:8b"
    backend: str  # "ollama" | "llama_cpp" | a custom identifier
    context_window: int
    approx_parameter_count_billions: Optional[float] = None
    gpu_accelerated: bool = False
    relative_cost_weight: float = 1.0  # see honesty note 3
    tags: frozenset[str] = frozenset()  # e.g. {"coding", "general", "fast"}

@dataclass(slots=True, frozen=True)
class LocalGenerationResult:
    text: str
    model_name: str
    backend: str
    latency_seconds: float
    prompt_tokens_estimate: Optional[int] = None
    completion_tokens_estimate: Optional[int] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True, frozen=True)
class TaskComplexity:
    """Caller's own assessment of what one generation call needs --
    LocalRuntime never invents this; CognitiveEngine/RAGEngine/whoever
    is calling supplies it (or accepts the token-count-only default)."""

    estimated_prompt_tokens: int
    requires_large_context: bool = False
    tags: frozenset[str] = frozenset()

# =============================================================================
# Latency tracking (real telemetry from actual calls)
# =============================================================================

@dataclass(slots=True)
class LatencyStats:
    call_count: int = 0
    total_latency_seconds: float = 0.0

    @property
    def average_latency_seconds(self) -> float:
        return self.total_latency_seconds / self.call_count if self.call_count else 0.0

    def record(self, latency_seconds: float) -> None:
        self.call_count += 1
        self.total_latency_seconds += latency_seconds

# =============================================================================
# Backend Protocol
# =============================================================================

@runtime_checkable
class LocalModelProvider(Protocol):
    """A backend capable of running one or more local models.
    `endpoint_host` is the network host this provider talks to (for
    air-gap validation) -- None for providers with no network endpoint
    at all (e.g. loading a GGUF file directly)."""

    endpoint_host: Optional[str]

    def check_available(self, model: ModelDescriptor) -> bool: ...
    def generate(self, model: ModelDescriptor, prompt: str, *, max_tokens: int, temperature: float) -> LocalGenerationResult: ...

# =============================================================================
# Ollama backend (real HTTP client, stdlib only)
# =============================================================================

class OllamaProvider:
    """
    Real HTTP client for a local Ollama server (https://ollama.com) --
    the most common way to run Llama/Mistral/Phi/etc. locally. Uses
    only `urllib` (standard library), so no extra Python dependency is
    needed beyond having Ollama itself installed and running.
    """

    def __init__(self, *, host: str = "localhost", port: int = 11434, timeout_seconds: float = 60.0) -> None:
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
        self.endpoint_host = host

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def check_available(self, model: ModelDescriptor) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=min(self._timeout_seconds, 5.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            available_names = {entry.get("name") for entry in payload.get("models", [])}
            return model.name in available_names
        except Exception:
            return False

    def generate(self, model: ModelDescriptor, prompt: str, *, max_tokens: int, temperature: float) -> LocalGenerationResult:
        body = json.dumps({
            "model": model.name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate", data=body, method="POST", headers={"Content-Type": "application/json"}
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ModelUnavailableError(f"Ollama at {self.base_url} is unreachable: {exc}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LocalRuntimeError(f"Ollama request to '{model.name}' failed: {exc}") from exc
        latency = time.monotonic() - started

        return LocalGenerationResult(
            text=payload.get("response", ""),
            model_name=model.name,
            backend="ollama",
            latency_seconds=latency,
            prompt_tokens_estimate=payload.get("prompt_eval_count"),
            completion_tokens_estimate=payload.get("eval_count"),
        )

# =============================================================================
# llama.cpp backend (real, optional local file loading -- no server needed)
# =============================================================================

class LlamaCppProvider:
    """
    Real integration with `llama-cpp-python` for loading a local GGUF
    model file directly (no server process, unlike Ollama). Optional
    dependency, lazily imported; degrades to reporting unavailable if
    it isn't installed or the model file doesn't exist. `endpoint_host`
    is always None -- there is no network endpoint at all.
    """

    def __init__(self, *, model_path: Path, n_gpu_layers: int = 0, n_ctx: int = 4096) -> None:
        self._model_path = model_path
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._llm: Any = None
        self.endpoint_host: Optional[str] = None

    def check_available(self, model: ModelDescriptor) -> bool:
        if not self._model_path.exists():
            return False
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelUnavailableError("The 'llama-cpp-python' package is not installed.") from exc
        try:
            self._llm = Llama(model_path=str(self._model_path), n_gpu_layers=self._n_gpu_layers, n_ctx=self._n_ctx, verbose=False)
        except Exception as exc:
            raise LocalRuntimeError(f"Failed to load local model '{self._model_path}'.") from exc
        return self._llm

    def generate(self, model: ModelDescriptor, prompt: str, *, max_tokens: int, temperature: float) -> LocalGenerationResult:
        llm = self._ensure_loaded()
        started = time.monotonic()
        try:
            output = llm(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:
            raise LocalRuntimeError(f"llama.cpp generation failed for '{model.name}': {exc}") from exc
        latency = time.monotonic() - started

        choices = output.get("choices") or []
        text = choices[0].get("text", "") if choices else ""
        usage = output.get("usage", {})
        return LocalGenerationResult(
            text=text, model_name=model.name, backend="llama_cpp", latency_seconds=latency,
            prompt_tokens_estimate=usage.get("prompt_tokens"), completion_tokens_estimate=usage.get("completion_tokens"),
        )

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class LocalRuntimeConfig:
    air_gapped: bool = True
    allowed_hosts: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
    default_max_tokens: int = 512
    default_temperature: float = 0.2
    default_timeout_seconds: float = 60.0
    record_to_knowledge_graph: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class _RegisteredModel:
    descriptor: ModelDescriptor
    provider: LocalModelProvider

# =============================================================================
# Local Runtime
# =============================================================================

class LocalRuntime:
    """
    StarkOS's local/air-gapped model orchestration module. See the
    module docstring's "Honesty about scope" section -- especially that
    no model ships with this module -- before relying on `generate()`.

    Satisfies the `Module` protocol (name/initialize/start/stop) and
    can be registered with the Kernel like any other module.
    """

    def __init__(self, *, services: ServiceContainer, config: Optional[LocalRuntimeConfig] = None) -> None:
        self._services = services
        self._config = config or LocalRuntimeConfig()
        self._models: dict[str, _RegisteredModel] = {}
        self._latency_stats: dict[str, LatencyStats] = defaultdict(LatencyStats)
        self._gpu_info = _detect_gpu()
        self._token_estimator = HeuristicTokenEstimator()

        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._digital_thread: Optional[DigitalThread] = None
        self._rag_engine: Any = None
        self._cognitive_engine: Any = None

        logger.info(
            "LocalRuntime constructed.",
            extra={"air_gapped": self._config.air_gapped, "gpu_available": self._gpu_info["gpu_available"]},
        )

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "local_runtime"

    @property
    def gpu_info(self) -> dict[str, Any]:
        return dict(self._gpu_info)

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to LocalRuntime.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to LocalRuntime.")

    def bind_digital_thread(self, digital_thread: DigitalThread) -> None:
        self._digital_thread = digital_thread
        logger.debug("DigitalThread bound to LocalRuntime.")

    def bind_rag_engine(self, rag_engine: Any) -> None:
        self._rag_engine = rag_engine
        logger.debug("RAGEngine bound to LocalRuntime.")

    def bind_cognitive_engine(self, cognitive_engine: Any) -> None:
        """Stored for a future LLM-backed GoalInterpreter to build on
        top of generate() -- LocalRuntime itself doesn't call back into
        CognitiveEngine. See the module docstring's honesty note."""
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to LocalRuntime.")

    async def initialize(self) -> None:
        logger.info("Initializing LocalRuntime.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._digital_thread is None:
            self._digital_thread = self._services.resolve_optional(DigitalThread)

        if not self._models:
            logger.warning("No local models registered yet -- generate() will raise ModelUnavailableError until register_model() is called.")

        logger.info(
            "LocalRuntime initialized.",
            extra={"registered_models": list(self._models.keys()), "gpu_available": self._gpu_info["gpu_available"]},
        )

    async def start(self) -> None:
        logger.info("LocalRuntime ready.", extra={"has_available_model": self.has_available_model()})

    async def stop(self) -> None:
        logger.info("LocalRuntime stopped.")

    # ------------------------------------------------------------------
    # Model registration (with air-gap enforcement)
    # ------------------------------------------------------------------

    def register_model(self, descriptor: ModelDescriptor, provider: LocalModelProvider) -> None:
        if self._config.air_gapped:
            self._validate_air_gapped(provider)
        self._models[descriptor.name] = _RegisteredModel(descriptor=descriptor, provider=provider)
        logger.info("Local model registered.", extra={"model": descriptor.name, "backend": descriptor.backend})

    def _validate_air_gapped(self, provider: LocalModelProvider) -> None:
        host = provider.endpoint_host
        if host is not None and host not in self._config.allowed_hosts:
            raise AirGappedViolationError(
                f"Refusing to register a provider targeting '{host}' while air_gapped=True "
                f"(allowed hosts: {sorted(self._config.allowed_hosts)})."
            )

    def unregister_model(self, name: str) -> bool:
        removed = self._models.pop(name, None) is not None
        if removed:
            logger.info("Local model unregistered.", extra={"model": name})
        return removed

    def list_models(self) -> tuple[ModelDescriptor, ...]:
        return tuple(registered.descriptor for registered in self._models.values())

    def has_available_model(self) -> bool:
        return any(self._is_available(registered) for registered in self._models.values())

    def _is_available(self, registered: _RegisteredModel) -> bool:
        try:
            return registered.provider.check_available(registered.descriptor)
        except Exception:
            logger.exception("Availability check raised for model '%s'.", registered.descriptor.name)
            return False

    # ------------------------------------------------------------------
    # Dynamic selection (real, transparent scoring over declared/measured data)
    # ------------------------------------------------------------------

    def select_model(self, complexity: Optional[TaskComplexity] = None) -> ModelDescriptor:
        ranked = self._rank_models(complexity or TaskComplexity(estimated_prompt_tokens=0))
        if not ranked:
            raise ModelUnavailableError("No local model backend is currently available.")
        return ranked[0].descriptor

    def _rank_models(self, complexity: TaskComplexity) -> list[_RegisteredModel]:
        available = [registered for registered in self._models.values() if self._is_available(registered)]

        def score(registered: _RegisteredModel) -> float:
            descriptor = registered.descriptor
            value = 0.0

            if descriptor.context_window >= complexity.estimated_prompt_tokens:
                value += 3.0
            else:
                value -= 10.0  # hard penalty -- would truncate or fail outright

            if complexity.requires_large_context:
                value += descriptor.context_window / 8192.0

            value += 2.0 * len(complexity.tags & descriptor.tags)

            stats = self._latency_stats.get(descriptor.name)
            if stats is not None and stats.call_count > 0:
                value -= stats.average_latency_seconds * 0.1  # real measured latency, once we have any

            value -= descriptor.relative_cost_weight * 0.5

            if descriptor.gpu_accelerated and self._gpu_info["gpu_available"]:
                value += 1.0

            return value

        return sorted(available, key=score, reverse=True)

    # ------------------------------------------------------------------
    # Generation (isolated execution, real fallback chain, honest failure)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        complexity: Optional[TaskComplexity] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        actor: Optional[str] = None,
    ) -> LocalGenerationResult:
        if not prompt or not prompt.strip():
            raise InvalidPromptError("prompt cannot be empty.")

        effective_complexity = complexity or TaskComplexity(estimated_prompt_tokens=self._token_estimator.estimate(prompt))
        effective_max_tokens = max_tokens if max_tokens is not None else self._config.default_max_tokens
        effective_temperature = temperature if temperature is not None else self._config.default_temperature

        ranked = self._rank_models(effective_complexity)
        if not ranked:
            raise ModelUnavailableError(
                "No local model backend is available. StarkOS ships no model of its own -- register one "
                "via register_model() (e.g. OllamaProvider with Ollama actually running, or LlamaCppProvider "
                "pointed at a local GGUF file)."
            )

        last_error: Optional[BaseException] = None
        for registered in ranked:
            try:
                result = await self._generate_with_isolation(registered, prompt, effective_max_tokens, effective_temperature)
            except Exception as exc:
                logger.warning(
                    "Model '%s' failed -- trying the next candidate if any.", registered.descriptor.name, exc_info=True
                )
                last_error = exc
                continue

            self._latency_stats[registered.descriptor.name].record(result.latency_seconds)
            self._record(prompt, result, actor)
            logger.info(
                "Local generation succeeded.",
                extra={"model": result.model_name, "backend": result.backend, "latency_seconds": round(result.latency_seconds, 3)},
            )
            return result

        raise ModelUnavailableError(f"All {len(ranked)} candidate model(s) failed.") from last_error

    async def _generate_with_isolation(
        self, registered: _RegisteredModel, prompt: str, max_tokens: int, temperature: float
    ) -> LocalGenerationResult:
        """Every call is time-boxed and runs off the event loop thread
        -- a slow or hung backend can't block the rest of StarkOS, and
        its exceptions are always caught here rather than propagating
        raw into the fallback loop above."""

        def _call() -> LocalGenerationResult:
            return registered.provider.generate(registered.descriptor, prompt, max_tokens=max_tokens, temperature=temperature)

        try:
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=self._config.default_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise LocalRuntimeError(
                f"Generation with '{registered.descriptor.name}' timed out after {self._config.default_timeout_seconds}s."
            ) from exc

    # ------------------------------------------------------------------
    # KnowledgeGraph / DigitalThread integration
    # ------------------------------------------------------------------

    def _record(self, prompt: str, result: LocalGenerationResult, actor: Optional[str]) -> None:
        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            try:
                self._knowledge_graph.remember(
                    f"Local generation via '{result.model_name}': {result.text[:200]}",
                    node_type="local_generation",
                    metadata={"model": result.model_name, "backend": result.backend, "latency_seconds": result.latency_seconds},
                    source="local_runtime",
                )
            except Exception:
                logger.exception("Failed to record local generation in KnowledgeGraph.")

        if self._digital_thread is not None:
            try:
                trace_id = self._digital_thread.begin_trace(f"Local generation via {result.model_name}", actor=actor)
                self._digital_thread.record_action(
                    trace_id=trace_id, description="Local model generation",
                    inputs={"prompt_preview": prompt[:200]},
                    method=f"LocalRuntime.{result.backend}:{result.model_name}",
                    parameters={"latency_seconds": result.latency_seconds},
                    result={"text_preview": result.text[:200]},
                    actor=actor,
                )
            except Exception:
                logger.exception("Failed to record local generation in DigitalThread.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "air_gapped": self._config.air_gapped,
            "gpu_info": self.gpu_info,
            "registered_models": [descriptor.name for descriptor in self.list_models()],
            "has_available_model": self.has_available_model(),
            "latency_stats": {
                name: {"call_count": stats.call_count, "average_latency_seconds": round(stats.average_latency_seconds, 4)}
                for name, stats in self._latency_stats.items()
            },
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "digital_thread_bound": self._digital_thread is not None,
            "rag_engine_bound": self._rag_engine is not None,
            "cognitive_engine_bound": self._cognitive_engine is not None,
        }

# =============================================================================
# RAGEngine adapter -- fulfills RAGEngine's own documented "wire in a real
# LLM" hook (see core.rag_engine.GenerationProvider / set_generation_provider)
# =============================================================================

class LocalRuntimeGenerationProvider:
    """
    Adapts LocalRuntime to RAGEngine's GenerationProvider Protocol.
    Self-declares `synthesis_method = "generated"` because, when a real
    local model is actually running behind it, that's an honest
    description -- unlike RAGEngine's own ExtractiveSynthesizer default.
    Wire this in only once `local_runtime.has_available_model()` is True
    (its own `check_available()` reflects exactly that).
    """

    synthesis_method = "generated"

    def __init__(self, local_runtime: LocalRuntime, *, max_tokens: int = 512, temperature: float = 0.2, timeout_seconds: float = 65.0) -> None:
        self._runtime = local_runtime
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds

    def check_available(self) -> bool:
        return self._runtime.has_available_model()

    def generate(self, query: str, context: str, sources: Sequence[SourceCitation]) -> str:
        prompt = self._build_prompt(query, context)

        def _run() -> LocalGenerationResult:
            # A fresh event loop in a dedicated worker thread -- this
            # sync method (required by GenerationProvider's Protocol)
            # may be called from a context that already has a running
            # asyncio loop, where asyncio.run() would raise; running in
            # its own thread sidesteps that entirely.
            return asyncio.run(self._runtime.generate(prompt, max_tokens=self._max_tokens, temperature=self._temperature))

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(_run).result(timeout=self._timeout_seconds)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise LocalRuntimeError("Local generation failed while adapting to RAGEngine.") from exc

        return result.text

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        if not context:
            return f"Question: {query}\nAnswer:"
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer based only on the context above:"