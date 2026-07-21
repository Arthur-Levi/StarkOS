"""
core/vision_engine.py
========================

Visual analysis and machine reverse-engineering orchestration for StarkOS.

Honesty about scope -- READ THIS FIRST
----------------------------------------
Single-image or single-video **mechanical 3D reconstruction** ("look at
one photo and rebuild the machine's parts, dimensions and assembly
order") is an open computer-vision research problem. It requires trained
models (part segmentation, structure-from-motion / NeRF-style geometry
inference, etc.) that do not exist in this codebase and cannot be
honestly approximated with a simple fallback the way, say, a hashing
embedding approximates semantic search.

So this module draws a hard, explicit line:

- Everything under "Visual analysis" below is **real, classical computer
  vision**, actually implemented with OpenCV/Pillow/pypdf: image
  dimensions/EXIF, dominant colors, edge density (Canny), contour
  extraction with geometric shape classification (circular/rectangular/
  polygon, via contour circularity and vertex count -- NOT semantic part
  identification), circle detection (Hough transform, useful for holes/
  shafts/bolts), and video/PDF ingestion (frame sampling, page/text/
  image extraction). This is genuinely useful signal and is fully
  tested against real generated images/video/PDFs.
- Everything under "machine reconstruction" (`reconstruct_machine`,
  `Model3D`, `DetectedPart`, `AssemblyStep`) is a **Protocol boundary**,
  not a working implementation. `VisionEngine`'s shipped default
  (`UnavailableReconstructionProvider`) always reports
  `ReconstructionStatus.UNAVAILABLE` and never invents part counts,
  dimensions, materials or 3D geometry. Real px->mm conversion for
  detected circles only happens if the caller supplies a real
  `pixels_per_mm` calibration; absent that, everything stays in pixel
  units. Wire in a real `VisionReconstructionProvider` (a trained model,
  a photogrammetry pipeline, a grounded multimodal vision model) to get
  actual reconstruction results -- exactly the same pattern used for
  `DigitalTwinQueryable` in `core.auto_engineer`.

Responsibilities
----------------
- Ingest images, videos and PDFs; extract real, classical visual/
  document signal from each.
- Provide the Protocol boundary + orchestration for machine
  reconstruction, assembly-process reconstruction and 3D model
  generation, without ever fabricating their content.
- Bridge detected parts (if a real reconstruction provider supplies any)
  into `core.auto_engineer` Assembly/BOM generation.
- Push observations to a bound Digital Twin (no concrete Digital Twin
  exists yet in StarkOS -- Protocol-only, same as `core.auto_engineer`).
- Record every ingested asset and reconstruction attempt into
  KnowledgeGraph as searchable long-term memory.
- Hand reconstruction outcomes to CognitiveEngine as new goals, so it
  can plan appropriate next steps.

`VisionEngine` satisfies the `Module` protocol (name/initialize/start/
stop) and registers with the Kernel like any other StarkOS module:

    vision = VisionEngine(services=services)
    vision.bind_knowledge_graph(knowledge_graph)
    vision.bind_auto_engineer(auto_engineer)
    kernel.register_module(vision, name="vision_engine", priority=220)

    asset = await vision.ingest("photos/gearbox.jpg")
    result = vision.reconstruct_machine([asset.id])  # -> UNAVAILABLE by default, honestly
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, Union, runtime_checkable

from core.auto_engineer import Assembly, AutoEngineer, Component
from core.cognitive_engine import CognitiveEngine
from core.identity import Identity
from core.knowledge_graph import KnowledgeGraph
from core.service_container import ServiceContainer

logger = logging.getLogger("starkos.vision_engine")

# =============================================================================
# Exceptions
# =============================================================================

class VisionEngineError(Exception):
    """Base exception for VisionEngine failures."""

class VisionProviderUnavailableError(VisionEngineError):
    """Raised when a required optional dependency (opencv-python, Pillow,
    pypdf) is not installed."""

class VisionAnalysisError(VisionEngineError):
    """Raised when ingesting or analyzing a specific asset fails."""

class ReconstructionError(VisionEngineError):
    """Raised when a VisionReconstructionProvider itself fails (raises)."""

# =============================================================================
# Media assets
# =============================================================================

class MediaType(Enum):
    IMAGE = auto()
    VIDEO = auto()
    PDF = auto()

@dataclass(slots=True, frozen=True)
class MediaAsset:
    """One piece of raw visual/document input handed to VisionEngine."""

    id: str
    media_type: MediaType
    path: Path
    label: str = ""
    added_at: datetime = field(default_factory=datetime.utcnow)

# =============================================================================
# Real, classical visual observations
# =============================================================================

@dataclass(slots=True, frozen=True)
class DetectedContour:
    """One real contour found by classical edge/contour detection.
    `shape_hint` is a *geometric* classification (circularity + vertex
    count) -- it says "this outline is roughly round/rectangular", not
    "this is a bearing" or "this is a bracket". Semantic part
    identification is not implemented."""

    shape_hint: str  # "circular" | "rectangular" | "polygon" | "irregular"
    area_px: float
    perimeter_px: float
    vertex_count: int
    bounding_box_px: tuple[int, int, int, int]  # x, y, width, height

@dataclass(slots=True, frozen=True)
class DetectedCircle:
    """One circle found by a Hough circle transform (useful for holes,
    shafts, bolt heads). `radius_mm` is only populated when the caller
    supplied a real `pixels_per_mm` calibration -- never invented."""

    center_px: tuple[float, float]
    radius_px: float
    radius_mm: Optional[float] = None

@dataclass(slots=True, frozen=True)
class ImageObservation:
    """Real, classical visual signal extracted from a single image (or
    video frame). This is raw geometric/photometric signal, not part
    identification or 3D geometry."""

    width: int
    height: int
    format: str
    mode: str
    dominant_colors: tuple[tuple[int, int, int], ...]
    edge_density: float  # fraction of pixels flagged as edges, 0..1
    contours: tuple[DetectedContour, ...] = ()
    circles: tuple[DetectedCircle, ...] = ()
    exif: dict[str, Any] = field(default_factory=dict)
    pixels_per_mm: Optional[float] = None

@dataclass(slots=True, frozen=True)
class VideoObservation:
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    sampled_frames: tuple[ImageObservation, ...] = ()

@dataclass(slots=True, frozen=True)
class PDFObservation:
    page_count: int
    extracted_text: str
    embedded_image_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

Observation = Union[ImageObservation, VideoObservation, PDFObservation]

# =============================================================================
# Real classical analyzers
# =============================================================================

class ImageAnalyzer:
    """
    Real, classical (non-ML) image analysis via OpenCV + Pillow, if
    installed: dimensions, EXIF, dominant colors, Canny edge density,
    contour extraction with geometric shape classification, and Hough
    circle detection. No dependency on either library is hard-required
    at import time -- `check_available()` reports whether they're present.
    """

    def check_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    def analyze(self, path: Path, *, pixels_per_mm: Optional[float] = None) -> ImageObservation:
        try:
            import cv2
        except ImportError as exc:
            raise VisionProviderUnavailableError("opencv-python (cv2) and numpy are required for image analysis.") from exc

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise VisionAnalysisError(f"Unable to decode image '{path}'.")

        fmt = path.suffix.lstrip(".").upper() or "UNKNOWN"
        exif = self._read_exif(path)
        return self.analyze_array(image, fmt=fmt, exif=exif, pixels_per_mm=pixels_per_mm)

    def analyze_array(
        self,
        image_bgr: Any,
        *,
        fmt: str = "ARRAY",
        exif: Optional[dict[str, Any]] = None,
        pixels_per_mm: Optional[float] = None,
    ) -> ImageObservation:
        """Analyze an already-decoded BGR image array (e.g. a video
        frame) -- avoids a round trip through disk."""
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise VisionProviderUnavailableError("opencv-python (cv2) and numpy are required for image analysis.") from exc

        try:
            height, width = image_bgr.shape[:2]
            channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
            mode = {1: "GRAYSCALE", 3: "BGR", 4: "BGRA"}.get(channels, f"{channels}-CHANNEL")

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if channels >= 3 else image_bgr

            dominant = self._dominant_colors(image_bgr if channels >= 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

            edges = cv2.Canny(gray, 80, 160)
            edge_density = float(np.count_nonzero(edges)) / edges.size if edges.size else 0.0

            contours = self._extract_contours(edges)
            circles = self._extract_circles(gray, pixels_per_mm=pixels_per_mm)

        except Exception as exc:
            raise VisionAnalysisError("Image array analysis failed.") from exc

        return ImageObservation(
            width=int(width),
            height=int(height),
            format=fmt,
            mode=mode,
            dominant_colors=dominant,
            edge_density=round(edge_density, 4),
            contours=contours,
            circles=circles,
            exif=exif or {},
            pixels_per_mm=pixels_per_mm,
        )

    def _read_exif(self, path: Path) -> dict[str, Any]:
        try:
            from PIL import ExifTags, Image
        except ImportError:
            return {}
        try:
            with Image.open(path) as img:
                raw_exif = img.getexif()
                data: dict[str, Any] = {}
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    data[str(tag)] = value if isinstance(value, (str, int, float)) else str(value)
                return data
        except Exception:
            # EXIF is best-effort metadata; its absence is not an error.
            return {}

    def _dominant_colors(self, image_bgr: Any, *, top_n: int = 5) -> tuple[tuple[int, int, int], ...]:
        import cv2
        import numpy as np

        small = cv2.resize(image_bgr, (48, 48), interpolation=cv2.INTER_AREA)
        pixels = small.reshape(-1, 3)
        binned = (pixels // 32 * 32).astype(np.int32)
        colors, counts = np.unique(binned, axis=0, return_counts=True)
        order = np.argsort(-counts)[:top_n]
        # BGR (OpenCV native) -> RGB for a more conventional external representation.
        return tuple(tuple(int(c) for c in colors[i][::-1]) for i in order)

    def _extract_contours(self, edges: Any, *, min_area: float = 25.0, max_results: int = 50) -> tuple[DetectedContour, ...]:
        import cv2

        raw_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        results: list[DetectedContour] = []
        for contour in raw_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)
            vertex_count = len(approx)

            # Vertex count from approxPolyDP is the primary, more reliable
            # signal. Circularity (4*pi*area/perimeter^2) is only used to
            # confirm round shapes among high-vertex-count contours -- a
            # PERFECT SQUARE already scores ~pi/4 = 0.785 on this metric,
            # so using circularity alone (or a threshold below ~0.85)
            # misclassifies rectangles as circular. Low-vertex shapes are
            # classified by vertex count first, unconditionally.
            circularity = (4 * 3.14159265 * area / (perimeter ** 2)) if perimeter > 0 else 0.0

            if vertex_count == 3:
                shape_hint = "polygon"
            elif vertex_count == 4:
                shape_hint = "rectangular"
            elif circularity > 0.85:
                shape_hint = "circular"
            elif vertex_count >= 5:
                shape_hint = "polygon"
            else:
                shape_hint = "irregular"

            results.append(
                DetectedContour(
                    shape_hint=shape_hint,
                    area_px=float(area),
                    perimeter_px=float(perimeter),
                    vertex_count=int(vertex_count),
                    bounding_box_px=(int(x), int(y), int(w), int(h)),
                )
            )

        results.sort(key=lambda c: c.area_px, reverse=True)
        return tuple(results[:max_results])

    def _extract_circles(self, gray: Any, *, pixels_per_mm: Optional[float]) -> tuple[DetectedCircle, ...]:
        import cv2
        import numpy as np

        height, width = gray.shape[:2]
        blurred = cv2.medianBlur(gray, 5)
        detected = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min(height, width) // 20),
            param1=100,
            param2=30,
            minRadius=5,
            maxRadius=min(height, width) // 2,
        )
        if detected is None:
            return ()

        circles: list[DetectedCircle] = []
        for cx, cy, radius in np.round(detected[0]).astype(float):
            radius_mm = (radius / pixels_per_mm) if pixels_per_mm else None
            circles.append(DetectedCircle(center_px=(float(cx), float(cy)), radius_px=float(radius), radius_mm=radius_mm))
        return tuple(circles)

class VideoAnalyzer:
    """Real video ingestion via OpenCV: frame count/fps/duration plus
    classical image analysis (via ImageAnalyzer) on a handful of sampled
    frames, avoiding a full-video, frame-by-frame processing cost."""

    def __init__(self, image_analyzer: Optional[ImageAnalyzer] = None) -> None:
        self._image_analyzer = image_analyzer or ImageAnalyzer()

    def check_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    def analyze(
        self,
        path: Path,
        *,
        sample_frames: int = 3,
        pixels_per_mm: Optional[float] = None,
    ) -> VideoObservation:
        try:
            import cv2
        except ImportError as exc:
            raise VisionProviderUnavailableError("opencv-python (cv2) is not installed.") from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VisionAnalysisError(f"Unable to open video '{path}'.")

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = (frame_count / fps) if fps > 0 else 0.0

            sampled: list[ImageObservation] = []
            if frame_count > 0 and sample_frames > 0:
                positions = sorted(
                    {
                        max(0, min(frame_count - 1, int(frame_count * i / (sample_frames + 1))))
                        for i in range(1, sample_frames + 1)
                    }
                )
                for position in positions:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    try:
                        sampled.append(
                            self._image_analyzer.analyze_array(
                                frame, fmt="VIDEO_FRAME", pixels_per_mm=pixels_per_mm
                            )
                        )
                    except VisionAnalysisError:
                        logger.exception("Failed to analyze a sampled video frame -- skipping it.")
        except Exception as exc:
            raise VisionAnalysisError(f"Failed to analyze video '{path}'.") from exc
        finally:
            capture.release()

        return VideoObservation(
            frame_count=frame_count,
            fps=round(fps, 3),
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
            sampled_frames=tuple(sampled),
        )

class PDFAnalyzer:
    """Real PDF ingestion via pypdf: page count, extracted text, and a
    best-effort embedded-image count. No OCR -- pages that are scanned
    images with no text layer will report empty `extracted_text`."""

    def check_available(self) -> bool:
        try:
            import pypdf  # noqa: F401
            return True
        except ImportError:
            return False

    def analyze(self, path: Path) -> PDFObservation:
        try:
            import pypdf
        except ImportError as exc:
            raise VisionProviderUnavailableError("pypdf is not installed.") from exc

        try:
            reader = pypdf.PdfReader(str(path))
            text_parts: list[str] = []
            image_count = 0
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    logger.debug("Text extraction failed for one PDF page -- continuing.")
                try:
                    image_count += len(page.images)
                except Exception:
                    pass  # Older pypdf versions may not expose page.images.

            raw_metadata = dict(reader.metadata) if reader.metadata else {}
            metadata = {str(key): str(value) for key, value in raw_metadata.items()}

            return PDFObservation(
                page_count=len(reader.pages),
                extracted_text="\n".join(text_parts).strip(),
                embedded_image_count=image_count,
                metadata=metadata,
            )
        except VisionEngineError:
            raise
        except Exception as exc:
            raise VisionAnalysisError(f"Failed to analyze PDF '{path}'.") from exc

# =============================================================================
# Machine reconstruction -- Protocol boundary, no fabricated default
# =============================================================================

class ReconstructionStatus(Enum):
    UNAVAILABLE = auto()  # no real reconstruction model configured (the always-safe default)
    PARTIAL = auto()  # a real provider ran but reported low confidence / incomplete data
    COMPLETE = auto()  # a real provider produced a full reconstruction

@dataclass(slots=True, frozen=True)
class DetectedPart:
    """One hypothesized mechanical part. Populated ONLY by a real,
    externally-supplied VisionReconstructionProvider -- VisionEngine's
    own default never creates instances of this."""

    id: str
    label: str
    part_type: str
    confidence: float  # 0..1, as reported by the provider that produced it
    estimated_dimensions_mm: Optional[tuple[float, float, float]] = None
    material_hypothesis: Optional[str] = None
    source_asset_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True, frozen=True)
class AssemblyStep:
    """One hypothesized step in reconstructing the assembly order."""

    order: int
    description: str
    part_ids: tuple[str, ...]
    confidence: float

@dataclass(slots=True, frozen=True)
class Model3D:
    """Reference to generated 3D geometry. VisionEngine never generates
    mesh data itself; `source` records which provider produced it."""

    format: str  # e.g. "step", "obj", "gltf" -- whatever the provider emits
    uri_or_path: Optional[str]
    source: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True, frozen=True)
class ReconstructionResult:
    status: ReconstructionStatus
    parts: tuple[DetectedPart, ...] = ()
    assembly_steps: tuple[AssemblyStep, ...] = ()
    model_3d: Optional[Model3D] = None
    message: str = ""
    provider_name: str = "none"

@runtime_checkable
class VisionReconstructionProvider(Protocol):
    """
    The actual "infer this machine's parts, assembly order and 3D
    geometry from images/video" capability. StarkOS v0.4 ships no
    implementation -- see the module docstring for why. Wire in a real
    model by implementing this Protocol.
    """

    def reconstruct(self, assets: Sequence[MediaAsset], observations: Sequence[Observation]) -> ReconstructionResult:
        ...

    def check_available(self) -> bool:
        ...

class UnavailableReconstructionProvider:
    """
    Default VisionReconstructionProvider: always reports
    ReconstructionStatus.UNAVAILABLE and never invents parts, dimensions
    or geometry. This is intentional, not a placeholder bug -- fabricating
    mechanical reverse-engineering data that looks authoritative but is
    made up is unsafe in a system whose stated purpose includes
    generating BOMs and manufacturing instructions from it.
    """

    def check_available(self) -> bool:
        return False

    def reconstruct(self, assets: Sequence[MediaAsset], observations: Sequence[Observation]) -> ReconstructionResult:
        return ReconstructionResult(
            status=ReconstructionStatus.UNAVAILABLE,
            message=(
                "No real vision-reconstruction model is configured. StarkOS does not ship one -- "
                "single-image/video mechanical reverse engineering requires an external model wired "
                "in via VisionReconstructionProvider. The real classical observations (contours, "
                "circles, edges) collected from your assets are available via get_observation(), but "
                "no parts, dimensions or 3D geometry were inferred."
            ),
            provider_name="unavailable_stub",
        )

# =============================================================================
# Digital Twin integration (write direction: push observations)
# =============================================================================

@runtime_checkable
class DigitalTwinUpdatable(Protocol):
    """
    The interface VisionEngine needs from a Digital Twin to push an
    observation into it. No concrete Digital Twin exists yet in StarkOS
    (see `core.auto_engineer.DigitalTwinQueryable` for the read-direction
    equivalent) -- Protocol-only until a real one is wired in via
    `bind_digital_twin()`.
    """

    def update_asset_state(self, asset_id: str, observed_state: dict[str, Any]) -> None:
        ...

# =============================================================================
# Configuration
# =============================================================================

@dataclass(slots=True)
class VisionEngineConfig:
    record_to_knowledge_graph: bool = True
    video_sample_frames: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# Vision Engine
# =============================================================================

class VisionEngine:
    """
    StarkOS visual analysis and machine-reconstruction orchestration
    module. Satisfies the `Module` protocol (name/initialize/start/stop)
    and can be registered with the Kernel like any other module. See the
    module docstring for the explicit boundary between what this class
    actually does (real classical CV/document ingestion) and what
    requires an externally-supplied model (`reconstruct_machine`).
    """

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
    _VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    _PDF_SUFFIXES = {".pdf"}

    def __init__(
        self,
        *,
        services: ServiceContainer,
        config: Optional[VisionEngineConfig] = None,
        reconstruction_provider: Optional[VisionReconstructionProvider] = None,
        image_analyzer: Optional[ImageAnalyzer] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
        pdf_analyzer: Optional[PDFAnalyzer] = None,
    ) -> None:
        self._services = services
        self._config = config or VisionEngineConfig()
        self._reconstruction_provider: VisionReconstructionProvider = reconstruction_provider or UnavailableReconstructionProvider()
        self._image_analyzer = image_analyzer or ImageAnalyzer()
        self._video_analyzer = video_analyzer or VideoAnalyzer(self._image_analyzer)
        self._pdf_analyzer = pdf_analyzer or PDFAnalyzer()

        self._kernel: Any = None
        self._identity: Optional[Identity] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._auto_engineer: Optional[AutoEngineer] = None
        self._cognitive_engine: Optional[CognitiveEngine] = None
        self._digital_twin: Any = None

        self._assets: dict[str, MediaAsset] = {}
        self._observations: dict[str, Observation] = {}

        self._image_available = True
        self._video_available = True
        self._pdf_available = True
        self._reconstruction_available = False

        logger.info("VisionEngine constructed.")

    # ------------------------------------------------------------------
    # Module protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "vision_engine"

    def bind_kernel(self, kernel: Any) -> None:
        """Kernel does not register itself into the ServiceContainer, so
        it is handed to modules explicitly -- mirrors Identity/VoiceInterface."""
        self._kernel = kernel
        logger.debug("Kernel bound to VisionEngine.")

    def bind_identity(self, identity: Identity) -> None:
        self._identity = identity
        logger.debug("Identity bound to VisionEngine.")

    def bind_knowledge_graph(self, knowledge_graph: KnowledgeGraph) -> None:
        self._knowledge_graph = knowledge_graph
        logger.debug("KnowledgeGraph bound to VisionEngine.")

    def bind_auto_engineer(self, auto_engineer: AutoEngineer) -> None:
        self._auto_engineer = auto_engineer
        logger.debug("AutoEngineer bound to VisionEngine.")

    def bind_cognitive_engine(self, cognitive_engine: CognitiveEngine) -> None:
        self._cognitive_engine = cognitive_engine
        logger.debug("CognitiveEngine bound to VisionEngine.")

    def bind_digital_twin(self, digital_twin: Any) -> None:
        """No concrete Digital Twin exists yet in StarkOS; see
        DigitalTwinUpdatable above."""
        self._digital_twin = digital_twin
        logger.debug("Digital Twin bound to VisionEngine.")

    async def initialize(self) -> None:
        logger.info("Initializing VisionEngine.")

        if self._identity is None:
            self._identity = self._services.resolve_optional(Identity)
        if self._knowledge_graph is None:
            self._knowledge_graph = self._services.resolve_optional(KnowledgeGraph)
        if self._auto_engineer is None:
            self._auto_engineer = self._services.resolve_optional(AutoEngineer)
        if self._cognitive_engine is None:
            self._cognitive_engine = self._services.resolve_optional(CognitiveEngine)

        self._image_available = await self._probe(self._image_analyzer)
        self._video_available = await self._probe(self._video_analyzer)
        self._pdf_available = await self._probe(self._pdf_analyzer)
        self._reconstruction_available = await self._probe(self._reconstruction_provider)

        if not self._image_available:
            logger.warning("Image analysis unavailable -- install opencv-python and numpy.")
        if not self._video_available:
            logger.warning("Video analysis unavailable -- install opencv-python.")
        if not self._pdf_available:
            logger.warning("PDF analysis unavailable -- install pypdf.")
        if not self._reconstruction_available:
            logger.warning(
                "No real VisionReconstructionProvider configured -- reconstruct_machine() "
                "will always report UNAVAILABLE. See the module docstring."
            )

        logger.info(
            "VisionEngine initialized.",
            extra={
                "image_available": self._image_available,
                "video_available": self._video_available,
                "pdf_available": self._pdf_available,
                "reconstruction_available": self._reconstruction_available,
            },
        )

    async def start(self) -> None:
        logger.info("VisionEngine ready.")

    async def stop(self) -> None:
        logger.info("VisionEngine stopped.")

    async def _probe(self, provider: Any) -> bool:
        check = getattr(provider, "check_available", None)
        if check is None:
            return True
        try:
            return bool(await asyncio.to_thread(check))
        except Exception:
            logger.exception("Availability probe failed for a vision provider.")
            return False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _infer_media_type(self, path: Path) -> MediaType:
        suffix = path.suffix.lower()
        if suffix in self._IMAGE_SUFFIXES:
            return MediaType.IMAGE
        if suffix in self._VIDEO_SUFFIXES:
            return MediaType.VIDEO
        if suffix in self._PDF_SUFFIXES:
            return MediaType.PDF
        raise VisionAnalysisError(f"Cannot infer media type for '{path}' -- pass media_type explicitly.")

    async def ingest(
        self,
        path: Union[str, Path],
        *,
        media_type: Optional[MediaType] = None,
        label: str = "",
        pixels_per_mm: Optional[float] = None,
    ) -> MediaAsset:
        resolved_path = Path(path)
        if not resolved_path.exists():
            raise VisionAnalysisError(f"File not found: '{resolved_path}'.")

        resolved_type = media_type or self._infer_media_type(resolved_path)
        asset = MediaAsset(id=str(uuid.uuid4()), media_type=resolved_type, path=resolved_path, label=label or resolved_path.name)

        observation = await self._analyze(asset, pixels_per_mm=pixels_per_mm)

        self._assets[asset.id] = asset
        self._observations[asset.id] = observation
        logger.info("Media asset ingested.", extra={"asset_id": asset.id, "media_type": resolved_type.name})

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            self._record_observation(asset, observation)

        return asset

    async def _analyze(self, asset: MediaAsset, *, pixels_per_mm: Optional[float]) -> Observation:
        try:
            if asset.media_type is MediaType.IMAGE:
                return await asyncio.to_thread(self._image_analyzer.analyze, asset.path, pixels_per_mm=pixels_per_mm)
            if asset.media_type is MediaType.VIDEO:
                return await asyncio.to_thread(
                    self._video_analyzer.analyze,
                    asset.path,
                    sample_frames=self._config.video_sample_frames,
                    pixels_per_mm=pixels_per_mm,
                )
            if asset.media_type is MediaType.PDF:
                return await asyncio.to_thread(self._pdf_analyzer.analyze, asset.path)
        except VisionEngineError:
            raise
        except Exception as exc:
            raise VisionAnalysisError(f"Analysis failed for asset '{asset.id}'.") from exc
        raise VisionAnalysisError(f"Unsupported media type '{asset.media_type}'.")

    def get_asset(self, asset_id: str) -> MediaAsset:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise VisionAnalysisError(f"Unknown asset '{asset_id}'.")
        return asset

    def get_observation(self, asset_id: str) -> Observation:
        if asset_id not in self._observations:
            raise VisionAnalysisError(f"No observation recorded for asset '{asset_id}'.")
        return self._observations[asset_id]

    def all_assets(self) -> tuple[MediaAsset, ...]:
        return tuple(self._assets.values())

    # ------------------------------------------------------------------
    # Machine reconstruction (Protocol-gated, honest by default)
    # ------------------------------------------------------------------

    def _collect_for_reconstruction(self, asset_ids: Sequence[str]) -> tuple[list[MediaAsset], list[Observation]]:
        assets: list[MediaAsset] = []
        observations: list[Observation] = []
        missing: list[str] = []
        for asset_id in asset_ids:
            asset = self._assets.get(asset_id)
            if asset is None or asset_id not in self._observations:
                missing.append(asset_id)
                continue
            assets.append(asset)
            observations.append(self._observations[asset_id])
        if missing:
            raise VisionAnalysisError(f"Unknown or unanalyzed asset id(s): {missing}")
        return assets, observations

    def reconstruct_machine(self, asset_ids: Sequence[str]) -> ReconstructionResult:
        """
        Attempt to reconstruct a machine's parts/assembly/3D geometry
        from one or more previously-ingested assets. With the default
        provider (no real model configured), this always returns
        ReconstructionStatus.UNAVAILABLE -- see the module docstring.
        """
        assets, observations = self._collect_for_reconstruction(asset_ids)

        try:
            result = self._reconstruction_provider.reconstruct(assets, observations)
        except Exception as exc:
            raise ReconstructionError("Vision reconstruction provider failed.") from exc

        logger.info(
            "Machine reconstruction attempted.",
            extra={"asset_count": len(asset_ids), "status": result.status.name, "provider": result.provider_name},
        )

        if self._config.record_to_knowledge_graph and self._knowledge_graph is not None:
            self._record_reconstruction(asset_ids, result)

        return result

    async def reconstruct_machine_async(self, asset_ids: Sequence[str]) -> ReconstructionResult:
        return await asyncio.to_thread(self.reconstruct_machine, asset_ids)

    def assembly_instructions(self, result: ReconstructionResult) -> tuple[str, ...]:
        """Render `result.assembly_steps` (if any) as ordered, readable
        instructions. Empty for the default UNAVAILABLE result."""
        if not result.assembly_steps:
            return ()
        ordered = sorted(result.assembly_steps, key=lambda step: step.order)
        return tuple(f"{step.order}. {step.description} (parts: {', '.join(step.part_ids)})" for step in ordered)

    # ------------------------------------------------------------------
    # AutoEngineer integration (BOM / manufacturing plan)
    # ------------------------------------------------------------------

    def to_assembly(self, result: ReconstructionResult, *, name: str) -> Assembly:
        """
        Convert a reconstruction's detected parts into an
        `core.auto_engineer.Assembly`. Requires `result.parts` to be
        non-empty (i.e. a real reconstruction provider produced them --
        the default UNAVAILABLE result never does). Component unit costs
        are always 0.0: VisionEngine has no way to know real costs
        either, and will not invent them -- fill them in from a real
        pricing source before generating a BOM you intend to act on.
        """
        if not result.parts:
            raise ReconstructionError("No detected parts to convert -- reconstruction was unavailable or empty.")

        components = [
            Component(
                part_number=part.id,
                description=part.label,
                quantity=1,
                unit_cost=0.0,
                category=part.part_type,
                attributes={"confidence": part.confidence, "material_hypothesis": part.material_hypothesis},
            )
            for part in result.parts
        ]
        return Assembly(id=str(uuid.uuid4()), name=name, components=components)

    def generate_bom_from_reconstruction(self, result: ReconstructionResult, *, name: str) -> Any:
        """Bridges to `AutoEngineer.generate_bom()`. Requires a bound
        AutoEngineer and a reconstruction result that actually contains
        detected parts."""
        if self._auto_engineer is None:
            raise VisionEngineError("No AutoEngineer bound -- call bind_auto_engineer() first.")
        assembly = self.to_assembly(result, name=name)
        return self._auto_engineer.generate_bom(assembly)

    def build_manufacturing_plan(self, result: ReconstructionResult, *, project_name: str) -> "ManufacturingPlan":
        bom = None
        if result.parts and self._auto_engineer is not None:
            try:
                bom = self.generate_bom_from_reconstruction(result, name=project_name)
            except VisionEngineError:
                logger.exception("Could not generate BOM for manufacturing plan.")

        return ManufacturingPlan(
            project_name=project_name,
            bom=bom,
            assembly_instructions=self.assembly_instructions(result),
            source_reconstruction_status=result.status,
        )

    # ------------------------------------------------------------------
    # Digital Twin integration
    # ------------------------------------------------------------------

    def push_observation_to_digital_twin(self, twin_asset_id: str, asset_id: str) -> None:
        """Push a previously-ingested asset's real observation into the
        bound Digital Twin (see DigitalTwinUpdatable)."""
        if self._digital_twin is None:
            raise VisionEngineError("No Digital Twin bound -- call bind_digital_twin() first.")

        update = getattr(self._digital_twin, "update_asset_state", None)
        if update is None:
            raise VisionEngineError("Bound Digital Twin does not implement update_asset_state().")

        observation = self.get_observation(asset_id)
        try:
            observed_state = asdict(observation)
        except Exception:
            observed_state = {}

        try:
            update(twin_asset_id, observed_state)
        except Exception as exc:
            raise VisionEngineError(f"Failed to push observation to Digital Twin for asset '{twin_asset_id}'.") from exc

        logger.info("Observation pushed to Digital Twin.", extra={"twin_asset_id": twin_asset_id, "asset_id": asset_id})

    # ------------------------------------------------------------------
    # CognitiveEngine integration
    # ------------------------------------------------------------------

    async def request_next_steps(self, result: ReconstructionResult, *, project_name: str) -> Any:
        """
        Hand the outcome of a reconstruction attempt to CognitiveEngine
        as a new goal, so it can plan appropriate next steps. Requires
        bind_cognitive_engine().
        """
        if self._cognitive_engine is None:
            raise VisionEngineError("No CognitiveEngine bound -- call bind_cognitive_engine() first.")

        if result.status is ReconstructionStatus.UNAVAILABLE:
            description = f"Visual reconstruction for '{project_name}' is unavailable: {result.message}"
        else:
            description = f"Plan next engineering steps for reconstructed project '{project_name}' ({len(result.parts)} parts detected)."

        return await self._cognitive_engine.pursue_goal(
            description, metadata={"vision_reconstruction": {"status": result.status.name}}
        )

    # ------------------------------------------------------------------
    # KnowledgeGraph integration
    # ------------------------------------------------------------------

    def _record_observation(self, asset: MediaAsset, observation: Observation) -> None:
        if self._knowledge_graph is None:
            return
        content = f"Visual observation ({asset.media_type.name}) of '{asset.label}': {self._describe_observation(observation)}"
        metadata = {"asset_id": asset.id, "media_type": asset.media_type.name, "path": str(asset.path)}
        try:
            self._knowledge_graph.remember(content, node_type="visual_observation", metadata=metadata, source="vision_engine")
        except Exception:
            logger.exception("Failed to record visual observation in KnowledgeGraph.")

    def _record_reconstruction(self, asset_ids: Sequence[str], result: ReconstructionResult) -> None:
        if self._knowledge_graph is None:
            return
        content = f"Machine reconstruction attempt ({len(asset_ids)} assets): status={result.status.name}, parts={len(result.parts)}"
        metadata = {
            "asset_ids": list(asset_ids),
            "status": result.status.name,
            "part_count": len(result.parts),
            "provider_name": result.provider_name,
        }
        try:
            self._knowledge_graph.remember(content, node_type="reconstruction_attempt", metadata=metadata, source="vision_engine")
        except Exception:
            logger.exception("Failed to record reconstruction attempt in KnowledgeGraph.")

    def _describe_observation(self, observation: Observation) -> str:
        if isinstance(observation, ImageObservation):
            return f"{observation.width}x{observation.height} {observation.format}, {len(observation.contours)} contours, {len(observation.circles)} circles, edge_density={observation.edge_density}"
        if isinstance(observation, VideoObservation):
            return f"{observation.frame_count} frames @ {observation.fps:.1f}fps, {observation.duration_seconds}s, {len(observation.sampled_frames)} sampled"
        if isinstance(observation, PDFObservation):
            return f"{observation.page_count} pages, {observation.embedded_image_count} embedded images, {len(observation.extracted_text)} chars of text"
        return str(observation)

    def recall_similar_observations(self, query: str, *, top_k: int = 5) -> tuple[Any, ...]:
        if self._knowledge_graph is None:
            raise VisionEngineError("No KnowledgeGraph bound -- call bind_knowledge_graph() first.")
        return self._knowledge_graph.recall(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        return {
            "assets_ingested": len(self._assets),
            "image_analysis_available": self._image_available,
            "video_analysis_available": self._video_available,
            "pdf_analysis_available": self._pdf_available,
            "reconstruction_available": self._reconstruction_available,
            "knowledge_graph_bound": self._knowledge_graph is not None,
            "auto_engineer_bound": self._auto_engineer is not None,
            "cognitive_engine_bound": self._cognitive_engine is not None,
            "digital_twin_bound": self._digital_twin is not None,
        }

# =============================================================================
# Manufacturing plan (BOM + assembly instructions bundled together)
# =============================================================================

@dataclass(slots=True, frozen=True)
class ManufacturingPlan:
    project_name: str
    bom: Optional[Any]  # core.auto_engineer.BillOfMaterials, if one was generated
    assembly_instructions: tuple[str, ...]
    source_reconstruction_status: ReconstructionStatus
    generated_at: datetime = field(default_factory=datetime.utcnow)