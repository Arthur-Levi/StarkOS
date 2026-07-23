"""
StarkOS v0.4
============

main.py

Application bootstrap and entry point.

Responsibilities
----------------
* Parse command-line arguments.
* Load external configuration.
* Configure logging.
* Build immutable bootstrap configuration.
* Delegate execution to the asynchronous runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.config
import os
import platform
import sys
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.auto_engineer import AutoEngineer, AutoEngineerConfig
from core.cognitive_engine import CognitiveEngine, CognitiveEngineConfig
from core.config_manager import ConfigManager, Environment
from core.event_bus import EventBus
from core.health_monitor import HealthMonitor
from core.identity import Identity
from core.kernel import Kernel, KernelConfig, KernelState
from core.knowledge_graph import KnowledgeGraph, KnowledgeGraphConfig
from core.lifecycle import LifecycleManager
from core.logger import LoggingConfig
from core.logger import LoggingConfigurationError as CoreLoggingConfigurationError
from core.logger import configure_logging as configure_core_logging
from core.logger import get_logger
from core.module_registry import ModuleRegistry
from core.service_container import ServiceContainer
from core.vision_engine import VisionEngine, VisionEngineConfig
from core.voice_interface import VoiceInterface, VoiceInterfaceConfig
from interfaces.cli.console import StarkConsole

logger = get_logger("main")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONFIG_PATH = Path("config/config.yaml")
DEFAULT_ENVIRONMENT = "development"
SUPPORTED_ENVIRONMENTS = ("development", "testing", "production")
DEFAULT_LOG_LEVEL = "INFO"
APPLICATION_NAME = "StarkOS"
APPLICATION_VERSION = "0.4"


def resolve_config_path(path: Path) -> Path:
    """Resolve a config path against the repository layout.

    The repository currently ships the config as Config.Yaml at the project
    root, while the bootstrap previously expected config/config.yaml.
    This helper accepts both locations and keeps the CLI behavior stable.
    """
    if path.exists():
        return path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        path,
        repo_root / path,
        repo_root / "Config.Yaml",
        repo_root / "config" / "config.yaml",
        repo_root / "config" / "Config.Yaml",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path

# =============================================================================
# Exceptions
# =============================================================================

class BootstrapError(RuntimeError):
    """Base exception raised during application bootstrap."""

class ConfigurationError(BootstrapError):
    """Raised when configuration loading fails."""

class LoggingConfigurationError(BootstrapError):
    """Raised when logging configuration fails."""

class UnsupportedEnvironmentError(BootstrapError):
    """Raised when the execution environment is unsupported."""

# =============================================================================
# Bootstrap Configuration
# =============================================================================

@dataclass(slots=True, frozen=True)
class BootstrapConfig:
    config_path: Path
    environment: str
    debug: bool
    demo: bool
    no_voice: bool
    log_level: str
    raw_config: Mapping[str, Any] = field(default_factory=dict)

# =============================================================================
# Startup Metrics
# =============================================================================

@dataclass(slots=True)
class StartupMetrics:
    started_at: float = field(default_factory=time.perf_counter)
    finished_at: float = 0.0
    environment: str = ""
    success: bool = False

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

# =============================================================================
# CLI
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="starkos",
        description=f"{APPLICATION_NAME} Runtime v{APPLICATION_VERSION}",
    )

    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML configuration file.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--demo", action="store_true", help="Execute the demonstration after startup.")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice interfaces.")
    parser.add_argument(
        "--env",
        default=os.getenv("STARK_ENV", DEFAULT_ENVIRONMENT),
        choices=SUPPORTED_ENVIRONMENTS,
        help="Execution environment.",
    )

    return parser

# =============================================================================
# Configuration
# =============================================================================

def load_config(path: Path) -> Mapping[str, Any]:
    resolved_path = resolve_config_path(path)
    if not resolved_path.exists():
        raise ConfigurationError(f"Configuration file not found: {resolved_path}")
    try:
        with resolved_path.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp)
    except yaml.YAMLError as exc:
        raise ConfigurationError("Invalid YAML configuration.") from exc
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a mapping.")
    return config
    return config

def configure_logging(config: Mapping[str, Any], *, debug: bool) -> None:
    """
    Configure logging for the whole process via core.logger -- the same
    centralized setup (console + optional rotating file, plain-text or
    JSON, correlation ids) every module's logger participates in.

    A `logging:` section with a `version` key is treated as a raw
    stdlib `dictConfig` schema for advanced use; otherwise StarkOS's own
    simple {level, structured, log_file} shape is used.
    """
    logging_cfg = config.get("logging") or {}

    if isinstance(logging_cfg, dict) and "version" in logging_cfg:
        try:
            logging.config.dictConfig(logging_cfg)
            logger.debug("Logging configured from a raw dictConfig section in YAML.")
            return
        except Exception as exc:
            raise LoggingConfigurationError("Invalid logging configuration.") from exc

    level_name = "DEBUG" if debug else str(logging_cfg.get("level", DEFAULT_LOG_LEVEL))
    log_file = logging_cfg.get("log_file")

    try:
        configure_core_logging(
            LoggingConfig(
                level=level_name,
                structured=bool(logging_cfg.get("structured", False)),
                log_file=Path(log_file) if log_file else None,
            )
        )
    except CoreLoggingConfigurationError as exc:
        raise LoggingConfigurationError("Invalid logging configuration.") from exc

# =============================================================================
# Bootstrap Assembly
# =============================================================================

def build_bootstrap_config(args: argparse.Namespace) -> BootstrapConfig:
    """
    Turn parsed CLI arguments into an immutable BootstrapConfig, loading the
    YAML configuration file along the way.

    NOTE: the original main.py called this function from async_main() but
    never defined it -- every run failed immediately with a NameError.
    """
    config_path = resolve_config_path(Path(args.config))
    raw_config = load_config(config_path)

    return BootstrapConfig(
        config_path=config_path,
        environment=args.env,
        debug=bool(args.debug),
        demo=bool(args.demo),
        no_voice=bool(args.no_voice),
        log_level="DEBUG" if args.debug else DEFAULT_LOG_LEVEL,
        raw_config=raw_config,
    )

# =============================================================================
# Helpers
# =============================================================================

def validate_runtime() -> None:
    if sys.version_info < (3, 11):
        raise UnsupportedEnvironmentError("Python 3.11 or newer is required.")

def print_banner() -> None:
    print(
        rf"""
========================================================================

   ███████╗████████╗ █████╗ ██████╗ ██╗  ██╗
   ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║ ██╔╝
   ███████╗   ██║   ███████║██████╔╝█████╔╝
   ╚════██║   ██║   ██╔══██║██╔══██╗██╔═██╗
   ███████║   ██║   ██║  ██║██║  ██║██║  ██╗
   ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

                     {APPLICATION_NAME}
                       Version {APPLICATION_VERSION}

Python : {platform.python_version()}
Platform: {platform.system()} {platform.release()}

========================================================================
"""
    )

def print_report(title: str, report: Mapping[str, Any]) -> None:
    print()
    print(f"== {title.upper()} ==")
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key:<22} {sub_value}")
        else:
            print(f"  {key:<24} {value}")
    print()

# =============================================================================
# Composition Root
# =============================================================================

def build_kernel(bootstrap: BootstrapConfig) -> Kernel:
    """
    Composition root: wires ConfigManager, EventBus, ModuleRegistry,
    LifecycleManager, HealthMonitor, Identity, the five v0.4 cognitive-
    stack modules (KnowledgeGraph, AutoEngineer, CognitiveEngine,
    VisionEngine, VoiceInterface) and Kernel together.

    IMPORTANT: Kernel.__init__ already registers ServiceContainer,
    ModuleRegistry, EventBus and LifecycleManager into the ServiceContainer
    it's given. Re-registering those same four here (as the original
    main.py did) raises DuplicateServiceError on every boot. This
    composition root only registers what Kernel does NOT register on its
    own: ConfigManager, HealthMonitor, Identity and the cognitive-stack
    modules below.

    Registration order matters only for readability of the startup log --
    every module resolves its siblings from the already-fully-populated
    ServiceContainer during its own initialize(), not from registration
    order, so there's no hidden ordering dependency here.
    """
    logger.info("Building StarkOS runtime.")

    services = ServiceContainer()

    config_manager = ConfigManager(
        services=services,
        config_path=bootstrap.config_path,
        environment=Environment(bootstrap.environment),
    )
    config_manager.load()
    config_manager.remember_state()
    services.register_instance(config_manager)

    event_bus = EventBus()
    config_manager.bind_event_bus(event_bus)

    registry = ModuleRegistry(service_container=services)
    lifecycle = LifecycleManager(registry=registry, services=services, event_publisher=event_bus)
    health_monitor = HealthMonitor(services=services, registry=registry)

    identity = Identity(services=services)
    services.register_instance(identity)

    # --- v0.4 cognitive stack -------------------------------------------
    knowledge_graph = KnowledgeGraph(
        services=services,
        config=KnowledgeGraphConfig(persist_path=Path("data/knowledge_graph.json")),
    )
    services.register_instance(knowledge_graph)

    auto_engineer = AutoEngineer(services=services, config=AutoEngineerConfig())
    auto_engineer.bind_knowledge_graph(knowledge_graph)
    services.register_instance(auto_engineer)

    cognitive_engine = CognitiveEngine(services=services, config=CognitiveEngineConfig())
    cognitive_engine.bind_identity(identity)
    cognitive_engine.bind_knowledge_graph(knowledge_graph)
    cognitive_engine.bind_auto_engineer(auto_engineer)
    services.register_instance(cognitive_engine)

    vision_engine = VisionEngine(services=services, config=VisionEngineConfig())
    vision_engine.bind_identity(identity)
    vision_engine.bind_knowledge_graph(knowledge_graph)
    vision_engine.bind_auto_engineer(auto_engineer)
    vision_engine.bind_cognitive_engine(cognitive_engine)
    services.register_instance(vision_engine)

    voice_interface = VoiceInterface(
        services=services,
        config=VoiceInterfaceConfig(enabled=not bootstrap.no_voice),
    )
    voice_interface.bind_identity(identity)
    services.register_instance(voice_interface)

    kernel_section = config_manager.kernel_config()
    kernel_config = KernelConfig(
        name=str(kernel_section.get("name", APPLICATION_NAME)),
        version=str(kernel_section.get("version", APPLICATION_VERSION)),
        startup_timeout=float(kernel_section.get("startup_timeout", 30.0)),
        shutdown_timeout=float(kernel_section.get("shutdown_timeout", 30.0)),
        enable_demo=bootstrap.demo,
        publish_events=bool(kernel_section.get("publish_events", True)),
        metadata={"environment": bootstrap.environment, "no_voice": bootstrap.no_voice},
    )

    kernel = Kernel(
        config=kernel_config,
        services=services,
        registry=registry,
        lifecycle=lifecycle,
        event_bus=event_bus,
        health_monitor=health_monitor,
    )

    identity.bind_kernel(kernel)
    identity.bind_event_bus(event_bus)
    auto_engineer.bind_kernel(kernel)
    cognitive_engine.bind_kernel(kernel)
    vision_engine.bind_kernel(kernel)
    voice_interface.bind_kernel(kernel)

    kernel.register_module(knowledge_graph, name="knowledge_graph", priority=100)
    kernel.register_module(auto_engineer, name="auto_engineer", priority=150)
    kernel.register_module(cognitive_engine, name="cognitive_engine", priority=200)
    kernel.register_module(vision_engine, name="vision_engine", priority=220)
    kernel.register_module(voice_interface, name="voice_interface", priority=250)

    logger.info("Kernel created successfully.", extra={"modules": len(registry.list_modules())})
    return kernel

# =============================================================================
# Kernel Initialization
# =============================================================================

async def initialize_kernel(kernel: Kernel) -> None:
    logger.info("Initializing Kernel.")
    try:
        await kernel.initialize()
    except Exception:
        logger.exception("Kernel initialization failed.")
        raise
    logger.info("Kernel initialized.")

# =============================================================================
# Runtime Startup
# =============================================================================

async def start_kernel(kernel: Kernel) -> None:
    logger.info("Starting runtime.")
    try:
        await kernel.start()
    except Exception:
        logger.exception("Kernel startup failed.")
        raise
    logger.info("Runtime online.")

# =============================================================================
# Console
# =============================================================================

async def run_console(kernel: Kernel) -> None:
    console = StarkConsole(kernel=kernel)
    logger.info("Launching StarkConsole.")
    try:
        await console.run()
    except KeyboardInterrupt:
        logger.warning("Console interrupted.")
        raise
    except Exception:
        logger.exception("Console terminated unexpectedly.")
        raise

# =============================================================================
# Application Runtime
# =============================================================================

async def async_main(argv: list[str] | None = None) -> int:
    metrics = StartupMetrics()
    kernel: Kernel | None = None

    try:
        validate_runtime()

        parser = build_argument_parser()
        args = parser.parse_args(argv)

        bootstrap = build_bootstrap_config(args)
        metrics.environment = bootstrap.environment

        configure_logging(bootstrap.raw_config, debug=bootstrap.debug)

        logger.info("Starting %s v%s", APPLICATION_NAME, APPLICATION_VERSION)
        logger.info("Environment: %s", bootstrap.environment)

        print_banner()

        kernel = build_kernel(bootstrap)

        await initialize_kernel(kernel)
        await start_kernel(kernel)

        if bootstrap.demo:
            logger.info("Running demonstration.")
            report = await kernel.demo()
            print_report("StarkOS Demonstration", report)

        await run_console(kernel)

        metrics.success = True
        return 0

    except KeyboardInterrupt:
        logger.warning("Shutdown requested by user.")
        return 130
    except BootstrapError:
        logger.exception("Bootstrap failed.")
        return 1
    except Exception:
        logger.exception("Unexpected fatal error.")
        return 2
    finally:
        metrics.finished_at = time.perf_counter()
        logger.info("Startup completed in %.3f seconds.", metrics.duration)

        if kernel is not None and kernel.state in (KernelState.RUNNING, KernelState.INITIALIZED):
            try:
                logger.info("Stopping Kernel.")
                await kernel.stop()
            except Exception:
                logger.exception("Kernel shutdown failed.")

# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    try:
        exit_code = asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.warning("Application interrupted.")
        exit_code = 130
    except Exception:
        logger.exception("Fatal startup error.")
        exit_code = 1
    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()