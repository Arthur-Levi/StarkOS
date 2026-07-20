"""
interfaces/cli/console.py

Interactive command-line interface for StarkOS.

Responsibilities
----------------
- Interactive REPL
- Command parsing
- ANSI rendering
- Delegation to Kernel
- Session management
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Sequence

from core.identity import Identity

logger = logging.getLogger(__name__)

class ANSI(str, Enum):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

def color(text: str, ansi: ANSI) -> str:
    return f"{ansi.value}{text}{ANSI.RESET.value}"

def banner() -> str:
    return color(
        r"""
   ███████╗████████╗ █████╗ ██████╗ ██╗  ██╗
   ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║ ██╔╝
   ███████╗   ██║   ███████║██████╔╝█████╔╝
   ╚════██║   ██║   ██╔══██║██╔══██╗██╔═██╗
   ███████║   ██║   ██║  ██║██║  ██║██║  ██╗
   ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

               StarkOS v0.4
""",
        ANSI.CYAN,
    )

class KernelProtocol(Protocol):
    """
    Structural contract the console relies on. Kept in sync with the real
    core.kernel.Kernel class -- diagnostics()/list_modules()/list_services()
    are plain (sync) methods, everything else that touches the lifecycle
    or the EventBus is async.
    """

    @property
    def state(self) -> Any:
        ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def restart(self) -> None:
        ...

    async def demo(self) -> dict[str, Any]:
        ...

    async def health(self) -> dict[str, Any]:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...

    def list_modules(self) -> Sequence[str]:
        ...

    def list_services(self) -> Sequence[str]:
        ...

    def resolve_service(self, service_type: type) -> Any:
        ...

    async def publish_event(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ...

class ConsoleState(Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"

CommandHandler = Callable[..., Awaitable[None]]

@dataclass(slots=True)
class Command:
    name: str
    handler: CommandHandler
    help: str

@dataclass(slots=True)
class StarkConsole:
    kernel: KernelProtocol
    prompt: str = field(default="STARK> ")
    state: ConsoleState = field(default=ConsoleState.CREATED, init=False)
    commands: Dict[str, Command] = field(default_factory=dict, init=False)
    history: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._register_builtin_commands()
        logger.info("Console initialized.")

    def _register_builtin_commands(self) -> None:
        self.register_command("help", self._cmd_help, "Show available commands.")
        self.register_command("exit", self._cmd_exit, "Terminate console.")
        self.register_command("quit", self._cmd_exit, "Terminate console.")
        self.register_command("status", self._cmd_status, "Display kernel status.")
        self.register_command("health", self._cmd_health, "Display health information.")
        self.register_command("modules", self._cmd_modules, "List registered modules.")
        self.register_command("services", self._cmd_services, "List registered services.")
        self.register_command("diagnostics", self._cmd_diagnostics, "Display diagnostics report.")
        self.register_command("publish", self._cmd_publish_event, "Publish an event.")
        self.register_command("clear", self._cmd_clear, "Clear terminal.")
        self.register_command("start", self._cmd_start, "Start the Kernel.")
        self.register_command("stop", self._cmd_stop, "Stop the Kernel.")
        self.register_command("restart", self._cmd_restart, "Restart the Kernel.")
        self.register_command("demo", self._cmd_demo, "Execute the official demonstration.")

    def register_command(self, name: str, handler: CommandHandler, help_text: str) -> None:
        self.commands[name] = Command(name=name, handler=handler, help=help_text)
        logger.debug("Registered command '%s'.", name)

    async def execute(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            print(color(str(exc), ANSI.RED))
            return

        command = tokens[0].lower()
        args = tokens[1:]

        handler = self.commands.get(command)
        if handler is None:
            logger.debug("Routing input to assistant.")
            await self._assistant_reply(line)
            return

        try:
            await handler.handler(*args)
        except Exception:
            logger.exception("Command failed.")
            print(color("Unexpected error.", ANSI.RED))

    async def run(self) -> None:
        self.state = ConsoleState.RUNNING
        print(banner())
        logger.info("Console started.")

        while self.state is ConsoleState.RUNNING:
            try:
                line = input(self.prompt)
            except KeyboardInterrupt:
                print()
                break
            except EOFError:
                print()
                break
            await self.execute(line)

        logger.info("Console terminated.")

    async def _cmd_help(self, *args: str) -> None:
        print()
        print(color("Available Commands", ANSI.YELLOW))
        print()
        for command in sorted(self.commands.values(), key=lambda c: c.name):
            print(f"{command.name:<15} {command.help}")
        print()

    async def _cmd_exit(self, *args: str) -> None:
        logger.info("Console shutting down.")
        try:
            await self.kernel.stop()
        except Exception:
            logger.exception("Kernel shutdown during exit failed.")
        self.state = ConsoleState.STOPPED
        print(color("Session terminated. Until next time.", ANSI.CYAN))

    async def _cmd_status(self, *args: str) -> None:
        self._jarvis("All primary systems are responding.")
        print(f"Kernel State : {self.kernel.state}")

    async def _cmd_health(self, *args: str) -> None:
        report = await self.kernel.health()
        self._jarvis("Current health assessment follows.")
        for key, value in report.items():
            print(f"{key:<20} {value}")

    async def _cmd_modules(self, *args: str) -> None:
        modules = self.kernel.list_modules()
        self._jarvis(f"{len(modules)} module(s) currently registered.")
        print()
        for module in modules:
            print(" •", module)

    async def _cmd_services(self, *args: str) -> None:
        services = self.kernel.list_services()
        self._jarvis(f"{len(services)} service(s) available.")
        print()
        for service in services:
            print(" •", service)

    async def _cmd_diagnostics(self, *args: str) -> None:
        # Kernel.diagnostics() is a plain sync method -- no I/O involved.
        report = self.kernel.diagnostics()
        self._jarvis("Diagnostics completed.")
        print()
        for key, value in report.items():
            print(f"{key:<24} {value}")

    async def _cmd_publish_event(self, *args: str) -> None:
        if not args:
            self._jarvis("A topic must be supplied.")
            return
        topic = args[0]
        await self.kernel.publish_event(topic, payload={})
        self._jarvis(f"Event '{topic}' published.")

    async def _cmd_clear(self, *args: str) -> None:
        print("\033[2J\033[H", end="")

    async def _cmd_start(self, *args: str) -> None:
        self._jarvis("Initializing primary systems. Please stand by.")
        try:
            await self.kernel.start()
            self._jarvis("All primary systems are now online.")
        except Exception:
            self._jarvis("Startup sequence failed. Review the system logs.")

    async def _cmd_stop(self, *args: str) -> None:
        self._jarvis("Beginning graceful shutdown sequence.")
        try:
            await self.kernel.stop()
            self._jarvis("Shutdown completed successfully.")
        except Exception:
            self._jarvis("Unable to complete shutdown safely.")

    async def _cmd_restart(self, *args: str) -> None:
        self._jarvis("Restarting all managed subsystems.")
        try:
            await self.kernel.restart()
            self._jarvis("Restart completed successfully.")
        except Exception:
            self._jarvis("Restart sequence could not be completed.")

    async def _cmd_demo(self, *args: str) -> None:
        self._jarvis("Preparing the engineering demonstration.")
        try:
            report = await self.kernel.demo()
            self._jarvis("Demonstration completed successfully.")
            print()
            for section, values in report.items():
                print(color(section.upper(), ANSI.YELLOW))
                if isinstance(values, dict):
                    for key, value in values.items():
                        print(f"  {key:<24} {value}")
                else:
                    print(" ", values)
                print()
        except Exception:
            logger.exception("Demonstration failed.")
            self._jarvis("The demonstration was interrupted by an unexpected error.")

    def _jarvis(self, message: str) -> None:
        print()
        print(color("JARVIS", ANSI.CYAN), message)
        print()

    async def _assistant_reply(self, prompt: str) -> None:
        # Free-text input that doesn't match a built-in command is routed to
        # the Identity subsystem, which already exists and already knows how
        # to hold a conversation -- previously this just printed a canned
        # placeholder string and never touched Identity at all.
        try:
            identity = self.kernel.resolve_service(Identity)
        except Exception:
            identity = None

        if identity is None:
            self._jarvis("Interesting request. The Identity subsystem is not available right now.")
            return

        try:
            response = identity.respond(prompt)
            self._jarvis(response.text)
            if response.suggestions:
                print(color("Suggestions:", ANSI.MAGENTA))
                for suggestion in response.suggestions:
                    print(" •", suggestion)
                print()
        except Exception:
            logger.exception("Assistant reply failed.")
            self._jarvis("I was unable to process that request just now.")