from __future__ import annotations

from typing import Protocol


class Prompter(Protocol):
    """Interaction surface for the install wizard.

    Production uses RichPrompter (terminal). Tests use RecordedPrompter
    (scripted answers, recorded prompts).
    """

    def ask(self, question: str, *, default: str | None = None) -> str: ...
    def confirm(self, question: str, *, default: bool = True) -> bool: ...
    def choose(
        self, question: str, options: list[str], *, default_index: int = 0
    ) -> int: ...
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...


class RecordedPrompter:
    """Test double: returns scripted answers; records prompts and messages."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts_seen: list[str] = []
        self.messages: list[tuple[str, str]] = []

    def _next(self) -> str:
        if not self._answers:
            raise RuntimeError("RecordedPrompter ran out of scripted answers")
        return self._answers.pop(0)

    def ask(self, question: str, *, default: str | None = None) -> str:
        self.prompts_seen.append(question)
        raw = self._next()
        if not raw and default is not None:
            return default
        return raw

    def confirm(self, question: str, *, default: bool = True) -> bool:
        self.prompts_seen.append(question)
        raw = self._next().strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "true", "1")

    def choose(
        self, question: str, options: list[str], *, default_index: int = 0
    ) -> int:
        self.prompts_seen.append(question)
        raw = self._next().strip()
        if not raw:
            return default_index
        idx = int(raw) - 1  # 1-indexed input
        if idx < 0 or idx >= len(options):
            raise ValueError(f"choose: index {idx + 1} out of range 1..{len(options)}")
        return idx

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warn(self, message: str) -> None:
        self.messages.append(("warn", message))


class RichPrompter:
    """Production adapter using rich.prompt."""

    def ask(self, question: str, *, default: str | None = None) -> str:
        from rich.prompt import Prompt
        return Prompt.ask(question, default=default or "")

    def confirm(self, question: str, *, default: bool = True) -> bool:
        from rich.prompt import Confirm
        return Confirm.ask(question, default=default)

    def choose(
        self, question: str, options: list[str], *, default_index: int = 0
    ) -> int:
        from rich.prompt import IntPrompt
        lines = [f"  [{i + 1}] {opt}" for i, opt in enumerate(options)]
        print("\n".join(lines))
        choice = IntPrompt.ask(question, default=default_index + 1)
        return choice - 1

    def info(self, message: str) -> None:
        from rich.console import Console
        Console().print(message)

    def warn(self, message: str) -> None:
        from rich.console import Console
        Console().print(f"[yellow]{message}[/yellow]")
