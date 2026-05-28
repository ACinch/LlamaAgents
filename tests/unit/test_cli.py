import sys
import io

from typer.testing import CliRunner

from llama_agents.cli import app


def test_cli_shows_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.stdout
    assert "serve" in result.stdout


def test_cli_renders_memory_events(capsys):
    from llama_agents.cli import _render_event
    from llama_agents.events import MemoryEvicted, MemoryStored

    _render_event(MemoryStored(blob_id="abcdef1234567890", kind="tool_result", scope="turn", bytes_=2048))
    _render_event(MemoryEvicted(blob_id="abcdef1234567890", turn=3, bytes_freed=2048))

    captured = capsys.readouterr()
    assert "mem:abcdef12" in captured.err
    assert "memory" in captured.err.lower() or "stored" in captured.err.lower()
    assert "evict" in captured.err.lower() or "-2.0 KB" in captured.err


def test_cli_renders_reviewer_verdict():
    from io import StringIO
    from rich.console import Console
    from llama_agents.cli import _render_event
    from llama_agents.events import ReviewerVerdict
    import llama_agents.cli as cli_mod

    buf = StringIO()
    cli_mod.console = Console(file=buf, force_terminal=False, no_color=True, width=200)
    _render_event(ReviewerVerdict(attempt=1, reviewer_idx=2, accepted=True,
                                  feedback=""))
    text = buf.getvalue()
    assert "reviewer 2" in text or "Reviewer 2" in text
    assert "✓" in text
