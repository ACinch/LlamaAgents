import asyncio
import time
from pathlib import Path

import pytest

from llama_agents.config import Config, QueueConfig, SandboxConfig, load_config
from llama_agents.queue.paths import ensure_dirs
from llama_agents.queue.worker import JobQueueWorker
from llama_agents.runtime import Runtime


@pytest.mark.live
@pytest.mark.asyncio
async def test_queue_e2e_processes_a_job(tmp_path: Path):
    """Drive the full stack — real Runtime, real llama-server — to confirm
    a job dropped in inbox ends up in done with a real model-generated response.
    """
    # Use the project's config but override sandbox + queue root for isolation.
    base_cfg = load_config("config.toml")
    cfg = Config.model_validate({
        **base_cfg.model_dump(),
        "sandbox": SandboxConfig(
            allowed_dirs=[tmp_path],
            shell_allowlist=base_cfg.sandbox.shell_allowlist,
        ).model_dump(),
        "queue": QueueConfig(
            enabled=True, root=tmp_path / "q",
            poll_interval_seconds=0.5, max_concurrent=1,
            max_retries=0, retry_backoff_seconds=0.0,
            max_iterations=8, drain_timeout_seconds=10.0,
        ).model_dump(),
    })

    rt = await Runtime.create(cfg)
    worker = JobQueueWorker(rt, cfg.queue)
    task = asyncio.create_task(worker.run())
    try:
        ensure_dirs(cfg.queue.root)
        (tmp_path / "q" / "inbox" / "hello.md").write_text(
            "Reply with exactly the word: pong"
        )
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if (tmp_path / "q" / "done" / "hello.md").exists():
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("job did not complete within 60s")

        result_text = (tmp_path / "q" / "done" / "hello.md").read_text()
        assert result_text.strip()  # non-empty
        events_text = (tmp_path / "q" / "done" / "hello.events.jsonl").read_text()
        assert '"type": "Done"' in events_text
    finally:
        await worker.drain(timeout=5.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await rt.aclose()
