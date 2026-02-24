from __future__ import annotations

import asyncio
import inspect

from llm_judge.main import app


def _run_maybe_async(fn):
    out = fn()
    # If handler returns a coroutine / awaitable, run it without pytest-asyncio
    if inspect.isawaitable(out):
        asyncio.run(out)
    return out


def test_app_startup_and_shutdown_handlers_execute() -> None:
    # FastAPI stores event handlers on the router
    for fn in app.router.on_startup:
        _run_maybe_async(fn)

    for fn in app.router.on_shutdown:
        _run_maybe_async(fn)
