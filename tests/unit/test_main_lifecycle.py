from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, Awaitable

from llm_judge.main import app


def _run_maybe_async(fn: Callable[[], Any]) -> Any:
    out = fn()

    # If handler returns a coroutine / awaitable, run it without pytest-asyncio.
    # mypy: asyncio.run requires a Coroutine, not a generic Awaitable, so wrap it.
    if inspect.isawaitable(out):

        async def _runner(a: Awaitable[Any]) -> Any:
            return await a

        asyncio.run(_runner(out))
        return None

    return out


def test_app_startup_and_shutdown_handlers_execute() -> None:
    # FastAPI stores event handlers on the router
    for fn in app.router.on_startup:
        _run_maybe_async(fn)

    for fn in app.router.on_shutdown:
        _run_maybe_async(fn)
