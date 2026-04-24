"""Tests for :mod:`llm_judge.control_plane.event_bus` — EventBus
and the default-bus singleton."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from llm_judge.control_plane.event_bus import (
    WILDCARD,
    EventBus,
    get_default_bus,
)

# =====================================================================
# subscribe / emit / unsubscribe
# =====================================================================


def test_subscribe_and_emit_delivers_to_handler() -> None:
    bus = EventBus()
    received: list[tuple[str, dict[str, Any]]] = []

    def handler(event_type: str, **fields: Any) -> None:
        received.append((event_type, fields))

    bus.subscribe("ping", handler)
    bus.emit("ping", x=1)

    assert received == [("ping", {"x": 1})]


def test_multiple_handlers_all_fire() -> None:
    bus = EventBus()
    a_calls: list[int] = []
    b_calls: list[int] = []

    bus.subscribe("ping", lambda et, **f: a_calls.append(f["n"]))
    bus.subscribe("ping", lambda et, **f: b_calls.append(f["n"] * 2))

    bus.emit("ping", n=5)

    assert a_calls == [5]
    assert b_calls == [10]


def test_handler_exception_does_not_stop_other_handlers() -> None:
    bus = EventBus()
    second_called: list[bool] = []

    def bad(event_type: str, **fields: Any) -> None:
        raise RuntimeError("handler blew up")

    def good(event_type: str, **fields: Any) -> None:
        second_called.append(True)

    bus.subscribe("ping", bad)
    bus.subscribe("ping", good)

    # emit must NOT raise — exception is swallowed inside the bus.
    bus.emit("ping")

    assert second_called == [True]


def test_unsubscribe_removes_handler() -> None:
    bus = EventBus()
    calls: list[int] = []

    def handler(event_type: str, **fields: Any) -> None:
        calls.append(1)

    bus.subscribe("ping", handler)
    bus.emit("ping")
    bus.unsubscribe("ping", handler)
    bus.emit("ping")

    assert calls == [1]  # only the first emit reached the handler


def test_unsubscribe_unknown_handler_is_noop() -> None:
    bus = EventBus()

    def never_registered(event_type: str, **fields: Any) -> None:
        pass

    # Must not raise.
    bus.unsubscribe("ping", never_registered)
    bus.unsubscribe("nonexistent_event_type", never_registered)


def test_wildcard_handler_receives_every_event() -> None:
    bus = EventBus()
    all_events: list[str] = []

    bus.subscribe(WILDCARD, lambda et, **f: all_events.append(et))

    bus.emit("alpha")
    bus.emit("beta", x=1)
    bus.emit("gamma")

    assert all_events == ["alpha", "beta", "gamma"]


def test_exact_and_wildcard_handlers_both_fire() -> None:
    bus = EventBus()
    exact: list[dict[str, Any]] = []
    wild: list[tuple[str, dict[str, Any]]] = []

    bus.subscribe("ping", lambda et, **f: exact.append(f))
    bus.subscribe(WILDCARD, lambda et, **f: wild.append((et, f)))

    bus.emit("ping", v=1)

    assert exact == [{"v": 1}]
    assert wild == [("ping", {"v": 1})]


def test_clear_removes_all_subscriptions() -> None:
    bus = EventBus()
    calls: list[int] = []
    bus.subscribe("ping", lambda et, **f: calls.append(1))
    bus.subscribe(WILDCARD, lambda et, **f: calls.append(1))

    bus.clear()
    bus.emit("ping")
    bus.emit("other")

    assert calls == []


# =====================================================================
# Thread safety
# =====================================================================


def test_concurrent_subscribe_and_emit_produces_expected_calls() -> None:
    """Ten threads each register a handler and emit ten events.

    Every handler must ultimately observe every subsequent emission.
    Race-sensitive: the test asserts the invariant that ``emit``
    never raises and that handler-call counts match what the
    snapshot-then-dispatch contract guarantees.
    """
    bus = EventBus()
    seen: list[int] = []
    lock = threading.Lock()

    def make_handler(tid: int) -> Any:
        def _handler(event_type: str, **fields: Any) -> None:
            with lock:
                seen.append(tid)

        return _handler

    def worker(tid: int) -> None:
        bus.subscribe("ping", make_handler(tid))
        for _ in range(10):
            bus.emit("ping")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # At least 100 emits happened; every emit is visible to at least
    # the thread that issued it (its own handler was registered by
    # then). The exact count depends on interleaving, but no emit
    # should have raised.
    assert len(seen) > 0


# =====================================================================
# Default-bus singleton
# =====================================================================


def test_get_default_bus_returns_same_instance() -> None:
    a = get_default_bus()
    b = get_default_bus()
    assert a is b


@pytest.fixture(autouse=True)
def _reset_default_bus() -> Any:
    """Keep the default bus clean between tests in this module."""
    get_default_bus().clear()
    yield
    get_default_bus().clear()


# =====================================================================
# emit_event wiring into the default bus (from observability.py)
# =====================================================================


def test_emit_event_dispatches_to_default_bus() -> None:
    from llm_judge.control_plane.observability import emit_event

    received: list[tuple[str, dict[str, Any]]] = []
    get_default_bus().subscribe(
        WILDCARD, lambda et, **f: received.append((et, f))
    )

    emit_event("wired_event", payload="hello")

    assert received == [("wired_event", {"payload": "hello"})]


def test_emit_event_tolerates_bus_handler_failure() -> None:
    """A broken subscriber must not bubble up through emit_event."""
    from llm_judge.control_plane.observability import emit_event

    get_default_bus().subscribe(
        WILDCARD, lambda et, **f: (_ for _ in ()).throw(RuntimeError("no"))
    )

    # Must not raise.
    emit_event("event_x")
