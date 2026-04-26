"""Observability primitives for the Control Plane.

Three things ship here:

  * :class:`Timer` — a context manager that measures elapsed wall
    time in milliseconds. Uses ``time.perf_counter`` for monotonic
    readings. ``duration_ms`` is available once the block exits.

  * :func:`timed` — a decorator that wraps a callable in a Timer
    and emits ``<event>_started`` / ``<event>_completed`` /
    ``<event>_failed`` events around it.

  * :func:`emit_event` — the single emission point for Control
    Plane events. Commit 1 logs through structlog only; Commit 2
    extends it to also dispatch to the in-memory ``EventBus``.

Event naming: every emitted event is a bare string (no dots, no
namespaces). Field names use snake_case. Field values must be
JSON-serialisable (str, int, float, bool, None, list, dict).
"""

from __future__ import annotations

import functools
import time
from types import TracebackType
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger()

F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """Context manager that records elapsed milliseconds.

    Usage::

        with Timer() as t:
            do_work()
        # t.duration_ms is populated

    Accessing :attr:`duration_ms` before ``__exit__`` runs returns
    ``0.0``. This is deliberate: the Timer is often used inside a
    try/except where the caller may inspect the value even if the
    block raised; returning ``0.0`` keeps that path quiet while
    still making it clear that no useful measurement happened.
    """

    __slots__ = ("_start", "_duration_ms")

    def __init__(self) -> None:
        self._start: float | None = None
        self._duration_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._start is not None:
            self._duration_ms = (time.perf_counter() - self._start) * 1000.0
        # Return None (falsy) so any exception propagates.

    @property
    def duration_ms(self) -> float:
        """Elapsed milliseconds. ``0.0`` until the ``with`` block exits."""
        return self._duration_ms

    @property
    def elapsed_ms(self) -> float:
        """Live elapsed milliseconds (readable mid-block).

        ``0.0`` before ``__enter__`` runs. Inside a ``with`` block,
        returns the wall time since entry. After ``__exit__``, returns
        the final locked-in duration — identical to :attr:`duration_ms`
        once the block has exited. Useful when a caller needs to
        report progress without closing the Timer.
        """
        if self._start is None:
            return 0.0
        if self._duration_ms > 0.0:
            return self._duration_ms
        return (time.perf_counter() - self._start) * 1000.0


def _extract_request_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Look for a request_id on the wrapped function's arguments.

    Checks (in order): ``kwargs["request_id"]``, first positional
    argument's ``request_id`` attribute, first positional argument's
    ``payload.request_id`` attribute. Returns ``None`` if not found.
    Used by :func:`timed` to tag emitted events with a correlation id.
    """
    rid = kwargs.get("request_id")
    if isinstance(rid, str):
        return rid
    if args:
        first = args[0]
        candidate = getattr(first, "request_id", None)
        if isinstance(candidate, str):
            return candidate
        payload = getattr(first, "payload", None)
        if payload is not None:
            candidate = getattr(payload, "request_id", None)
            if isinstance(candidate, str):
                return candidate
    return None


def timed(
    event_name: str,
    *,
    sub_capability_id: str | None = None,
    capability_id: str | None = None,
) -> Callable[[F], F]:
    """Decorator factory that times a function and emits events.

    Default mode (``sub_capability_id`` not set) emits three events
    per call:
      * ``<event_name>_started`` on entry (fields: request_id if found)
      * ``<event_name>_completed`` on successful return
        (fields: duration_ms, request_id)
      * ``<event_name>_failed`` on exception
        (fields: duration_ms, request_id, error_type, error_message),
        then re-raises.

    Sub-capability mode (``sub_capability_id`` set) emits the
    ``sub_capability_started`` / ``sub_capability_completed`` /
    ``sub_capability_failed`` event family. ``capability_id`` is
    required in this mode and is included on every emission alongside
    ``sub_capability_id``. The ``event_name`` argument is ignored in
    sub-capability mode (the event names are fixed); pass any string.

    The wrapped function's return value and signature are preserved.
    """
    if sub_capability_id is not None and not capability_id:
        raise ValueError(
            "timed: capability_id is required when sub_capability_id is set"
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = _extract_request_id(args, kwargs)
            sub_mode = sub_capability_id is not None

            if sub_mode:
                started_event = "sub_capability_started"
                completed_event = "sub_capability_completed"
                failed_event = "sub_capability_failed"
                base_fields: dict[str, Any] = {
                    "capability_id": capability_id,
                    "sub_capability_id": sub_capability_id,
                }
            else:
                started_event = f"{event_name}_started"
                completed_event = f"{event_name}_completed"
                failed_event = f"{event_name}_failed"
                base_fields = {}

            start_fields: dict[str, Any] = dict(base_fields)
            if request_id is not None:
                start_fields["request_id"] = request_id
            emit_event(started_event, **start_fields)

            timer = Timer()
            try:
                with timer:
                    result = func(*args, **kwargs)
            except Exception as exc:
                fail_fields: dict[str, Any] = dict(base_fields)
                fail_fields["duration_ms"] = timer.duration_ms
                fail_fields["status"] = "failure"
                fail_fields["error_type"] = type(exc).__name__
                fail_fields["error_message"] = str(exc)[:500]
                if request_id is not None:
                    fail_fields["request_id"] = request_id
                emit_event(failed_event, **fail_fields)
                raise

            done_fields: dict[str, Any] = dict(base_fields)
            done_fields["duration_ms"] = timer.duration_ms
            if sub_mode:
                done_fields["status"] = "success"
            if request_id is not None:
                done_fields["request_id"] = request_id
            emit_event(completed_event, **done_fields)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def emit_sub_capability_skipped(
    *,
    capability_id: str,
    sub_capability_id: str,
    request_id: str,
    reason: str,
) -> None:
    """Emit a ``sub_capability_skipped`` event.

    Used to mark sub-capabilities that are not engaged in the current
    scenario (e.g. CAP-1 Discovery in single-eval). The ``reason``
    field is a short snake_case slug documenting why the sub-cap did
    not run; aggregation tools key off it to surface 0/N engagement
    in batch reports.
    """
    emit_event(
        "sub_capability_skipped",
        capability_id=capability_id,
        sub_capability_id=sub_capability_id,
        request_id=request_id,
        reason=reason,
    )


def emit_event(event_type: str, **fields: Any) -> None:
    """Emit a Control Plane event.

    Logs the event through structlog at INFO level, then dispatches
    to the in-memory :class:`~llm_judge.control_plane.event_bus.EventBus`
    so local subscribers (e.g. the demo CLI) can observe the stream.

    Best-effort semantics: if the bus dispatch raises, the failure is
    logged and swallowed — emission is observability, not a
    control-path concern.
    """
    logger.info(event_type, **fields)
    # Lazy import so Commit 1 users who do not touch the bus path
    # still work; in normal Control Plane use the bus is always
    # available.
    try:
        from llm_judge.control_plane.event_bus import get_default_bus

        get_default_bus().emit(event_type, **fields)
    except Exception:
        logger.exception(
            "observability.emit_event.bus_dispatch_failed",
            event_type=event_type,
        )
