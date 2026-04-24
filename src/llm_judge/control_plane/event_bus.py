"""In-memory pub/sub event bus for Control Plane observability.

Subscribers register per ``event_type`` or under the wildcard
``"*"`` to receive every event. The demo CLI (:mod:`tools.demo_platform`)
relies on the wildcard subscription to render a live event feed.

Thread safety: subscribe/unsubscribe/emit acquire a single
``threading.Lock``. Handler exceptions are caught and logged via
:mod:`structlog` — a failing handler does NOT stop other handlers
for the same event and does NOT propagate to ``emit``'s caller.

Iteration order is registration order (no priorities).

Use :func:`get_default_bus` to get the process-wide singleton.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# A handler receives (event_type, **fields). Fields mirror whatever
# the emitter passed to :meth:`EventBus.emit`.
EventHandler = Callable[..., None]

WILDCARD: str = "*"


class EventBus:
    """Thread-safe in-memory pub/sub.

    ``subscribe(event_type, handler)`` registers a handler. A handler
    registered for :data:`WILDCARD` (``"*"``) receives every event.
    ``emit(event_type, **fields)`` dispatches to all matching
    handlers; exceptions raised by a handler are logged and swallowed.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register ``handler`` for ``event_type`` (or ``"*"`` for all)."""
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a previously-registered handler. No-op if not found."""
        with self._lock:
            handlers = self._handlers.get(event_type)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass  # handler was not registered — silent no-op
            if not handlers:
                # Remove the bucket so iterate doesn't see an empty list.
                self._handlers.pop(event_type, None)

    def emit(self, event_type: str, **fields: Any) -> None:
        """Dispatch ``event_type`` with ``fields`` to all subscribers.

        Calls handlers registered for ``event_type`` first, then
        wildcard handlers. Each handler runs in the caller's thread;
        exceptions are caught and logged.
        """
        # Snapshot under lock; invoke handlers outside the lock so a
        # slow handler cannot block subscribe/unsubscribe.
        with self._lock:
            exact = list(self._handlers.get(event_type, ()))
            wild = list(self._handlers.get(WILDCARD, ()))

        for handler in exact + wild:
            try:
                handler(event_type, **fields)
            except Exception:
                logger.exception(
                    "event_bus.handler_raised",
                    event_type=event_type,
                    handler=getattr(handler, "__name__", repr(handler)),
                )

    def clear(self) -> None:
        """Remove every subscription. Used by tests for isolation."""
        with self._lock:
            self._handlers.clear()


_default_bus: EventBus | None = None
_default_bus_lock = threading.Lock()


def get_default_bus() -> EventBus:
    """Return the process-wide default :class:`EventBus` singleton.

    Lazy-initialised under a lock so concurrent first-callers see the
    same instance. Tests can call ``get_default_bus().clear()`` to
    reset subscriber state between cases.
    """
    global _default_bus
    if _default_bus is None:
        with _default_bus_lock:
            if _default_bus is None:
                _default_bus = EventBus()
    return _default_bus
