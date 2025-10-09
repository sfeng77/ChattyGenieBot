from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, TypedDict

ProgressEventType = Literal[
    "turn_started",
    "turn_finished",
    "turn_failed",
    "llm_started",
    "llm_finished",
    "tool_started",
    "tool_finished",
    "tool_error",
]


class ProgressEvent(TypedDict, total=False):
    type: ProgressEventType
    text: str
    meta: Dict[str, Any]


ProgressListener = Callable[[ProgressEvent], Awaitable[None]]


class ProgressDispatcher:
    """Manage listeners that consume progress events."""

    def __init__(self) -> None:
        self._listeners: List[ProgressListener] = []
        self._logger = logging.getLogger(__name__)

    async def emit(self, event: ProgressEvent) -> None:
        listeners = list(self._listeners)
        for listener in listeners:
            try:
                await listener(event)
            except Exception:  # noqa: BLE001
                self._logger.warning("Progress listener failed", exc_info=True)

    def add_listener(self, listener: ProgressListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def remove() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return remove


class NullProgressDispatcher(ProgressDispatcher):
    """Dispatcher that ignores all events."""

    def __init__(self) -> None:
        super().__init__()

    async def emit(self, event: ProgressEvent) -> None:  # noqa: D401
        return

    def add_listener(self, listener: ProgressListener) -> Callable[[], None]:  # noqa: D401
        return lambda: None


__all__ = [
    "ProgressDispatcher",
    "NullProgressDispatcher",
    "ProgressEvent",
    "ProgressEventType",
    "ProgressListener",
]
