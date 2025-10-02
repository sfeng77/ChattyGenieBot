from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Literal

Role = Literal["user", "assistant"]


@dataclass
class Turn:
    role: Role
    content: str


class ConversationStore:
    def __init__(self, max_turns: int) -> None:
        self._max_turns = max_turns
        self._store: Dict[int, Deque[Turn]] = {}

    def get(self, chat_id: int) -> List[Turn]:
        return list(self._store.get(chat_id, deque()))

    def add(self, chat_id: int, role: Role, content: str) -> None:
        queue = self._store.setdefault(chat_id, deque(maxlen=self._max_turns))
        queue.append(Turn(role=role, content=content))

    def reset(self, chat_id: int) -> None:
        self._store.pop(chat_id, None)

    def all(self) -> Iterable[tuple[int, List[Turn]]]:
        for chat_id, turns in self._store.items():
            yield chat_id, list(turns)


__all__ = ["ConversationStore", "Turn", "Role"]
