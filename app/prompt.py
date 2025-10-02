from typing import Iterable, List, Sequence

from app.rag.retriever import Retrieval
from app.state import Turn

BASE_SYSTEM_PROMPT = (
    "You are Chatty Genie, a concise assistant that answers questions using the provided context. "
    "If the retrieval context is empty or does not contain the answer, say you do not know. "
    "Format code or commands using markdown fenced blocks when helpful."
)


def render_context(retrievals: Sequence[Retrieval]) -> str:
    if not retrievals:
        return ""
    lines: List[str] = ["Relevant context:"]
    for item in retrievals:
        label = f"{item.doc_path}#chunk{item.chunk_index}"
        lines.append(f"- ({item.score:.2f}) {label}: {item.text}")
    return "\n".join(lines)


def render_history(history: Iterable[Turn]) -> str:
    lines: List[str] = []
    for turn in history:
        if turn.role == "user":
            lines.append(f"User: {turn.content}")
        else:
            lines.append(f"Assistant: {turn.content}")
    return "\n".join(lines)


def build_prompt(history: Iterable[Turn], user_message: str, retrievals: Sequence[Retrieval]) -> tuple[str, str]:
    context_block = render_context(retrievals)
    history_block = render_history(history)
    parts: List[str] = []
    if context_block:
        parts.append(context_block)
    if history_block:
        parts.append(history_block)
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    prompt = "\n\n".join(parts)
    return BASE_SYSTEM_PROMPT, prompt


__all__ = ["BASE_SYSTEM_PROMPT", "build_prompt"]
