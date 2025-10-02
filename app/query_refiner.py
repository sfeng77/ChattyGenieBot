import logging
from typing import Iterable

from app.ollama_client import OllamaClient
from app.state import Turn

LOGGER = logging.getLogger(__name__)

QUERY_REFINER_SYSTEM_PROMPT = (
    "You rewrite end-user requests into concise search queries that retrieve the most relevant documents. "
    "Return a single line query under 24 words, no numbering, no explanations."
)


async def refine_query(
    ollama: OllamaClient,
    user_text: str,
    history: Iterable[Turn],
) -> str:
    """Use the model to produce a better search query before retrieval."""
    recent_turns = list(history)[-4:]
    history_lines: list[str] = []
    for turn in recent_turns:
        prefix = "User" if turn.role == "user" else "Assistant"
        history_lines.append(f"{prefix}: {turn.content}")

    prompt_parts: list[str] = []
    if history_lines:
        prompt_parts.append("Recent conversation:\n" + "\n".join(history_lines))
    prompt_parts.append(f"User request:\n{user_text}")
    prompt_parts.append(
        "Rewrite the user request as a focused search query capturing the essential nouns and intent. "
        "Avoid pronouns and filler words."
    )
    prompt = "\n\n".join(prompt_parts)

    LOGGER.debug("Refining search query (chars=%s)", len(prompt))

    chunks: list[str] = []
    try:
        async for chunk in ollama.generate_stream(
            prompt=prompt,
            system_prompt=QUERY_REFINER_SYSTEM_PROMPT,
        ):
            chunks.append(chunk)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Query refinement failed: %s", exc)
        return user_text

    refined = "".join(chunks).strip()
    if not refined:
        LOGGER.debug("Refiner returned empty output; falling back to original text")
        return user_text

    first_line = refined.splitlines()[0].strip().strip('"')
    if not first_line:
        LOGGER.debug("Refiner produced no usable line; falling back to original text")
        return user_text

    truncated = first_line[:256]
    LOGGER.debug("Refined query: %s", truncated)
    return truncated


__all__ = ["refine_query"]
