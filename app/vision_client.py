from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from app.config import Settings

LOGGER = logging.getLogger(__name__)


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort HTTP status extraction for logging."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        response_code = getattr(response, "status_code", None)
        if isinstance(response_code, int):
            return response_code
    return None


async def analyze_image(
    image_bytes: bytes,
    caption: Optional[str],
    settings: Settings,
    *,
    mime_type: str = "image/jpeg",
    temperature: Optional[float] = None,
    max_output_tokens: int = 512,
) -> str:
    """Call the configured vision model and return the textual answer."""
    prompt_text = (caption or "Describe this image.").strip() or "Describe this image."
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    system_prompt = settings.vision_system_prompt.strip() or "You are a concise vision assistant."
    request_temperature = temperature if temperature is not None else settings.vision_temperature

    LOGGER.info(
        "Calling vision model",
        extra={
            "model": settings.vision_model,
            "temperature": request_temperature,
            "prompt_preview": prompt_text[:80],
            "image_bytes": len(image_bytes),
        },
    )

    client_params = {"api_key": settings.openai_api_key, "timeout": settings.vision_timeout}
    if settings.openai_api_base:
        client_params["base_url"] = settings.openai_api_base

    async with AsyncOpenAI(**client_params) as client:
        try:
            fallback_text = await _run_chat_fallback(
                client=client,
                model=settings.vision_model,
                system_prompt=system_prompt,
                prompt_text=prompt_text,
                data_url=data_url,
                temperature=request_temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Vision model request failed", exc_info=True)        

    return fallback_text

async def _run_chat_fallback(
    *,
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    prompt_text: str,
    data_url: str,
    temperature: float,
    max_output_tokens: int,
) -> Optional[str]:
    """Try chat.completions as a fallback when responses.create is unavailable."""

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    try:
        chat = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
    except Exception as exc:
        fallback_status = _extract_status_code(exc)
        LOGGER.exception(
            "Chat fallback failed (status=%s): %s",
            fallback_status if fallback_status is not None else "n/a",
            exc,
            extra={
                "error_type": type(exc).__name__,
                "status_code": fallback_status,
                "error_message": str(exc),
            },
            exc_info=True,
        )
        return None

    for choice in chat.choices or []:
        message_obj = getattr(choice, "message", None)
        if message_obj is None:
            continue
        content = getattr(message_obj, "content", None)
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
            if parts:
                return " ".join(parts).strip()
    return None
