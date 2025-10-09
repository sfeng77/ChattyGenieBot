from __future__ import annotations

import base64
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from app.config import Settings

LOGGER = logging.getLogger(__name__)


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

    input_messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_base64": encoded},
            ],
        },
    ]

    LOGGER.debug(
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
            response = await client.responses.create(
                model=settings.vision_model,
                input=input_messages,
                temperature=request_temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            LOGGER.info("Vision responses endpoint failed; attempting chat fallback", extra={"error_type": type(exc).__name__})
            fallback_text = await _run_chat_fallback(
                client=client,
                model=settings.vision_model,
                system_prompt=system_prompt,
                prompt_text=prompt_text,
                data_url=data_url,
                temperature=request_temperature,
                max_output_tokens=max_output_tokens,
                original_exception=exc,
            )
            if fallback_text is not None:
                return fallback_text
            raise

    return _extract_text_from_response(response)


def _extract_text_from_response(response) -> str:
    for item in response.output or []:
        for piece in item.get("content", []):
            if piece.get("type") in {"output_text", "text"}:
                text = (piece.get("text") or "").strip()
                if text:
                    return text
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    raise RuntimeError("vision model returned no text output")


async def _run_chat_fallback(
    *,
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    prompt_text: str,
    data_url: str,
    temperature: float,
    max_output_tokens: int,
    original_exception: Exception,
) -> Optional[str]:
    """Try chat.completions as a fallback when responses.create is unavailable."""
    status = None
    if isinstance(original_exception, httpx.HTTPStatusError):
        status = original_exception.response.status_code
    elif hasattr(original_exception, "status_code"):
        status = getattr(original_exception, "status_code")

    message = str(original_exception)
    if status not in {404, 405, 501} and "404" not in message and "Not Found" not in message:
        return None

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
        LOGGER.info("Chat fallback failed", extra={"error_type": type(exc).__name__})
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
