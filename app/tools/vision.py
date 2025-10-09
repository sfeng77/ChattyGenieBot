from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Dict, Optional

import httpx
from agents import function_tool
from pydantic import Field

from app.config import Settings
from app.vision_client import analyze_image
from app.vision_image import prepare_image

LOGGER = logging.getLogger(__name__)

FileIdArg = Annotated[
    str,
    Field(min_length=5, max_length=128, description="Telegram file identifier for the image."),
]
CaptionArg = Annotated[
    Optional[str],
    Field(description="Optional caption or user question about the image."),
]


async def _fetch_file_bytes(token: str, file_id: str, timeout: float) -> tuple[bytes, str]:
    base_url = f"https://api.telegram.org/bot{token}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        meta_resp = await client.get(f"{base_url}/getFile", params={"file_id": file_id})
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        if not meta.get("ok"):
            raise RuntimeError(f"Telegram getFile failed: {meta}")
        result = meta.get("result") or {}
        file_path = result.get("file_path")
        if not file_path:
            raise RuntimeError("Telegram response missing file_path")
        download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
        file_resp = await client.get(download_url)
        file_resp.raise_for_status()
        return file_resp.content, result.get("file_unique_id") or file_id


def create_disabled_vision_tool(message: str | None = None):
    notice = message or "vision_analyze is disabled for this deployment."

    @function_tool(name_override="vision_analyze")
    async def vision_analyze(file_id: FileIdArg, caption: CaptionArg = None) -> Dict[str, object]:
        LOGGER.debug("vision_analyze requested while disabled", extra={"file_id": file_id})
        return {"error": notice, "file_id": file_id}

    return vision_analyze


def create_vision_tool(settings: Settings):
    telegram_token = settings.telegram_bot_token
    timeout = max(settings.vision_timeout, 1.0)
    max_edge = max(int(settings.vision_max_edge or 0), 0)

    @function_tool(name_override="vision_analyze")
    async def vision_analyze(file_id: FileIdArg, caption: CaptionArg = None) -> Dict[str, object]:
        try:
            raw_bytes, unique_id = await _fetch_file_bytes(telegram_token, file_id, timeout)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to download telegram file", extra={"file_id": file_id})
            return {
                "error": f"Unable to download the image: {exc}",
                "file_id": file_id,
            }

        try:
            prepared_bytes, (width, height), mime_type = await asyncio.to_thread(
                prepare_image,
                raw_bytes,
                max_edge,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to prepare image", extra={"file_id": file_id})
            return {
                "error": f"Unable to process the image: {exc}",
                "file_id": file_id,
            }

        try:
            answer = await analyze_image(prepared_bytes, caption, settings, mime_type=mime_type)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Vision model call failed", extra={"file_id": file_id})
            return {
                "error": f"vision model error: {exc}",
                "file_id": file_id,
            }

        return {
            "text": answer,
            "model": settings.vision_model,
            "dimensions": {"width": width, "height": height},
            "mime_type": mime_type,
            "note": "Image descriptions may be approximate.",
            "file_id": unique_id,
        }

    return vision_analyze


__all__ = ["create_vision_tool", "create_disabled_vision_tool"]
