from __future__ import annotations

import asyncio
from pathlib import Path


async def convert_audio_to_wav(ffmpeg_path: str, source: Path, target: Path) -> None:
    """Convert an audio file to 16kHz mono WAV using ffmpeg."""
    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        stderr_text = (stderr.decode("utf-8", errors="ignore") or "").strip()
        raise RuntimeError(f"ffmpeg failed with exit code {process.returncode}: {stderr_text}")
    if not target.exists():
        raise RuntimeError("ffmpeg did not produce an output file.")


__all__ = ["convert_audio_to_wav"]
