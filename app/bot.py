import asyncio
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Sequence, Set

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update, User, Voice
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.error import BadRequest

from app.agent_runtime import AgentRuntime
from app.config import Settings
from app.asr import ASRService, TranscriptionResult, create_asr_service
from app.progress import ProgressDispatcher, ProgressEvent

LOGGER = logging.getLogger(__name__)

MAX_PROGRESS_LINES = 40
MAX_PROGRESS_CHARS = 3000


SETTINGS_KEY = "settings"
AGENT_RUNTIME_KEY = "agent_runtime"
WHITELIST_KEY = "whitelist_user_ids"
BOT_ID_KEY = "bot_id"
BOT_USERNAME_KEY = "bot_username"
ASR_SERVICE_KEY = "asr_service"
FFMPEG_PATH_KEY = "ffmpeg_path"

HandlerFunc = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


async def _should_respond(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    mention_text: str | None = None,
    *,
    require_reply: bool = False,
    
) -> bool:
    """Decide whether the bot should respond in this context.

    Preserves existing behavior across handlers:
    - respond in private chats, or when mentioned, or when replied to.
    """
    # Determine chat type safely
    is_private_chat = False
    try:
        is_private_chat = bool(message.chat and message.chat.type == ChatType.PRIVATE)
    except Exception:
        is_private_chat = False

    bot_id, bot_username = await _ensure_bot_identity(context)

    replied_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    )

    if require_reply:
        return replied_to_bot

    mentioned = False
    if mention_text:
        lowered = mention_text.lower()
        if bot_username:
            mentioned = f"@{bot_username.lower()}" in lowered

    return bool(is_private_chat or mentioned or replied_to_bot)


def _resolve_ffmpeg_path(candidate: str | None) -> str | None:
    if not candidate:
        return None
    candidate = candidate.strip()
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_file():
        return str(path)
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    return None


def _truncate(text: str, limit: int = 160) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _escape_markdown_url(url: str) -> str:
    """Escape characters in URLs that break Telegram Markdown parsing."""
    return (
        url.replace("\\", "")
        .replace(")", r"\)")
        .replace("(", r"\(")
        .replace("_", r"\_")
    )


_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
_DOUBLE_UNDERSCORE_PATTERN = re.compile(r"__(.+?)__")


def _normalize_markdown_for_telegram(text: str) -> str:
    """Convert common Markdown to Telegram legacy Markdown-compatible form."""
    def replace_bold(match: re.Match[str]) -> str:
        return f"*{match.group(1)}*"

    text = _BOLD_PATTERN.sub(replace_bold, text)
    text = _DOUBLE_UNDERSCORE_PATTERN.sub(r"_\1_", text)
    return text


def _prepare_agent_message(text: str, citations: Sequence[str] | None = None) -> str:
    """Append missing citations and normalize formatting for Telegram Markdown."""
    message = text.strip()
    unique_urls: list[str] = []
    if citations:
        for url in citations:
            clean = (url or "").strip()
            if not clean or clean in unique_urls:
                continue
            unique_urls.append(clean)
    if unique_urls:
        missing: list[tuple[int, str]] = []
        for idx, url in enumerate(unique_urls, start=1):
            marker = f"[{idx}]("
            if marker not in message:
                missing.append((idx, url))
        if missing:
            footer_line = ", ".join(f"[{idx}]({_escape_markdown_url(url)})" for idx, url in missing)
            message = f"{message.rstrip()}\n\nSources / 来源: {footer_line}"
    return _normalize_markdown_for_telegram(message)


def _is_markdown_parse_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "markdown" in lowered or "parse" in lowered


async def _send_markdown_with_fallback(
    send_func: Callable[..., Awaitable[Any]],
    text: str,
    *,
    raw_text: str | None = None,
    disable_preview: bool = False,
    **kwargs: Any,
) -> None:
    fallback_text = raw_text if raw_text is not None else text
    try:
        await send_func(
            text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=disable_preview,
            **kwargs,
        )
    except BadRequest as exc:
        if _is_markdown_parse_error(exc):
            LOGGER.warning("Markdown parse failed, retrying without formatting: %s", exc)
            await send_func(fallback_text, disable_web_page_preview=disable_preview, **kwargs)
        else:
            raise


def _format_timeline(timeline: List[str]) -> str:
    if not timeline:
        return "Thinking..."
    return "Thinking...\n" + "\n".join(timeline)


def _is_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    whitelist: Set[int] = context.application.bot_data.get(WHITELIST_KEY, set())
    if not whitelist:
        return True
    user = update.effective_user
    if user is None:
        return False
    return user.id in whitelist


async def _deny_access(update: Update) -> None:
    message = update.effective_message
    if message is not None:
        await message.reply_text("Sorry, you are not authorized to use this bot.")


def require_authorized(fn: HandlerFunc) -> HandlerFunc:
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _is_authorized(update, context):
            await _deny_access(update)
            return
        await fn(update, context)

    return wrapper


@require_authorized
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.info("/start invoked by chat_id=%s", message.chat_id)
    # Log the command
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    from_user = message.from_user
    runtime.log_message(
        message.chat_id,
        content=message.text or "/start",
        sender_id=str(from_user.id) if from_user and from_user.id is not None else None,
        created_at=message.date,
        metadata={"telegram_message_id": message.id},
    )
    await message.reply_text("Hi! I am Agent Mushroom. Send me a message and I will reply using OpenAI Agents.")


@require_authorized
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.info("/help invoked by chat_id=%s", message.chat_id)
    # Log the command
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    from_user = message.from_user
    runtime.log_message(
        message.chat_id,
        content=message.text or "/help",
        sender_id=str(from_user.id) if from_user and from_user.id is not None else None,
        created_at=message.date,
        metadata={"telegram_message_id": message.id},
    )
    await message.reply_text(
        "Commands:\n"
        "/start - welcome message\n"
        "/reset - clear conversation memory\n"
        "/progress - toggle live progress updates for this chat\n"
        "/recap - summarize last 1h/1d of this chat"
    )


@require_authorized
async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.info("/reset invoked by chat_id=%s", message.chat_id)
    # Log the command
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    from_user = message.from_user
    runtime.log_message(
        message.chat_id,
        content=message.text or "/reset",
        sender_id=str(from_user.id) if from_user and from_user.id is not None else None,
        created_at=message.date,
        metadata={"telegram_message_id": message.id},
    )
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    await runtime.reset(message.chat_id)
    await message.reply_text("Conversation memory cleared.")


@require_authorized
async def toggle_progress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    current = bool(context.chat_data.get("progress_enabled", False))
    new_state = not current
    context.chat_data["progress_enabled"] = new_state
    status = "enabled" if new_state else "disabled"
    # Log the command
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    from_user = message.from_user
    runtime.log_message(
        message.chat_id,
        content=message.text or "/progress",
        sender_id=str(from_user.id) if from_user and from_user.id is not None else None,
        created_at=message.date,
        metadata={"telegram_message_id": message.id, "progress_enabled": new_state},
    )
    await message.reply_text(f"Progress updates {status} for this chat.")


@require_authorized
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.photo:
        return
    settings: Settings = context.application.bot_data[SETTINGS_KEY]
    if not settings.vision_enabled:
        await message.reply_text("Image analysis is disabled for this deployment.")
        return

    photo = message.photo[-1]
    file_id = photo.file_id
    caption = (message.caption or "").strip()

    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    chat_id = message.chat_id
    from_user = message.from_user
    sender_id: str | None = str(from_user.id) if from_user and from_user.id is not None else None

    user_lines = ["User sent an image.", f"telegram_file_id={file_id}"]
    if caption:
        user_lines.append(f"Caption: {caption}")
    else:
        user_lines.append("Caption: (none provided)")
    user_lines.append("Use the vision_analyze tool if you need to inspect the image before answering.")
    agent_input = "\n".join(user_lines)
    # Log every photo message
    runtime.log_message(
        chat_id,
        content=agent_input,
        sender_id=sender_id,
        created_at=message.date,
        metadata={
            "telegram_message_id": message.id,
            "caption": caption or "",
        },
    )

    # Decide whether to respond
    if not await _should_respond(message, context, mention_text=caption):
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    placeholder = await message.reply_text("Analyzing image...")

    try:
        LOGGER.info("Running vision agent with input:\n%s", agent_input)
        response = await runtime.run_message(chat_id, agent_input, sender_id=sender_id, log_user=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Agent vision run failed")
        await placeholder.edit_text(f"Agent error: {exc}")
        return

    final_text = (response or "").strip() or "I am sorry, I could not describe that image."
    prepared_text = _prepare_agent_message(final_text, [])
    try:
        await _send_markdown_with_fallback(
            placeholder.edit_text,
            prepared_text,
            raw_text=final_text,
            disable_preview=False,
        )
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to edit vision placeholder", exc_info=True)
        try:
            await _send_markdown_with_fallback(
                message.reply_text,
                prepared_text,
                raw_text=final_text,
                disable_preview=False,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to send vision reply fallback", exc_info=True)
            await message.reply_text(final_text)


def _recap_window(period: str) -> tuple[str, datetime]:
    p = (period or "").strip().lower()
    if p == "1h":
        return "the last 1 hour", datetime.now(timezone.utc) - timedelta(hours=1)
    if p == "1d":
        return "the last 1 day", datetime.now(timezone.utc) - timedelta(days=1)
    raise ValueError("unsupported period")


@require_authorized
async def recap_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.info("/recap invoked by chat_id=%s", message.chat_id)
    # Log the command
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    from_user = message.from_user
    runtime.log_message(
        message.chat_id,
        content=message.text or "/recap",
        sender_id=str(from_user.id) if from_user and from_user.id is not None else None,
        created_at=message.date,
        metadata={"telegram_message_id": message.id},
    )

    text = (message.text or "").strip()
    parts = text.split()
    period: str | None = parts[1].lower() if len(parts) >= 2 else None

    if period not in {"1h", "1d"}:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("1 hour", callback_data="recap:1h"),
                    InlineKeyboardButton("1 day", callback_data="recap:1d"),
                ]
            ]
        )
        await message.reply_text("Choose recap period:", reply_markup=keyboard)
        return

    try:
        label, start = _recap_window(period)
    except ValueError:
        await message.reply_text("Unsupported period. Use 1h or 1d.")
        return

    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    placeholder = await message.reply_text(f"Summarizing {label}...")

    try:
        summary = await runtime.recap_history(message.chat_id, start=start)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Recap generation failed")
        await placeholder.edit_text(f"Recap error: {exc}")
        return

    if not summary:
        final_text = f"No messages in {label}."
    else:
        final_text = f"Recap ({label}):\n\n{summary}"

    try:
        await placeholder.edit_text(final_text)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to edit recap message", exc_info=True)
        await message.reply_text(final_text)


@require_authorized
async def handle_recap_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    data = (query.data or "").strip()
    if not data.startswith("recap:"):
        await query.answer()
        return
    period = data.split(":", 1)[1]
    try:
        label, start = _recap_window(period)
    except ValueError:
        await query.answer(text="Unsupported period", show_alert=False)
        return

    await query.answer()
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    # Log the action similar to command
    try:
        runtime.log_message(
            query.message.chat_id,
            content=f"/recap {period}",
            sender_id=str(getattr(update.effective_user, "id", "")) or None,
        )
    except Exception:
        LOGGER.exception("Failed to log recap callback", exc_info=True)

    await context.bot.send_chat_action(chat_id=query.message.chat_id, action=ChatAction.TYPING)
    try:
        summary = await runtime.recap_history(query.message.chat_id, start=start)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Recap generation failed")
        await query.message.reply_text(f"Recap error: {exc}")
        return

    if not summary:
        text = f"No messages in {label}."
    else:
        text = f"Recap ({label}):\n\n{summary}"
    await query.message.reply_text(text)


@require_authorized
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.voice:
        return

    settings: Settings = context.application.bot_data[SETTINGS_KEY]
    if not settings.asr_enabled:
        await message.reply_text("Voice transcription is disabled for this deployment.")
        return

    asr_service: ASRService | None = context.application.bot_data.get(ASR_SERVICE_KEY)
    if asr_service is None:
        await message.reply_text("Voice transcription service is unavailable.")
        return

    ffmpeg_path: str | None = context.application.bot_data.get(FFMPEG_PATH_KEY)
    if not ffmpeg_path:
        await message.reply_text("Cannot transcribe audio because ffmpeg is not available on the server.")
        return

    voice = message.voice
    duration = int(getattr(voice, "duration", 0) or 0)
    if settings.max_audio_duration_s > 0 and duration > settings.max_audio_duration_s:
        await message.reply_text(
            f"Voice note too long ({duration}s). Limit is {settings.max_audio_duration_s} seconds."
        )
        return

    file_size = int(getattr(voice, "file_size", 0) or 0)
    if settings.max_audio_size_mb > 0:
        max_bytes = settings.max_audio_size_mb * 1024 * 1024
        if file_size and file_size > max_bytes:
            size_mb = file_size / (1024 * 1024)
            await message.reply_text(
                f"Voice note too large ({size_mb:.1f} MB). Limit is {settings.max_audio_size_mb} MB."
            )
            return

    placeholder = await message.reply_text("Transcribing...")

    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    from_user = message.from_user
    sender_id: str | None = str(from_user.id) if from_user and from_user.id is not None else None

    try:
        transcription = await _transcribe_voice_note(voice, ffmpeg_path, asr_service)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Voice transcription failed")
        try:
            await placeholder.edit_text(f"Transcription failed: {exc}")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to edit transcription failure message", exc_info=True)
            await message.reply_text(f"Transcription failed: {exc}")
        return

    transcript_text = (transcription.text or "").strip()
    has_transcript = bool(transcript_text)
    if not has_transcript:
        transcript_text = "(no speech detected)"

    display_name = _format_user_display_name(from_user)
    display_text = f"[{display_name}] 说: {transcript_text}"

    if settings.transcribe_echo_enabled:
        try:
            await placeholder.edit_text(display_text)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to edit transcription placeholder", exc_info=True)
            await message.reply_text(display_text)
    else:
        try:
            await context.bot.delete_message(chat_id=message.chat_id, message_id=placeholder.message_id)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to delete transcription placeholder", exc_info=True)

    metadata: dict[str, Any] = {
        "telegram_message_id": message.id,
        "voice_file_id": voice.file_id,
        "voice_mime_type": voice.mime_type,
        "voice_duration": duration,
        "voice_file_size": file_size,
        "asr_language": transcription.language,
        "asr_duration": transcription.duration,
        "asr_status": "ok" if has_transcript else "empty",
    }
    if transcription.segments:
        metadata["asr_segments_preview"] = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in transcription.segments[:5]
        ]

    runtime.log_message(
        message.chat_id,
        content=transcript_text,
        sender_id=sender_id,
        created_at=message.date,
        metadata=metadata,
    )

    # For voice, preserve behavior: only respond when replying to the bot
    reply_to_bot = await _should_respond(message, context, require_reply=True)
    if not (reply_to_bot and has_transcript):
        return

    await _run_agent_turn(message, context, runtime, settings, transcript_text, sender_id)


@require_authorized
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return
    chat_id = message.chat_id
    from_user = message.from_user
    sender_id: str | None = str(from_user.id) if from_user and from_user.id is not None else None
    user_text = message.text.strip()
    LOGGER.info("Received text message (chat_id=%s, chars=%s)", chat_id, len(user_text))
    if not user_text:
        await message.reply_text("Please send some text.")
        return

    settings: Settings = context.application.bot_data[SETTINGS_KEY]
    if len(user_text) > settings.max_input_chars:
        LOGGER.info("Message exceeds max_input_chars=%s", settings.max_input_chars)
        await message.reply_text(f"Message too long. Limit is {settings.max_input_chars} characters.")
        return

    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    # Log every text message first
    runtime.log_message(
        chat_id,
        content=user_text,
        sender_id=sender_id,
        created_at=message.date,
        metadata={
            "telegram_message_id": message.id,
            "reply_to_message_id": getattr(message.reply_to_message, "id", None),
        },
    )

    # Decide whether to respond (private, mention, or reply)
    if not await _should_respond(message, context, mention_text=user_text):
        return

    await _run_agent_turn(message, context, runtime, settings, user_text, sender_id)



def _parse_whitelist(raw: str | None) -> Set[int]:
    if not raw:
        return set()
    items = set()
    for piece in raw.split(','):
        candidate = piece.strip()
        if not candidate:
            continue
        try:
            items.add(int(candidate))
        except ValueError:
            LOGGER.warning("Ignoring invalid telegram user id in whitelist: %s", candidate)
    return items


async def _convert_audio_to_wav(ffmpeg_path: str, source: Path, target: Path) -> None:
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


async def _transcribe_voice_note(voice: Voice, ffmpeg_path: str, asr_service: ASRService) -> TranscriptionResult:
    telegram_file = await voice.get_file()
    with tempfile.TemporaryDirectory(prefix="cg-voice-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        source_path = temp_dir_path / "input"
        target_path = temp_dir_path / "converted.wav"
        await telegram_file.download_to_drive(custom_path=str(source_path))
        await _convert_audio_to_wav(ffmpeg_path, source_path, target_path)
        result = await asyncio.to_thread(asr_service.transcribe, str(target_path))
    return result


def _format_user_display_name(user: User | None) -> str:
    if user is None:
        return "user"
    if user.username:
        return user.username
    full_name = (user.full_name or "").strip()
    if full_name:
        return full_name
    parts = [p for p in [user.first_name, user.last_name] if p]
    if parts:
        return " ".join(parts)
    if user.id:
        return str(user.id)
    return "user"


async def _run_agent_turn(
    message: Message,
    context: ContextTypes.DEFAULT_TYPE,
    runtime: AgentRuntime,
    settings: Settings,
    user_text: str,
    sender_id: str | None,
) -> None:
    progress_enabled = bool(context.chat_data.get("progress_enabled", False))

    await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
    placeholder = await message.reply_text("Thinking...")

    dispatcher = ProgressDispatcher()
    throttle_seconds = max(settings.progress_edit_throttle_ms / 1000.0, 0.1)
    keep_timeline = settings.progress_keep_timeline
    last_edit = 0.0
    final_text: str | None = None
    error_text: str | None = None
    citation_urls: List[str] = []

    def _record_citations(meta: dict[str, Any]) -> None:
        urls: List[str] = []
        meta_urls = meta.get("results_urls")
        if isinstance(meta_urls, (list, tuple)):
            for candidate in meta_urls:
                if isinstance(candidate, str):
                    urls.append(candidate)
        top_url = meta.get("top_url")
        if isinstance(top_url, str):
            urls.insert(0, top_url)
        for url in urls:
            clean = url.strip()
            if clean and clean not in citation_urls:
                citation_urls.append(clean)

    def _timeline_line(meta: dict[str, Any] | None, label: str, body: str) -> str:
        elapsed: str | None = None
        if meta and isinstance(meta.get("elapsed_seconds"), (int, float)):
            elapsed = f"{meta['elapsed_seconds']:.2f}s"
        parts = [p for p in [elapsed, label, body] if p]
        return " | ".join(parts)

    timeline: List[str] = []
    if progress_enabled:
        timeline.append(_timeline_line({"elapsed_seconds": 0.0}, "[user]", _truncate(user_text, 200)))

    def _trim_timeline() -> None:
        if not progress_enabled:
            return
        while len(timeline) > MAX_PROGRESS_LINES:
            if len(timeline) <= 1:
                break
            timeline.pop(1)
        while len("\n".join(timeline)) > MAX_PROGRESS_CHARS and len(timeline) > 1:
            timeline.pop(1)

    async def apply_edit(force: bool = False) -> None:
        if not progress_enabled:
            return
        nonlocal last_edit
        now = time.monotonic()
        if not force and (now - last_edit) < throttle_seconds:
            return
        try:
            await placeholder.edit_text(_format_timeline(timeline))
            last_edit = now
        except Exception:  # noqa: BLE001
            LOGGER.exception("Progress edit failed", exc_info=True)

    async def on_progress(event: ProgressEvent) -> None:
        nonlocal final_text, error_text
        event_type = event.get("type")
        meta = (event.get("meta") or {}).copy()
        text = event.get("text") or ""

        if event_type == "turn_started":
            if progress_enabled:
                timeline.append(_timeline_line(meta, "[agent]", "turn started"))
                _trim_timeline()
                await apply_edit(force=True)
            return
        if event_type == "llm_started":
            preview = meta.get("prompt_preview")
            system_prompt = meta.get("system_prompt")
            if preview:
                LOGGER.info("Prompt preview: %s", _truncate(str(preview), 160))
            if system_prompt:
                LOGGER.info("System prompt snapshot: %s", _truncate(str(system_prompt), 160))
            if progress_enabled:
                timeline.append(_timeline_line(meta, "[model]", "model call started"))
                _trim_timeline()
                await apply_edit()
            return
        if event_type == "llm_finished":
            preview = meta.get("response_preview") or text
            if progress_enabled and preview:
                timeline.append(_timeline_line(meta, "[model]", f"response: {_truncate(str(preview), 32)}"))
                _trim_timeline()
                await apply_edit()
            return
        if event_type == "tool_started":
            name = text or meta.get("tool") or "tool"
            if progress_enabled:
                timeline.append(_timeline_line(meta, "[tool]", f"{name} started"))
                _trim_timeline()
                await apply_edit()
            return
        if event_type == "tool_finished":
            name = text or meta.get("tool") or "tool"
            if name == "web_search":
                _record_citations(meta)
            if name == "web_search":
                url = meta.get("top_url") or "(no url)"
                summary_val = meta.get("summary") or ""
                first_line = str(summary_val).splitlines()[0] if str(summary_val).splitlines() else ""
                preview_line = _truncate(first_line, settings.progress_tool_result_max_chars)
                body = f"url: {url} | {preview_line}".strip(" |")
            else:
                summary = meta.get("summary")
                count = meta.get("results_count")
                query = meta.get("query")
                details: List[str] = []
                if query:
                    details.append(f"query=\"{_truncate(str(query), 120)}\"")
                if count is not None:
                    details.append(f"{count} results")
                if summary:
                    details.append(_truncate(str(summary), settings.progress_tool_result_max_chars))
                body = "; ".join(details) if details else "finished"
            if progress_enabled:
                timeline.append(_timeline_line(meta, "[tool]", f"{name} {body}"))
                _trim_timeline()
                await apply_edit()
            return
        if event_type == "tool_error":
            name = text or meta.get("tool") or "tool"
            error_msg = meta.get("error") or text or "tool error"
            if progress_enabled:
                timeline.append(_timeline_line(meta, "[tool-error]", f"{name}: {_truncate(str(error_msg), 160)}"))
                _trim_timeline()
                await apply_edit(force=True)
            return
        if event_type == "turn_failed":
            reason = (event.get("text") or "Agent run failed.").strip()
            exc_type = meta.get("exception_type")
            if exc_type:
                reason = f"{exc_type}: {reason}"
            error_text = reason
            if progress_enabled:
                timeline.append(_timeline_line(meta, "[error]", reason))
                _trim_timeline()
                await apply_edit(force=True)
            return
        if event_type == "turn_finished":
            final_text = (event.get("text") or "").strip()
            if progress_enabled and final_text:
                timeline.append(_timeline_line(meta, "[done]", "model responded with final answer"))
                _trim_timeline()
                await apply_edit()
            return

        if progress_enabled and text:
            timeline.append(_timeline_line(meta, "[event]", _truncate(text, 160)))
            _trim_timeline()
            await apply_edit()

    remove_listener = dispatcher.add_listener(on_progress)
    await apply_edit(force=True)

    try:
        try:
            response = await runtime.run_message_with_progress(
                message.chat_id,
                user_text,
                dispatcher,
                enable_progress=True,
                sender_id=sender_id,
                log_user=False,
            )
        except Exception as exc:  # noqa: BLE001
            if error_text is None:
                error_text = str(exc)
            raise
        finally:
            remove_listener()
    except Exception:
        LOGGER.exception("Agent run failed")
        await placeholder.edit_text(f"Agent error: {error_text or 'Agent run failed.'}")
        return

    final_plain = final_text or response or "I am sorry, I could not formulate a response."
    final_plain = final_plain.strip() or "I am sorry, I could not formulate a response."
    prepared_plain = _prepare_agent_message(final_plain, citation_urls)

    try:
        if progress_enabled:
            await _send_markdown_with_fallback(
                message.reply_text,
                prepared_plain,
                raw_text=final_plain,
                disable_preview=False,
            )
        else:
            await _send_markdown_with_fallback(
                placeholder.edit_text,
                prepared_plain,
                raw_text=final_plain,
                disable_preview=False,
            )
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to send final message", exc_info=True)
        try:
            await _send_markdown_with_fallback(
                placeholder.edit_text,
                prepared_plain,
                raw_text=final_plain,
                disable_preview=False,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Fallback edit failed", exc_info=True)
            await message.reply_text(final_plain)
    if progress_enabled and not keep_timeline:
        try:
            await context.bot.delete_message(chat_id=message.chat_id, message_id=placeholder.message_id)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to delete progress message", exc_info=True)



def build_application(settings: Settings) -> Application:
    LOGGER.info("Building application with model=%s", settings.openai_model)
    runtime = AgentRuntime(settings)

    asr_service: ASRService | None = None
    if settings.asr_enabled:
        try:
            asr_service = create_asr_service(
                backend=settings.asr_backend,
                model=settings.asr_model,
                device=settings.asr_device,
                compute_type=settings.asr_compute_type,
                beam_size=settings.asr_beam_size,
                vad_filter=settings.asr_vad_filter,
                condition_on_previous_text=settings.asr_condition_on_previous_text,
            )
            LOGGER.info(
                "Initialized ASR backend=%s model=%s device=%s compute_type=%s",
                settings.asr_backend,
                settings.asr_model,
                settings.asr_device,
                settings.asr_compute_type,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to initialize ASR service")
            asr_service = None
    else:
        LOGGER.info("ASR disabled by configuration")

    ffmpeg_path_resolved = _resolve_ffmpeg_path(settings.ffmpeg_path)
    if settings.asr_enabled and not ffmpeg_path_resolved:
        LOGGER.warning("ASR is enabled but ffmpeg was not found at %s", settings.ffmpeg_path)

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.bot_data[SETTINGS_KEY] = settings
    application.bot_data[AGENT_RUNTIME_KEY] = runtime
    application.bot_data[WHITELIST_KEY] = _parse_whitelist(settings.whitelisted_user_ids)
    application.bot_data[ASR_SERVICE_KEY] = asr_service
    application.bot_data[FFMPEG_PATH_KEY] = ffmpeg_path_resolved

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("progress", toggle_progress))
    application.add_handler(CommandHandler("recap", recap_command))
    application.add_handler(CallbackQueryHandler(handle_recap_callback, pattern=r"^recap:(1h|1d)$"))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    return application


async def shutdown(application: Application) -> None:
    LOGGER.info("Shutting down application")
    runtime: AgentRuntime = application.bot_data.get(AGENT_RUNTIME_KEY)
    if runtime is not None:
        await runtime.aclose()


__all__ = [
    "build_application",
    "shutdown",
]


async def _ensure_bot_identity(context: ContextTypes.DEFAULT_TYPE) -> tuple[int | None, str | None]:
    bot_id = context.application.bot_data.get(BOT_ID_KEY)
    bot_username = context.application.bot_data.get(BOT_USERNAME_KEY)
    if bot_id is not None and bot_username is not None:
        return bot_id, bot_username
    me = await context.bot.get_me()
    bot_id = me.id
    bot_username = me.username or ""
    context.application.bot_data[BOT_ID_KEY] = bot_id
    context.application.bot_data[BOT_USERNAME_KEY] = bot_username
    return bot_id, bot_username
