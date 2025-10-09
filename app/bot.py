import logging
import time
from functools import wraps
from typing import Any, Awaitable, Callable, List, Set

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.agent_runtime import AgentRuntime
from app.config import Settings
from app.progress import ProgressDispatcher, ProgressEvent

LOGGER = logging.getLogger(__name__)

MAX_PROGRESS_LINES = 40
MAX_PROGRESS_CHARS = 3000


SETTINGS_KEY = "settings"
AGENT_RUNTIME_KEY = "agent_runtime"
WHITELIST_KEY = "whitelist_user_ids"

HandlerFunc = Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]


def _truncate(text: str, limit: int = 160) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


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
    await message.reply_text("Hi! I am Agent Mushroom. Send me a message and I will reply using OpenAI Agents.")


@require_authorized
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.info("/help invoked by chat_id=%s", message.chat_id)
    await message.reply_text("Commands:\n/start - welcome message\n/reset - clear conversation memory\n/progress - toggle live progress updates for this chat")


@require_authorized
async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.info("/reset invoked by chat_id=%s", message.chat_id)
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

    user_lines = ["User sent an image.", f"telegram_file_id={file_id}"]
    if caption:
        user_lines.append(f"Caption: {caption}")
    else:
        user_lines.append("Caption: (none provided)")
    user_lines.append("Use the vision_analyze tool if you need to inspect the image before answering.")
    agent_input = "\n".join(user_lines)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    placeholder = await message.reply_text("Analyzing image...")

    try:
        response = await runtime.run_message(chat_id, agent_input)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Agent vision run failed")
        await placeholder.edit_text(f"Agent error: {exc}")
        return

    final_text = (response or "").strip() or "I am sorry, I could not describe that image."
    try:
        await placeholder.edit_text(final_text)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to edit vision placeholder", exc_info=True)
        await message.reply_text(final_text)



@require_authorized
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return
    chat_id = message.chat_id
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
    progress_enabled = bool(context.chat_data.get("progress_enabled", False))

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    placeholder = await message.reply_text("Thinking...")

    if not progress_enabled:
        try:
            response = await runtime.run_message(chat_id, user_text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Agent run failed")
            await placeholder.edit_text(f"Agent error: {exc}")
            return
        final_text = response.strip() or "I am sorry, I could not formulate a response."
        try:
            await placeholder.edit_text(final_text)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to edit message", exc_info=True)
            await message.reply_text(final_text)
        return

    dispatcher = ProgressDispatcher()
    throttle_seconds = max(settings.progress_edit_throttle_ms / 1000.0, 0.1)
    keep_timeline = settings.progress_keep_timeline
    last_edit = 0.0
    final_text: str | None = None
    error_text: str | None = None

    def _timeline_line(meta: dict[str, Any] | None, label: str, body: str) -> str:
        elapsed: str | None = None
        if meta and isinstance(meta.get("elapsed_seconds"), (int, float)):
            elapsed = f"{meta['elapsed_seconds']:.2f}s"
        parts = [p for p in [elapsed, label, body] if p]
        return " | ".join(parts)

    timeline: List[str] = [_timeline_line({"elapsed_seconds": 0.0}, "[user]", _truncate(user_text, 200))]

    def _trim_timeline() -> None:
        # Keep only the most recent progress entries to avoid hitting Telegram limits
        while len(timeline) > MAX_PROGRESS_LINES:
            if len(timeline) <= 1:
                break
            timeline.pop(1)
        while len('\n'.join(timeline)) > MAX_PROGRESS_CHARS and len(timeline) > 1:
            timeline.pop(1)

    async def apply_edit(force: bool = False) -> None:
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
            timeline.append(_timeline_line(meta, "[model]", "model call started"))
            _trim_timeline()
            await apply_edit()
            return
        if event_type == "llm_finished":
            preview = meta.get("response_preview") or text
            if preview:
                timeline.append(_timeline_line(meta, "[model]", f"response: {_truncate(str(preview), 32)}"))
                _trim_timeline()
                await apply_edit()
            return
        if event_type == "tool_started":
            name = text or meta.get("tool") or "tool"
            timeline.append(_timeline_line(meta, "[tool]", f"{name} started"))
            _trim_timeline()
            await apply_edit()
            return
        if event_type == "tool_finished":
            name = text or meta.get("tool") or "tool"
            if name == "web_search":
                url = meta.get("top_url") or "(no url)"
                summary_val = meta.get("summary") or ""
                first_line = str(summary_val).splitlines()[0] if str(summary_val).splitlines() else ""
                preview_line = _truncate(first_line, settings.progress_tool_result_max_chars)
                body = f"url: {url} | {preview_line}".strip(' |')
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
            timeline.append(_timeline_line(meta, "[tool]", f"{name} {body}"))
            _trim_timeline()
            await apply_edit()
            return
        if event_type == "tool_error":
            name = text or meta.get("tool") or "tool"
            error_msg = meta.get("error") or text or "tool error"
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
            timeline.append(_timeline_line(meta, "[error]", reason))
            _trim_timeline()
            await apply_edit(force=True)
            return
        if event_type == "turn_finished":
            final_text = (event.get("text") or "").strip()
            if final_text:
                timeline.append(_timeline_line(meta, "[done]", "model responded with final answer"))
                _trim_timeline()
                await apply_edit()
            return

        if text:
            timeline.append(_timeline_line(meta, "[event]", _truncate(text, 160)))
            _trim_timeline()
            await apply_edit()

    remove_listener = dispatcher.add_listener(on_progress)
    await apply_edit(force=True)

    try:
        try:
            response = await runtime.run_message_with_progress(chat_id, user_text, dispatcher, enable_progress=True)
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

    try:
        await message.reply_text(final_plain)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Failed to send final message", exc_info=True)
        try:
            await placeholder.edit_text(final_plain)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Fallback edit failed", exc_info=True)
            await message.reply_text(final_plain)
    if not keep_timeline:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=placeholder.message_id)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to delete progress message", exc_info=True)



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



def build_application(settings: Settings) -> Application:
    LOGGER.info("Building application with model=%s", settings.openai_model)
    runtime = AgentRuntime(settings)

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.bot_data[SETTINGS_KEY] = settings
    application.bot_data[AGENT_RUNTIME_KEY] = runtime
    application.bot_data[WHITELIST_KEY] = _parse_whitelist(settings.whitelisted_user_ids)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("progress", toggle_progress))
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




