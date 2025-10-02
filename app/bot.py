import asyncio
import logging
import time
from typing import List

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from app.config import Settings
from app.ollama_client import OllamaClient
from app.prompt import build_prompt
from app.query_refiner import refine_query
from app.rag.indexer import build_index_async
from app.rag.retriever import RAGRetriever
from app.state import ConversationStore

LOGGER = logging.getLogger(__name__)

SETTINGS_KEY = "settings"
OLLAMA_KEY = "ollama_client"
RETRIEVER_KEY = "retriever"
STORE_KEY = "conversation_store"
REINDEX_LOCK_KEY = "reindex_lock"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.debug("/start invoked by chat_id=%s", message.chat_id)
    await message.reply_text(
        "Hi! I am Chatty Genie. Send me a message and I will answer using your local documents."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.debug("/help invoked by chat_id=%s", message.chat_id)
    await message.reply_text(
        "Commands:\n/start - welcome message\n/reset - clear conversation memory\n/reindex - rebuild document index (owner only)"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.debug("/reset invoked by chat_id=%s", message.chat_id)
    store: ConversationStore = context.application.bot_data[STORE_KEY]
    store.reset(message.chat_id)
    await message.reply_text("Conversation memory cleared.")


def _can_reindex(settings: Settings, user_id: int) -> bool:
    allowed = settings.allowed_reindex_user_ids
    if not allowed:
        return True
    if len(allowed) == 1 and isinstance(allowed[0], str):
        try:
            parsed = [int(part.strip()) for part in str(allowed[0]).split(",") if part.strip()]
        except ValueError:
            parsed = []
        else:
            settings.allowed_reindex_user_ids = parsed
            allowed = settings.allowed_reindex_user_ids
    permitted = user_id in allowed
    LOGGER.debug("Reindex permission check for user_id=%s -> %s", user_id, permitted)
    return permitted


async def reindex(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    user_id = update.effective_user.id if update.effective_user else None
    LOGGER.debug("/reindex invoked by user_id=%s", user_id)
    settings: Settings = context.application.bot_data[SETTINGS_KEY]
    if user_id is None or not _can_reindex(settings, user_id):
        await message.reply_text("You are not allowed to rebuild the index.")
        return

    lock: asyncio.Lock = context.application.bot_data.setdefault(REINDEX_LOCK_KEY, asyncio.Lock())
    if lock.locked():
        LOGGER.debug("Reindex requested while lock held")
        await message.reply_text("Index rebuild already in progress. Please wait.")
        return

    status_message = await message.reply_text("Rebuilding the index. This may take a while...")
    LOGGER.info("User %s triggered index rebuild", user_id)

    async with lock:
        ollama: OllamaClient = context.application.bot_data[OLLAMA_KEY]
        try:
            await build_index_async(settings, ollama)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to rebuild index")
            await status_message.edit_text(f"Index rebuild failed: {exc}")
            return
    await status_message.edit_text("Index rebuild complete.")
    LOGGER.info("Index rebuild completed successfully")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return
    chat_id = message.chat_id
    user_text = message.text.strip()
    LOGGER.debug("Received text message (chat_id=%s, chars=%s)", chat_id, len(user_text))
    if not user_text:
        await message.reply_text("Please send some text.")
        return

    settings: Settings = context.application.bot_data[SETTINGS_KEY]
    if len(user_text) > settings.max_input_chars:
        LOGGER.debug("Message exceeds max_input_chars=%s", settings.max_input_chars)
        await message.reply_text(f"Message too long. Limit is {settings.max_input_chars} characters.")
        return

    store: ConversationStore = context.application.bot_data[STORE_KEY]
    retriever: RAGRetriever = context.application.bot_data[RETRIEVER_KEY]
    ollama: OllamaClient = context.application.bot_data[OLLAMA_KEY]

    history = store.get(chat_id)
    LOGGER.debug("Loaded history turns=%s", len(history))

    try:
        search_query = await refine_query(ollama, user_text, history)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Search query refinement crashed: %s", exc)
        search_query = user_text
    LOGGER.debug("Search query used for retrieval: %s", search_query)


    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        retrievals = await retriever.retrieve(search_query)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Retrieval failed")
        await message.reply_text(f"Retrieval failed: {exc}")
        return

    LOGGER.debug("RAG returned %s chunks", len(retrievals))
    if retrievals:
        for idx, item in enumerate(retrievals, start=1):
            snippet = item.text.replace("\n", " ")
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            LOGGER.debug(
                "Retrieval %s | score=%.4f | path=%s | offset=%s | snippet=%s",
                idx,
                item.score,
                item.doc_path,
                item.offset,
                snippet,
            )
    else:
        LOGGER.debug("No retrieval context found for chat_id=%s", chat_id)

    system_prompt, prompt = build_prompt(history, user_text, retrievals)
    LOGGER.debug(
        "Constructed prompt (history_len=%s, prompt_chars=%s, has_context=%s)",
        len(history),
        len(prompt),
        bool(retrievals),
    )
    LOGGER.debug("System prompt: %s", system_prompt)
    prompt_preview = prompt if len(prompt) <= 2000 else f"{prompt[:2000]}..."
    LOGGER.debug("Prompt sent to model:\n%s", prompt_preview)

    placeholder = await message.reply_text("Thinking...")
    buffer: List[str] = []
    last_edit = time.monotonic()
    throttle_seconds = 0.4
    final_text = ""

    try:
        async for chunk in ollama.generate_stream(prompt=prompt, system_prompt=system_prompt):
            buffer.append(chunk)
            now = time.monotonic()
            if now - last_edit >= throttle_seconds:
                try:
                    await placeholder.edit_text("".join(buffer))
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Failed to edit message", exc_info=True)
                last_edit = now
        final_text = "".join(buffer).strip()
        LOGGER.debug("Generation completed (chars=%s)", len(final_text))
        if not final_text:
            final_text = "I am sorry, I could not formulate a response."
        try:
            await placeholder.edit_text(final_text)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to set final text", exc_info=True)
            await message.reply_text(final_text)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("LLM streaming failed")
        await placeholder.edit_text(f"Generation failed: {exc}")
        return

    store.add(chat_id, "user", user_text)
    store.add(chat_id, "assistant", final_text)
    LOGGER.debug("Stored turns for chat_id=%s (total_turns=%s)", chat_id, len(store.get(chat_id)))


def build_application(settings: Settings) -> Application:
    LOGGER.debug("Building application with settings: top_k=%s, max_turns=%s", settings.top_k, settings.max_turns)
    ollama = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        embed_model=settings.embed_model,
        timeout=settings.ollama_request_timeout,
        temperature=settings.ollama_temperature,
    )
    retriever = RAGRetriever(
        index_dir=settings.index_dir,
        top_k=settings.top_k,
        client=ollama,
    )
    store = ConversationStore(settings.max_turns)

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.bot_data[SETTINGS_KEY] = settings
    application.bot_data[OLLAMA_KEY] = ollama
    application.bot_data[RETRIEVER_KEY] = retriever
    application.bot_data[STORE_KEY] = store
    application.bot_data[REINDEX_LOCK_KEY] = asyncio.Lock()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("reindex", reindex))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    return application


async def shutdown(application: Application) -> None:
    LOGGER.debug("Shutting down application")
    ollama: OllamaClient = application.bot_data.get(OLLAMA_KEY)
    if ollama is not None:
        await ollama.aclose()


__all__ = [
    "build_application",
    "shutdown",
]
