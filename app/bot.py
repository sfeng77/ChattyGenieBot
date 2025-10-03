import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from app.agent_runtime import AgentRuntime
from app.config import Settings

LOGGER = logging.getLogger(__name__)

SETTINGS_KEY = "settings"
AGENT_RUNTIME_KEY = "agent_runtime"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.debug("/start invoked by chat_id=%s", message.chat_id)
    await message.reply_text("Hi! I am Chatty Genie. Send me a message and I will reply using OpenAI Agents.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.debug("/help invoked by chat_id=%s", message.chat_id)
    await message.reply_text("Commands:\n/start - welcome message\n/reset - clear conversation memory")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None:
        return
    LOGGER.debug("/reset invoked by chat_id=%s", message.chat_id)
    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]
    await runtime.reset(message.chat_id)
    await message.reply_text("Conversation memory cleared.")


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

    runtime: AgentRuntime = context.application.bot_data[AGENT_RUNTIME_KEY]

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    placeholder = await message.reply_text("Thinking...")

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
        LOGGER.debug("Failed to edit message", exc_info=True)
        await message.reply_text(final_text)


def build_application(settings: Settings) -> Application:
    LOGGER.debug("Building application with model=%s", settings.openai_model)
    runtime = AgentRuntime(settings)

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.bot_data[SETTINGS_KEY] = settings
    application.bot_data[AGENT_RUNTIME_KEY] = runtime

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text))

    return application


async def shutdown(application: Application) -> None:
    LOGGER.debug("Shutting down application")
    runtime: AgentRuntime = application.bot_data.get(AGENT_RUNTIME_KEY)
    if runtime is not None:
        await runtime.aclose()


__all__ = [
    "build_application",
    "shutdown",
]

