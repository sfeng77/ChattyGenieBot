from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated

import dateparser
from agents import function_tool
from pydantic import Field

from app.features.reminder.store import ReminderStore

if TYPE_CHECKING:
    from telegram.ext import Application

LOGGER = logging.getLogger(__name__)

current_chat_id: ContextVar[int | None] = ContextVar("current_chat_id", default=None)


def schedule_reminder_job(
    application: "Application",
    *,
    chat_id: int,
    reminder_id: int,
    message: str,
    delay: float,
    store: ReminderStore,
) -> None:
    """Register a one-shot PTB JobQueue job that fires the reminder."""

    async def _fire(context) -> None:
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"⏰ 提醒：{message}")
        except Exception:
            LOGGER.exception("Failed to deliver reminder chat_id=%s id=%s", chat_id, reminder_id)
        finally:
            store.mark_done(reminder_id)

    application.job_queue.run_once(_fire, when=delay, name=f"reminder-{reminder_id}")


def create_reminder_tool(store: ReminderStore, application: "Application"):
    """Return a set_reminder agent tool bound to *store* and *application*."""

    @function_tool(name_override="set_reminder")
    async def set_reminder(
        time_expression: Annotated[
            str,
            Field(
                description=(
                    "Natural language time for the reminder, e.g. '明天早上9点', "
                    "'in 2 hours', 'next Monday at 3pm'."
                )
            ),
        ],
        message: Annotated[
            str,
            Field(description="Reminder content to send, e.g. '开会', 'take medicine'."),
        ],
    ) -> dict:
        """Schedule a Telegram reminder for the current chat at the specified time."""
        chat_id = current_chat_id.get()
        if chat_id is None:
            return {"success": False, "error": "Unable to determine chat context."}

        now = datetime.now(timezone.utc)
        parsed = dateparser.parse(
            time_expression,
            languages=["zh", "en"],
            settings={
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )
        if parsed is None:
            return {
                "success": False,
                "error": (
                    f"无法解析时间 '{time_expression}'，"
                    "请用更清晰的表达，比如'明天早上9点'或'30分钟后'。"
                ),
            }
        if parsed <= now:
            return {
                "success": False,
                "error": (
                    f"解析的时间 {parsed.strftime('%Y-%m-%d %H:%M')} 已经过去，"
                    "请重新指定时间。"
                ),
            }

        delay = (parsed - now).total_seconds()
        reminder_id = store.add_reminder(chat_id=chat_id, remind_at=parsed, message=message)
        schedule_reminder_job(
            application,
            chat_id=chat_id,
            reminder_id=reminder_id,
            message=message,
            delay=delay,
            store=store,
        )
        return {
            "success": True,
            "remind_at": parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "message": message,
            "reminder_id": reminder_id,
        }

    return set_reminder


__all__ = ["create_reminder_tool", "schedule_reminder_job", "current_chat_id"]
