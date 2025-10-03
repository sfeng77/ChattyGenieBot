import asyncio
import logging

from app.bot import build_application, shutdown as bot_shutdown
from app.config import get_settings


def main() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    application = build_application(settings)
    try:
        application.run_polling()
    finally:
        asyncio.run(bot_shutdown(application))


if __name__ == "__main__":
    main()

