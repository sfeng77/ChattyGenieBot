import asyncio

from app.bot import build_application, shutdown as bot_shutdown
from app.config import get_settings
from app.logging_utils import setup_logging


def main() -> None:
    settings = get_settings()
    setup_logging(settings)
    application = build_application(settings)
    try:
        application.run_polling()
    finally:
        asyncio.run(bot_shutdown(application))


if __name__ == "__main__":
    main()
