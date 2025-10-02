import asyncio
import logging

from app.bot import build_application, shutdown as bot_shutdown
from app.config import get_settings


async def run_bot() -> None:
    settings = get_settings()
    application = build_application(settings)

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    try:
        await application.updater.idle()
    finally:
        await application.stop()
        await application.shutdown()
        await bot_shutdown(application)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
