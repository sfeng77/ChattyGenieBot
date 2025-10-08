import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings


_configured = False


def setup_logging(settings: "Settings") -> None:
    """Configure logging to stream to console and append to the configured log file."""
    global _configured
    if _configured:
        return

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s | %(message)s"

    file_handler = logging.FileHandler(settings.log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[console_handler, file_handler],
        force=True,
    )

    logging.getLogger("httpx").setLevel(log_level)
    logging.getLogger(__name__).info("Logging configured (console_level=%s, file=%s)", logging.getLevelName(log_level), settings.log_file_path)
    _configured = True
