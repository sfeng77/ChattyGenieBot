from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_api_base: str | None = Field("http://localhost:11434/v1", env="OPENAI_API_BASE")
    openai_model: str = Field("gpt-oss:20b", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.2, env="OPENAI_TEMPERATURE")
    openai_tracing_enabled: bool = Field(False, env="OPENAI_TRACING_ENABLED")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_input_chars: int = Field(4000, env="MAX_INPUT_CHARS")
    sessions_db_path: Path = Field(Path("data") / "sessions" / "sessions.db", env="SESSIONS_DB_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.sessions_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
