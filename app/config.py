from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_api_base: str | None = Field("http://localhost:11434/v1", alias="OPENAI_API_BASE")
    openai_model: str = Field("gpt-oss:20b", alias="OPENAI_MODEL")
    openai_temperature: float = Field(0.2, alias="OPENAI_TEMPERATURE")
    openai_tracing_enabled: bool = Field(False, alias="OPENAI_TRACING_ENABLED")
    web_search_enabled: bool = Field(False, alias="WEB_SEARCH_ENABLED")
    web_search_base_url: str = Field("https://ollama.com", alias="WEB_SEARCH_BASE_URL")
    web_search_endpoint: str = Field("/api/web_search", alias="WEB_SEARCH_ENDPOINT")
    web_search_timeout: float = Field(15.0, alias="WEB_SEARCH_TIMEOUT")
    web_search_default_max_results: int = Field(5, alias="WEB_SEARCH_MAX_RESULTS")
    web_search_api_key: str | None = Field(None, alias="OLLAMA_API_KEY")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    max_input_chars: int = Field(4000, alias="MAX_INPUT_CHARS")
    sessions_db_path: Path = Field(Path("data") / "sessions" / "sessions.db", alias="SESSIONS_DB_PATH")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.sessions_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
