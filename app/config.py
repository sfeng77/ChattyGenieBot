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
    log_file_path: Path = Field(Path("data") / "logs" / "chattygenie.log", alias="LOG_FILE_PATH")
    finance_enabled: bool = Field(False, alias="FINANCE_ENABLED")
    finance_provider: str = Field("alpha_vantage", alias="FINANCE_PROVIDER")
    finance_api_key: str | None = Field(None, alias="FINANCE_API_KEY")
    finance_timeout: float = Field(10.0, alias="FINANCE_TIMEOUT")
    finance_default_window_days: int = Field(7, alias="FINANCE_WINDOW_DAYS")
    finance_cache_ttl_minutes: int = Field(10, alias="FINANCE_CACHE_TTL_MINUTES")
    max_input_chars: int = Field(4000, alias="MAX_INPUT_CHARS")
    progress_edit_throttle_ms: int = Field(800, alias="PROGRESS_EDIT_THROTTLE_MS")
    progress_keep_timeline: bool = Field(False, alias="PROGRESS_KEEP_TIMELINE")
    progress_tool_result_max_chars: int = Field(160, alias="PROGRESS_TOOL_RESULT_MAX_CHARS")
    history_prune_enabled: bool = Field(True, alias="HISTORY_PRUNE_ENABLED")
    history_prune_threshold_items: int = Field(60, alias="HISTORY_PRUNE_THRESHOLD_ITEMS")
    history_keep_last_items: int = Field(12, alias="HISTORY_KEEP_LAST_ITEMS")
    history_summary_max_chars: int = Field(800, alias="HISTORY_SUMMARY_MAX_CHARS")
    whitelisted_user_ids: str | None = Field(None, alias="WHITELISTED_USER_IDS")
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
    settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
