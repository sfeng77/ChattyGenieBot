from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

DEFAULT_LOG_DIR = Path("data") / "logs"
DEFAULT_LOG_FILENAME = "chattygenie.log"
DEFAULT_LOG_FILE_PATH = DEFAULT_LOG_DIR / DEFAULT_LOG_FILENAME


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
    log_dir: Path = Field(DEFAULT_LOG_DIR, alias="LOG_DIR")
    log_file_path: Path = Field(DEFAULT_LOG_FILE_PATH, alias="LOG_FILE_PATH")
    finance_enabled: bool = Field(False, alias="FINANCE_ENABLED")
    finance_provider: str = Field("alpha_vantage", alias="FINANCE_PROVIDER")
    finance_api_key: str | None = Field(None, alias="FINANCE_API_KEY")
    finance_timeout: float = Field(10.0, alias="FINANCE_TIMEOUT")
    finance_default_window_days: int = Field(7, alias="FINANCE_WINDOW_DAYS")
    finance_cache_ttl_minutes: int = Field(10, alias="FINANCE_CACHE_TTL_MINUTES")
    vision_enabled: bool = Field(False, alias="VISION_ENABLED")
    vision_model: str = Field("gemma3:12b", alias="VISION_MODEL")
    vision_temperature: float = Field(0.2, alias="VISION_TEMPERATURE")
    vision_timeout: float = Field(20.0, alias="VISION_TIMEOUT")
    vision_max_edge: int = Field(1280, alias="VISION_MAX_EDGE")
    vision_system_prompt: str = Field(
        "You are a concise vision assistant. Focus on factual observations and answer user questions about the image.",
        alias="VISION_SYSTEM_PROMPT",
    )
    max_input_chars: int = Field(4000, alias="MAX_INPUT_CHARS")
    progress_edit_throttle_ms: int = Field(800, alias="PROGRESS_EDIT_THROTTLE_MS")
    progress_keep_timeline: bool = Field(False, alias="PROGRESS_KEEP_TIMELINE")
    progress_tool_result_max_chars: int = Field(160, alias="PROGRESS_TOOL_RESULT_MAX_CHARS")
    history_prune_enabled: bool = Field(True, alias="HISTORY_PRUNE_ENABLED")
    history_prune_threshold_items: int = Field(60, alias="HISTORY_PRUNE_THRESHOLD_ITEMS")
    history_keep_last_items: int = Field(12, alias="HISTORY_KEEP_LAST_ITEMS")
    history_summary_max_chars: int = Field(800, alias="HISTORY_SUMMARY_MAX_CHARS")
    # Maximum characters of raw history to feed into recap transcript
    history_recap_transcript_chars: int = Field(8000, alias="HISTORY_RECAP_TRANSCRIPT_CHARS")
    whitelisted_user_ids: str | None = Field(None, alias="WHITELISTED_USER_IDS")
    sessions_db_path: Path = Field(Path("data") / "sessions" / "sessions.db", alias="SESSIONS_DB_PATH")
    chat_history_db_path: Path = Field(Path("data") / "history" / "chat_history.db", alias="CHAT_HISTORY_DB_PATH")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def _apply_log_dir(self) -> "Settings":
        if "log_file_path" not in self.model_fields_set:
            filename = self.log_file_path.name
            self.log_file_path = self.log_dir / filename
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.sessions_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chat_history_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
