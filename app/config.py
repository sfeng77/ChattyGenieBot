from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    llm_model: str = Field("llama3.1", env="LLM_MODEL")
    embed_model: str = Field("nomic-embed-text", env="EMBED_MODEL")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    docs_dir: Path = Field(Path("data") / "docs", env="DOCS_DIR")
    index_dir: Path = Field(Path("data") / "index", env="INDEX_DIR")

    top_k: int = Field(5, env="TOP_K")
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(120, env="CHUNK_OVERLAP")
    max_turns: int = Field(12, env="MAX_TURNS")
    max_input_chars: int = Field(4000, env="MAX_INPUT_CHARS")

    moderation_enabled: bool = Field(True, env="MODERATION_ENABLED")

    ollama_request_timeout: float = Field(120.0, env="OLLAMA_TIMEOUT")
    ollama_temperature: float = Field(0.2, env="OLLAMA_TEMPERATURE")

    allowed_reindex_user_ids: List[int] = Field(default_factory=list, env="ALLOWED_REINDEX_USER_IDS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
