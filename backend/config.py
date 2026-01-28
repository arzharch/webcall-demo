from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    ENV: str = "development"

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    SQLITE_DB_PATH: str = os.path.join(DATA_DIR, "bella_voice.db")

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Logging
    LOG_LEVEL: str = "INFO"
    JSON_LOGS: bool = False

    # OpenTelemetry
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "bella-voice-ai"
    OTEL_TRACE_EXPORTER: str = "console"

    # LLM
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.4
    GEMINI_MAX_TOKENS: int = 512

    # Audio
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_FRAME_MS: int = 30

    # TTS
    TTS_VOICE: str = "en-IN-Neural2-A"
    TTS_MODEL: str = "models/gemini-2.5-flash-tts"
    TTS_SPEAKING_RATE: float = 1.0
    TTS_PITCH: float = 0.0

    # Cost tracking defaults (USD)
    STT_COST_PER_MINUTE: float = 0.006
    TTS_COST_PER_MILLION_CHARS: float = 15.0
    LLM_INPUT_COST_PER_1K_TOKENS: float = 0.12
    LLM_OUTPUT_COST_PER_1K_TOKENS: float = 0.36

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    _settings = Settings()
    os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS", _settings.GOOGLE_APPLICATION_CREDENTIALS
    )
    os.makedirs(_settings.DATA_DIR, exist_ok=True)
    return _settings


# Module-level settings instance for easy import
settings = get_settings()
