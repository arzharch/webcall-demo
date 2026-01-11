from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str
    DEEPGRAM_API_KEY: str
    GOOGLE_API_KEY: str
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    
    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    TRANSCRIPTS_DIR: str = os.path.join(BASE_DIR, "data", "transcripts")
    TICKETS_DIR: str = os.path.join(BASE_DIR, "data", "tickets")
    SQLITE_DB_PATH: str = os.path.join(BASE_DIR, "data", "bella_voice.db")
    
    # LLM Config
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    
    # STT Config (Deepgram)
    DEEPGRAM_MODEL: str = "nova-2"
    DEEPGRAM_LANGUAGE: str = "en"
    
    # TTS Config (Google Cloud)
    TTS_LANGUAGE_CODE: str = "en-US"
    TTS_VOICE_NAME: str = "en-US-Neural2-F"
    TTS_SPEAKING_RATE: float = 1.0
    TTS_PITCH: float = 0.0
    
    # Voice Call + Costing
    MAX_CALL_MINUTES: int = 5
    SIGNALING_TOKEN_TTL_SECONDS: int = 300
    STT_COST_PER_MINUTE: float = 0.004  # USD per audio minute
    TTS_COST_PER_MILLION_CHARS: float = 12.0
    LLM_INPUT_COST_PER_1K_TOKENS: float = 0.35
    LLM_OUTPUT_COST_PER_1K_TOKENS: float = 1.05

    # VAD Config
    VAD_AGGRESSIVENESS: int = 2
    VAD_START_WINDOW_MS: int = 240
    VAD_END_WINDOW_MS: int = 600
    VAD_FRAME_MS: int = 30

    # Restaurant Template
    TOTAL_TABLES: int = 18
    TABLES_PER_SLOT: int = 12
    RESERVATION_SLOT_MINUTES: int = 30
    RESERVATION_SERVICE_START: str = "11:00"
    RESERVATION_SERVICE_END: str = "23:00"
    MAX_PARTY_SIZE: int = 12
    WEEKLY_EVENTS: List[dict] = [
        {"day": "wednesday", "title": "Wine Flight Wednesday", "description": "Four-course tasting menu with curated Italian wines."},
        {"day": "saturday", "title": "Live Jazz Supper", "description": "Trio performance between 7-10 PM with chef specials."},
    ]

    # Audio Config
    SAMPLE_RATE: int = 16000
    AUDIO_BUFFER_SECONDS: int = 3
    MIN_TRANSCRIPTION_LENGTH: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

def ensure_directories():
    """Create necessary directories if they don't exist"""
    settings = get_settings()
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(settings.TICKETS_DIR, exist_ok=True)
