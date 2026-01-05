from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str
    DEEPGRAM_API_KEY: str
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
