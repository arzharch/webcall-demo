from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Keys & Auth
    GEMINI_API_KEY: str
    
    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    DEBUG: bool = False
    
    # Paths
    BASE_DIR: str = str(Path(__file__).parent)
    DATA_DIR: str = str(Path(__file__).parent / "data")
    MODELS_DIR: str = str(Path(__file__).parent / "models")
    TICKETS_FILE: str = str(Path(__file__).parent / "data" / "tickets.json")
    KB_FILE: str = str(Path(__file__).parent / "data" / "restaurant_kb.json")
    FAISS_INDEX: str = str(Path(__file__).parent / "data" / "faiss_index")
    
    # STT Config (Faster-Whisper)
    STT_MODEL: str = "base"
    STT_DEVICE: str = "cpu"
    STT_COMPUTE_TYPE: str = "int8"
    STT_LANGUAGE: str = "en"
    
    # TTS Config (Coqui XTTS)
    TTS_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    TTS_LANGUAGE: str = "en"
    
    # LLM Config (Gemini)
    GEMINI_MODEL: str = "gemini-1.5-flash"
    MAX_TOKENS: int = 300
    TEMPERATURE: float = 0.7
    
    # RAG Config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_RESULTS: int = 3
    CHUNK_SIZE: int = 500
    
    # Conversation Config
    MAX_CONVERSATION_TURNS: int = 50
    CONTEXT_WINDOW: int = 10
    SILENCE_TIMEOUT: float = 2.5
    
    # Audio Config
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    return settings
