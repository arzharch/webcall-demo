from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str
    
    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    TICKETS_FILE: str = os.path.join(BASE_DIR, "data", "tickets.json")
    KB_FILE: str = os.path.join(BASE_DIR, "data", "restaurant_kb.json")
    FAISS_INDEX: str = os.path.join(BASE_DIR, "data", "faiss_index")
    
    # STT Config (Faster-Whisper)
    STT_MODEL: str = "base"  # Options: tiny, base, small, medium, large
    STT_DEVICE: str = "cpu"
    STT_COMPUTE_TYPE: str = "int8"
    
    # TTS Config (Coqui XTTS)
    TTS_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    
    # LLM Config (Gemini)
    GEMINI_MODEL: str = "gemini-1.5-flash" # Updated model
    MAX_TOKENS: int = 250
    TEMPERATURE: float = 0.7
    AGENT_VERBOSE: bool = False # Added for configurable verbosity
    
    # RAG Config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_RESULTS: int = 3
    
    # Conversation Config
    MAX_CONVERSATION_TURNS: int = 50
    CONTEXT_WINDOW: int = 10
    
    # Audio Config
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance that also ensures necessary directories exist."""
    settings = Settings()
    
    # Create directories if they don't exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    
    return settings
