"""
Test configuration and fixtures for the Bella Voice API test suite.
"""
import os
import sys
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables BEFORE importing app modules
os.environ["TESTING"] = "true"
os.environ["DEEPGRAM_API_KEY"] = "test_deepgram_key"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
os.environ["OPENAI_API_KEY"] = "test_openai_key"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_deepgram():
    """Mock Deepgram client for STT tests."""
    with patch("services.stt_service.DeepgramClient") as mock:
        mock_instance = MagicMock()
        mock_instance.listen.asynclive.v.return_value = AsyncMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_google_tts():
    """Mock Google TTS client for TTS tests."""
    with patch("services.tts_service.texttospeech.TextToSpeechClient") as mock:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.audio_content = b"fake_audio_data" * 100
        mock_instance.synthesize_speech.return_value = mock_response
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_redis():
    """Mock Redis cache for tests."""
    with patch("services.tts_service.RedisCache") as mock:
        mock_instance = MagicMock()
        mock_instance.get_tts_audio.return_value = None  # Cache miss by default
        mock_instance.set_tts_audio.return_value = True
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_audio_chunk():
    """Generate sample PCM audio data."""
    import numpy as np
    # Generate 100ms of silence at 16kHz
    samples = np.zeros(1600, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def test_db(tmp_path):
    """
    Create a test database instance.
    Uses a temporary directory for isolation.
    """
    import database as db
    
    # Create temp database path
    test_db_path = tmp_path / "data" / "bella.db"
    test_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Patch the DB_PATH to use our test location
    original_db_path = db.DB_PATH
    db.DB_PATH = test_db_path
    
    # Initialize the test database
    db.init_database()
    
    yield db
    
    # Cleanup
    db.DB_PATH = original_db_path
    if test_db_path.exists():
        test_db_path.unlink()


@pytest.fixture
def mock_agent():
    """Mock BellaAgent for testing without LLM calls."""
    with patch("services.voice_session.BellaAgent") as mock:
        mock_instance = MagicMock()
        mock_instance.process = AsyncMock(return_value="I can help you with that!")
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai():
    """Mock OpenAI API for agent tests."""
    with patch("agent.agent.ChatOpenAI") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance
