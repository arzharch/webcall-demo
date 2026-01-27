"""
Tests for Voice Session management.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    ws = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_bytes = AsyncMock(return_value=b"audio-data")
    return ws


class TestVoiceSessionCreation:
    """Test VoiceSession creation and initialization."""
    
    def test_session_creation(self, mock_websocket, test_db):
        """Test creating a new voice session."""
        with patch("services.voice_session.STTService"), \
             patch("services.voice_session.TTSService"), \
             patch("services.voice_session.BellaAgent"), \
             patch("services.voice_session.db", test_db):
            
            from services.voice_session import VoiceSession, CallStatus
            
            session = VoiceSession(
                websocket=mock_websocket,
                caller_name="John Doe",
                session_id="test-session-123"
            )
            
            assert session.session_id == "test-session-123"
            assert session.caller_name == "John Doe"
            assert session.status == CallStatus.CONNECTING
    
    def test_session_auto_generates_id(self, mock_websocket, test_db):
        """Test that session ID is auto-generated if not provided."""
        with patch("services.voice_session.STTService"), \
             patch("services.voice_session.TTSService"), \
             patch("services.voice_session.BellaAgent"), \
             patch("services.voice_session.db", test_db):
            
            from services.voice_session import VoiceSession
            
            session = VoiceSession(
                websocket=mock_websocket,
                caller_name="Jane"
            )
            
            assert session.session_id is not None
            assert len(session.session_id) > 0


class TestVoiceSessionLifecycle:
    """Test voice session lifecycle."""
    
    @pytest.mark.asyncio
    async def test_session_start(self, mock_websocket, test_db, mock_google_tts):
        """Test starting a voice session."""
        with patch("services.voice_session.STTService") as MockSTT, \
             patch("services.voice_session.TTSService") as MockTTS, \
             patch("services.voice_session.BellaAgent") as MockAgent, \
             patch("services.voice_session.db", test_db):
            
            # Configure mocks
            mock_stt = AsyncMock()
            mock_stt.start = AsyncMock()
            MockSTT.return_value = mock_stt
            
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=b"greeting-audio")
            MockTTS.return_value = mock_tts
            
            from services.voice_session import VoiceSession, CallStatus
            
            session = VoiceSession(
                websocket=mock_websocket,
                caller_name="Test User",
                session_id="start-test"
            )
            
            await session.start()
            
            assert session.status == CallStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_session_end(self, mock_websocket, test_db, mock_google_tts):
        """Test ending a voice session."""
        with patch("services.voice_session.STTService") as MockSTT, \
             patch("services.voice_session.TTSService") as MockTTS, \
             patch("services.voice_session.BellaAgent") as MockAgent, \
             patch("services.voice_session.db", test_db):
            
            mock_stt = AsyncMock()
            mock_stt.start = AsyncMock()
            mock_stt.stop = AsyncMock()
            MockSTT.return_value = mock_stt
            
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=b"audio")
            MockTTS.return_value = mock_tts
            
            from services.voice_session import VoiceSession, CallStatus
            
            session = VoiceSession(
                websocket=mock_websocket,
                caller_name="End Test",
                session_id="end-test"
            )
            
            await session.start()
            await session.end("test_ended")
            
            assert session.status == CallStatus.ENDED


class TestVoiceSessionMetrics:
    """Test session metrics tracking."""
    
    def test_metrics_initialization(self, mock_websocket, test_db):
        """Test that metrics are initialized correctly."""
        with patch("services.voice_session.STTService"), \
             patch("services.voice_session.TTSService"), \
             patch("services.voice_session.BellaAgent"), \
             patch("services.voice_session.db", test_db):
            
            from services.voice_session import VoiceSession
            
            session = VoiceSession(
                websocket=mock_websocket,
                caller_name="Metrics User"
            )
            
            assert session.metrics.turn_count == 0
            assert session.metrics.interruption_count == 0
            assert session.metrics.error_count == 0
    
    def test_average_latency_calculation(self, mock_websocket, test_db):
        """Test average latency calculation."""
        with patch("services.voice_session.STTService"), \
             patch("services.voice_session.TTSService"), \
             patch("services.voice_session.BellaAgent"), \
             patch("services.voice_session.db", test_db):
            
            from services.voice_session import VoiceSessionMetrics
            
            metrics = VoiceSessionMetrics()
            metrics.response_latencies = [100, 200, 300]
            
            assert metrics.avg_latency_ms() == 200.0
    
    def test_average_latency_empty(self, mock_websocket, test_db):
        """Test average latency with no data."""
        with patch("services.voice_session.STTService"), \
             patch("services.voice_session.TTSService"), \
             patch("services.voice_session.BellaAgent"), \
             patch("services.voice_session.db", test_db):
            
            from services.voice_session import VoiceSessionMetrics
            
            metrics = VoiceSessionMetrics()
            
            assert metrics.avg_latency_ms() == 0.0


class TestVoiceSessionManager:
    """Test VoiceSessionManager functionality."""
    
    def test_manager_singleton(self, test_db):
        """Test that manager is a singleton."""
        with patch("services.voice_session.db", test_db):
            from services.voice_session import get_voice_session_manager
            
            manager1 = get_voice_session_manager()
            manager2 = get_voice_session_manager()
            
            assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_create_session(self, mock_websocket, test_db, mock_google_tts):
        """Test creating a session through manager."""
        with patch("services.voice_session.STTService") as MockSTT, \
             patch("services.voice_session.TTSService") as MockTTS, \
             patch("services.voice_session.BellaAgent") as MockAgent, \
             patch("services.voice_session.db", test_db):
            
            mock_stt = AsyncMock()
            mock_stt.start = AsyncMock()
            MockSTT.return_value = mock_stt
            
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=b"audio")
            MockTTS.return_value = mock_tts
            
            from services.voice_session import VoiceSessionManager
            
            manager = VoiceSessionManager()
            
            session = await manager.create_session(
                websocket=mock_websocket,
                caller_name="Manager Test"
            )
            
            assert session is not None
            assert session.caller_name == "Manager Test"
    
    def test_get_all_sessions(self, test_db):
        """Test getting all active sessions."""
        with patch("services.voice_session.db", test_db):
            from services.voice_session import VoiceSessionManager
            
            manager = VoiceSessionManager()
            
            sessions = manager.get_all_sessions()
            
            assert isinstance(sessions, list)
