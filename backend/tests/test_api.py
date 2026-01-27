"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def client(test_db, mock_redis):
    """Create test client with mocked dependencies."""
    # Patch database module path before importing app
    with patch("main.db", test_db):
        from main import app
        with TestClient(app) as client:
            yield client


class TestRootEndpoint:
    """Test root API endpoint."""
    
    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns status."""
        with patch("main.get_health_checker") as mock_checker:
            mock_health = MagicMock()
            mock_health.check_all = AsyncMock(return_value={
                "healthy": True,
                "checks": {"database": True}
            })
            mock_checker.return_value = mock_health
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


class TestCallEndpoints:
    """Test call-related endpoints."""
    
    def test_list_calls(self, client, test_db):
        """Test listing calls."""
        # Create some test calls
        test_db.create_call("call-1", "User 1")
        test_db.create_call("call-2", "User 2")
        
        response = client.get("/calls")
        
        assert response.status_code == 200
        data = response.json()
        assert "calls" in data
    
    def test_get_call(self, client, test_db):
        """Test getting a specific call."""
        test_db.create_call("test-call-123", "Test User")
        
        response = client.get("/calls/test-call-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-call-123"
        assert data["caller_name"] == "Test User"
    
    def test_get_nonexistent_call(self, client):
        """Test getting a call that doesn't exist."""
        response = client.get("/calls/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_get_call_transcripts(self, client, test_db):
        """Test getting call transcripts."""
        test_db.create_call("transcript-call", "Transcript User")
        test_db.add_transcript("transcript-call", 1, "user", "Hello")
        test_db.add_transcript("transcript-call", 1, "assistant", "Hi there!")
        
        response = client.get("/calls/transcript-call/transcripts")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["transcripts"]) == 2


class TestBookingEndpoints:
    """Test booking-related endpoints."""
    
    def test_create_booking(self, client, test_db):
        """Test creating a booking via API."""
        response = client.post("/bookings", json={
            "name": "Alice",
            "party_size": 4,
            "booking_date": "2026-02-15",
            "booking_time": "19:00",
            "notes": "Window seat"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Alice"
        assert data["party_size"] == 4
    
    def test_list_bookings(self, client, test_db):
        """Test listing bookings."""
        test_db.create_booking("Bob", 2, "2026-02-15", "18:00")
        test_db.create_booking("Carol", 3, "2026-02-15", "19:00")
        
        response = client.get("/bookings")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
    
    def test_list_bookings_by_name(self, client, test_db):
        """Test filtering bookings by name."""
        test_db.create_booking("David", 2, "2026-02-15", "18:00")
        test_db.create_booking("Eve", 3, "2026-02-15", "19:00")
        
        response = client.get("/bookings?name=David")
        
        assert response.status_code == 200
        data = response.json()
        for booking in data:
            assert booking["name"] == "David"
    
    def test_get_booking(self, client, test_db):
        """Test getting a specific booking."""
        created = test_db.create_booking("Frank", 2, "2026-02-15", "20:00")
        
        response = client.get(f"/bookings/{created['id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Frank"
    
    def test_get_nonexistent_booking(self, client):
        """Test getting a booking that doesn't exist."""
        response = client.get("/bookings/99999")
        
        assert response.status_code == 404
    
    def test_cancel_booking(self, client, test_db):
        """Test canceling a booking."""
        created = test_db.create_booking("Grace", 4, "2026-02-15", "18:00")
        
        response = client.delete(f"/bookings/{created['id']}")
        
        assert response.status_code == 200


class TestStatsEndpoint:
    """Test statistics endpoint."""
    
    def test_get_stats(self, client, test_db):
        """Test getting daily stats."""
        # Create a call first so stats have data
        test_db.create_call("stats-test-call", "Stats User")
        test_db.end_call("stats-test-call", "user_ended", 1, 0, 0, 100, 100, 100)
        
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "date" in data
        assert "total_calls" in data


class TestValidation:
    """Test input validation."""
    
    def test_booking_party_size_validation(self, client):
        """Test party size validation."""
        # Too small
        response = client.post("/bookings", json={
            "name": "Test",
            "party_size": 0,
            "booking_date": "2026-02-15",
            "booking_time": "19:00"
        })
        assert response.status_code == 422
        
        # Too large
        response = client.post("/bookings", json={
            "name": "Test",
            "party_size": 25,
            "booking_date": "2026-02-15",
            "booking_time": "19:00"
        })
        assert response.status_code == 422
    
    def test_booking_name_required(self, client):
        """Test name is required for booking."""
        response = client.post("/bookings", json={
            "party_size": 4,
            "booking_date": "2026-02-15",
            "booking_time": "19:00"
        })
        assert response.status_code == 422
