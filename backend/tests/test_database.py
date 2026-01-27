"""
Tests for the Database module.
Uses the actual database functions from backend/database.py.
"""
import pytest
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatabaseInitialization:
    """Test database initialization."""
    
    def test_database_tables_created(self, test_db):
        """Test that all required tables are created."""
        with test_db.get_db() as conn:
            cursor = conn.cursor()
            
            # Check calls table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='calls'")
            assert cursor.fetchone() is not None
            
            # Check bookings table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bookings'")
            assert cursor.fetchone() is not None
            
            # Check transcripts table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transcripts'")
            assert cursor.fetchone() is not None
            
            # Check call_analytics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='call_analytics'")
            assert cursor.fetchone() is not None


class TestCallOperations:
    """Test call CRUD operations."""
    
    def test_create_call(self, test_db):
        """Test creating a new call record."""
        call_id = "test-session-123"
        caller_name = "John Doe"
        
        result = test_db.create_call(call_id, caller_name)
        
        assert result is not None
        assert result["id"] == call_id
        assert result["caller_name"] == caller_name
        assert result["status"] == "active"
    
    def test_create_call_with_phone(self, test_db):
        """Test creating call with phone number."""
        call_id = "test-phone-123"
        
        result = test_db.create_call(call_id, "Jane", phone_number="+1234567890")
        
        assert result["phone_number"] == "+1234567890"
    
    def test_get_call(self, test_db):
        """Test retrieving a call by ID."""
        call_id = "test-get-call-456"
        test_db.create_call(call_id, "Jane Doe")
        
        call = test_db.get_call(call_id)
        
        assert call is not None
        assert call["id"] == call_id
        assert call["caller_name"] == "Jane Doe"
    
    def test_get_nonexistent_call(self, test_db):
        """Test retrieving a non-existent call returns None."""
        call = test_db.get_call("nonexistent-session")
        assert call is None
    
    def test_update_call(self, test_db):
        """Test updating call fields."""
        call_id = "test-update-789"
        test_db.create_call(call_id, "Bob Smith")
        
        result = test_db.update_call(
            call_id,
            turn_count=5,
            interruption_count=2
        )
        
        assert result is True
        
        call = test_db.get_call(call_id)
        assert call["turn_count"] == 5
        assert call["interruption_count"] == 2
    
    def test_end_call(self, test_db):
        """Test ending a call with analytics."""
        call_id = "test-end-call-789"
        test_db.create_call(call_id, "Bob Smith")
        
        result = test_db.end_call(
            call_id=call_id,
            end_reason="user_ended",
            turn_count=5,
            interruption_count=2,
            error_count=0,
            total_tts_ms=5000,
            total_stt_ms=3000,
            total_llm_ms=8000
        )
        
        # Verify call ended
        call = test_db.get_call(call_id)
        assert call["status"] == "completed"
        assert call["ended_at"] is not None
        assert call["turn_count"] == 5
    
    def test_get_active_calls(self, test_db):
        """Test retrieving only active calls."""
        # Create active and completed calls
        test_db.create_call("active-1", "Active User 1")
        test_db.create_call("active-2", "Active User 2")
        test_db.create_call("completed-1", "Completed User")
        test_db.end_call("completed-1", "user_ended", 1, 0, 0, 100, 200, 300)
        
        active_calls = test_db.get_active_calls()
        
        # Should only get active calls
        active_ids = [c["id"] for c in active_calls]
        assert "active-1" in active_ids
        assert "active-2" in active_ids
        assert "completed-1" not in active_ids


class TestBookingOperations:
    """Test booking CRUD operations."""
    
    def test_create_booking(self, test_db):
        """Test creating a booking record."""
        booking = test_db.create_booking(
            name="Alice",
            party_size=4,
            booking_date="2026-02-01",
            booking_time="19:00",
            call_id="booking-session-123",
            notes="Window seat please"
        )
        
        assert booking is not None
        assert booking["name"] == "Alice"
        assert booking["party_size"] == 4
        assert booking["booking_date"] == "2026-02-01"
    
    def test_get_booking(self, test_db):
        """Test retrieving a booking by ID."""
        created = test_db.create_booking(
            name="Bob",
            party_size=2,
            booking_date="2026-02-02",
            booking_time="18:00"
        )
        
        booking = test_db.get_booking(created["id"])
        
        assert booking is not None
        assert booking["name"] == "Bob"
        assert booking["party_size"] == 2
    
    def test_find_bookings_by_name(self, test_db):
        """Test finding bookings by name."""
        test_db.create_booking("Charlie", 2, "2026-02-01", "18:00")
        test_db.create_booking("Charlie", 4, "2026-02-02", "19:00")
        test_db.create_booking("David", 3, "2026-02-01", "20:00")
        
        bookings = test_db.find_bookings(name="Charlie")
        
        assert len(bookings) == 2
        for b in bookings:
            assert b["name"] == "Charlie"
    
    def test_cancel_booking(self, test_db):
        """Test canceling a booking."""
        created = test_db.create_booking("Eve", 3, "2026-02-01", "20:00")
        
        result = test_db.cancel_booking(created["id"])
        
        assert result is True
        
        booking = test_db.get_booking(created["id"])
        assert booking["status"] == "cancelled"
    
    def test_update_booking(self, test_db):
        """Test updating a booking."""
        created = test_db.create_booking("Frank", 2, "2026-02-01", "18:00")
        
        result = test_db.update_booking(
            created["id"],
            party_size=4,
            booking_time="19:00"
        )
        
        # update_booking returns the updated booking dict or None
        assert result is not None
        assert result["party_size"] == 4
        assert result["booking_time"] == "19:00"


class TestTranscriptOperations:
    """Test transcript operations."""
    
    def test_add_transcript(self, test_db):
        """Test adding transcript entries."""
        call_id = "transcript-test-123"
        test_db.create_call(call_id, "Transcript User")
        
        test_db.add_transcript(call_id, 1, "user", "I want to make a reservation")
        test_db.add_transcript(call_id, 1, "assistant", "I'd be happy to help!")
        
        transcripts = test_db.get_transcripts(call_id)
        
        assert len(transcripts) == 2
        assert transcripts[0]["role"] == "user"
        assert transcripts[1]["role"] == "assistant"
    
    def test_transcript_with_intent(self, test_db):
        """Test adding transcript with detected intent."""
        call_id = "intent-test-123"
        test_db.create_call(call_id, "Intent User")
        
        test_db.add_transcript(
            call_id, 1, "user", 
            "I want to book a table",
            intent="make_booking"
        )
        
        transcripts = test_db.get_transcripts(call_id)
        assert transcripts[0]["intent"] == "make_booking"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_special_characters_in_name(self, test_db):
        """Test handling special characters in caller name."""
        call_id = "special-chars-session"
        caller_name = "O'Brien-Smith (Jr.)"
        
        test_db.create_call(call_id, caller_name)
        
        call = test_db.get_call(call_id)
        assert call["caller_name"] == caller_name
    
    def test_unicode_in_transcript(self, test_db):
        """Test handling unicode in transcripts."""
        call_id = "unicode-test"
        test_db.create_call(call_id, "Unicode User")
        
        test_db.add_transcript(call_id, 1, "user", "I'd like to book for 2 people 🎉")
        
        transcripts = test_db.get_transcripts(call_id)
        assert "🎉" in transcripts[0]["content"]
