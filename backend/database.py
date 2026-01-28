"""
SQLite Database for Bella Voice AI.
Handles calls, bookings, transcripts, and analytics.
Designed for easy migration to Supabase later.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent / "data" / "bella.db"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database with all required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # ==================== CALLS TABLE ====================
        # Stores call session metadata and analytics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY,
                caller_name TEXT NOT NULL,
                phone_number TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration_seconds INTEGER,
                status TEXT DEFAULT 'active',  -- active, completed, transferred, error
                end_reason TEXT,  -- user_ended, transferred, error, timeout
                turn_count INTEGER DEFAULT 0,
                interruption_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                total_tts_ms INTEGER DEFAULT 0,
                total_stt_ms INTEGER DEFAULT 0,
                total_llm_ms INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0.0,
                metadata TEXT  -- JSON for additional data
            )
        """)
        
        # ==================== BOOKINGS TABLE ====================
        # Restaurant bookings (replaces tickets.json)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT,
                name TEXT NOT NULL,
                party_size INTEGER NOT NULL,
                booking_date DATE NOT NULL,
                booking_time TIME NOT NULL,
                notes TEXT,
                status TEXT DEFAULT 'confirmed',  -- confirmed, cancelled, completed, no_show
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cancelled_at TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            )
        """)
        
        # ==================== TRANSCRIPTS TABLE ====================
        # Store conversation history for analytics and debugging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                role TEXT NOT NULL,  -- user, assistant, system
                content TEXT NOT NULL,
                intent TEXT,  -- Detected intent for user messages
                latency_ms INTEGER,  -- Response latency for assistant messages
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(id)
            )
        """)
        
        # ==================== CALL_ANALYTICS TABLE ====================
        # Aggregated analytics per call (computed at end of call)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS call_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT UNIQUE NOT NULL,
                avg_response_latency_ms INTEGER,
                max_response_latency_ms INTEGER,
                intents_detected TEXT,  -- JSON array of intents
                booking_made BOOLEAN DEFAULT FALSE,
                booking_id INTEGER,
                sentiment_score REAL,  -- Future: sentiment analysis
                success_score REAL,  -- Future: call success metric
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (call_id) REFERENCES calls(id),
                FOREIGN KEY (booking_id) REFERENCES bookings(id)
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_calls_started ON calls(started_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookings_date ON bookings(booking_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookings_name ON bookings(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_call ON transcripts(call_id)")
        
        logger.info("Database initialized successfully")


# ==================== CALL OPERATIONS ====================

def create_call(call_id: str, caller_name: str, phone_number: Optional[str] = None) -> Dict[str, Any]:
    """Create a new call record."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO calls (id, caller_name, phone_number)
            VALUES (?, ?, ?)
        """, (call_id, caller_name, phone_number))
        
        return {
            "id": call_id,
            "caller_name": caller_name,
            "phone_number": phone_number,
            "status": "active"
        }


def update_call(call_id: str, **kwargs) -> bool:
    """Update call fields."""
    if not kwargs:
        return False
    
    allowed_fields = {
        "ended_at", "duration_seconds", "status", "end_reason",
        "turn_count", "interruption_count", "error_count",
        "total_tts_ms", "total_stt_ms", "total_llm_ms",
        "estimated_cost_usd", "metadata"
    }
    
    fields = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not fields:
        return False
    
    set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [call_id]
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE calls SET {set_clause} WHERE id = ?", values)
        return cursor.rowcount > 0


def end_call(
    call_id: str,
    end_reason: str,
    turn_count: int = 0,
    interruption_count: int = 0,
    error_count: int = 0,
    total_tts_ms: int = 0,
    total_stt_ms: int = 0,
    total_llm_ms: int = 0
) -> Dict[str, Any]:
    """End a call and compute analytics."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get call start time
        cursor.execute("SELECT started_at FROM calls WHERE id = ?", (call_id,))
        row = cursor.fetchone()
        if not row:
            return {"error": "Call not found"}
        
        started_at = datetime.fromisoformat(row["started_at"])
        ended_at = datetime.utcnow()  # Use UTC to match CURRENT_TIMESTAMP
        duration_seconds = int((ended_at - started_at).total_seconds())
        
        # Clamp to reasonable values (0 to 4 hours max)
        duration_seconds = max(0, min(duration_seconds, 14400))
        
        # Estimate cost (rough approximation)
        # STT: ~$0.006/min, TTS: ~$0.000015/char, LLM: ~$0.002/1K tokens
        estimated_cost = (duration_seconds / 60) * 0.01  # Rough estimate
        
        # Update call
        cursor.execute("""
            UPDATE calls SET
                ended_at = ?,
                duration_seconds = ?,
                status = 'completed',
                end_reason = ?,
                turn_count = ?,
                interruption_count = ?,
                error_count = ?,
                total_tts_ms = ?,
                total_stt_ms = ?,
                total_llm_ms = ?,
                estimated_cost_usd = ?
            WHERE id = ?
        """, (
            ended_at.isoformat(), duration_seconds, end_reason,
            turn_count, interruption_count, error_count,
            total_tts_ms, total_stt_ms, total_llm_ms,
            estimated_cost, call_id
        ))
        
        # Compute and store analytics
        cursor.execute("""
            SELECT 
                AVG(latency_ms) as avg_latency,
                MAX(latency_ms) as max_latency
            FROM transcripts 
            WHERE call_id = ? AND role = 'assistant' AND latency_ms IS NOT NULL
        """, (call_id,))
        latency_row = cursor.fetchone()
        
        cursor.execute("""
            SELECT DISTINCT intent FROM transcripts 
            WHERE call_id = ? AND intent IS NOT NULL
        """, (call_id,))
        intents = [row["intent"] for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT id FROM bookings WHERE call_id = ? LIMIT 1
        """, (call_id,))
        booking_row = cursor.fetchone()
        
        cursor.execute("""
            INSERT OR REPLACE INTO call_analytics 
            (call_id, avg_response_latency_ms, max_response_latency_ms, 
             intents_detected, booking_made, booking_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            call_id,
            int(latency_row["avg_latency"]) if latency_row["avg_latency"] else None,
            int(latency_row["max_latency"]) if latency_row["max_latency"] else None,
            json.dumps(intents),
            booking_row is not None,
            booking_row["id"] if booking_row else None
        ))
        
        return {
            "call_id": call_id,
            "duration_seconds": duration_seconds,
            "turn_count": turn_count,
            "intents": intents,
            "booking_made": booking_row is not None,
            "estimated_cost_usd": estimated_cost
        }


def get_call(call_id: str) -> Optional[Dict[str, Any]]:
    """Get call by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM calls WHERE id = ?", (call_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_active_calls() -> List[Dict[str, Any]]:
    """Get all active calls."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM calls WHERE status = 'active'")
        return [dict(row) for row in cursor.fetchall()]


# ==================== BOOKING OPERATIONS ====================

def create_booking(
    name: str,
    party_size: int,
    booking_date: str,
    booking_time: str,
    notes: Optional[str] = None,
    call_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new booking."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO bookings (call_id, name, party_size, booking_date, booking_time, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (call_id, name, party_size, booking_date, booking_time, notes))
        
        booking_id = cursor.lastrowid
        
        return {
            "id": booking_id,
            "name": name,
            "party_size": party_size,
            "booking_date": booking_date,
            "booking_time": booking_time,
            "notes": notes,
            "status": "confirmed"
        }


def get_booking(booking_id: int) -> Optional[Dict[str, Any]]:
    """Get booking by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def find_bookings(name: Optional[str] = None, date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find bookings by name or date."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        conditions = ["status != 'cancelled'"]
        params = []
        
        if name:
            conditions.append("LOWER(name) = LOWER(?)")
            params.append(name)
        if date:
            conditions.append("booking_date = ?")
            params.append(date)
        
        query = f"SELECT * FROM bookings WHERE {' AND '.join(conditions)}"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def update_booking(
    booking_id: int,
    party_size: Optional[int] = None,
    booking_date: Optional[str] = None,
    booking_time: Optional[str] = None,
    name: Optional[str] = None,
    notes: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Update a booking."""
    updates = {}
    if party_size is not None:
        updates["party_size"] = party_size
    if booking_date is not None:
        updates["booking_date"] = booking_date
    if booking_time is not None:
        updates["booking_time"] = booking_time
    if name is not None:
        updates["name"] = name
    if notes is not None:
        updates["notes"] = notes
    
    if not updates:
        return get_booking(booking_id)
    
    updates["updated_at"] = datetime.now().isoformat()
    
    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [booking_id]
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE bookings SET {set_clause} WHERE id = ?", values)
        
        if cursor.rowcount > 0:
            # Read back from the same connection before commit
            cursor.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        return None


def cancel_booking(booking_id: int) -> bool:
    """Cancel a booking."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE bookings 
            SET status = 'cancelled', cancelled_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (booking_id,))
        return cursor.rowcount > 0


# ==================== TRANSCRIPT OPERATIONS ====================

def add_transcript(
    call_id: str,
    turn_number: int,
    role: str,
    content: str,
    intent: Optional[str] = None,
    latency_ms: Optional[int] = None
) -> int:
    """Add a transcript entry."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transcripts (call_id, turn_number, role, content, intent, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (call_id, turn_number, role, content, intent, latency_ms))
        return cursor.lastrowid


def get_transcripts(call_id: str) -> List[Dict[str, Any]]:
    """Get all transcripts for a call."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM transcripts 
            WHERE call_id = ? 
            ORDER BY turn_number, timestamp
        """, (call_id,))
        return [dict(row) for row in cursor.fetchall()]


# ==================== ANALYTICS OPERATIONS ====================

def get_call_analytics(call_id: str) -> Optional[Dict[str, Any]]:
    """Get analytics for a specific call."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ca.*, c.caller_name, c.duration_seconds, c.turn_count
            FROM call_analytics ca
            JOIN calls c ON c.id = ca.call_id
            WHERE ca.call_id = ?
        """, (call_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result.get("intents_detected"):
                result["intents_detected"] = json.loads(result["intents_detected"])
            return result
        return None


def get_daily_stats(date: Optional[str] = None) -> Dict[str, Any]:
    """Get aggregated daily statistics."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Call stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_calls,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_calls,
                AVG(duration_seconds) as avg_duration,
                SUM(estimated_cost_usd) as total_cost
            FROM calls
            WHERE DATE(started_at) = ?
        """, (date,))
        call_stats = dict(cursor.fetchone())
        
        # Booking stats
        cursor.execute("""
            SELECT COUNT(*) as total_bookings
            FROM bookings
            WHERE DATE(created_at) = ? AND status = 'confirmed'
        """, (date,))
        booking_stats = dict(cursor.fetchone())
        
        return {
            "date": date,
            **call_stats,
            **booking_stats
        }


# Initialize on import
init_database()
