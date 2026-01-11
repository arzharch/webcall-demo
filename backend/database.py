import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiosqlite

from backend.config import get_settings


CREATE_TABLES_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS call_sessions (
    session_id TEXT PRIMARY KEY,
    caller_name TEXT NOT NULL,
    signaling_token TEXT NOT NULL,
    token_expires_at INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    last_agent_message TEXT,
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    stt_seconds REAL DEFAULT 0,
    tts_characters INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS call_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    FOREIGN KEY(session_id) REFERENCES call_sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    caller_name TEXT NOT NULL,
    reservation_date TEXT NOT NULL,
    reservation_time TEXT NOT NULL,
    party_size INTEGER NOT NULL,
    status TEXT NOT NULL,
    special_requests TEXT,
    event_tag TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(reservation_date, reservation_time, caller_name)
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    item_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES call_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reservations_date ON reservations(reservation_date, reservation_time);
CREATE INDEX IF NOT EXISTS idx_messages_session ON call_messages(session_id);
"""


class Database:
    """Thin async wrapper around SQLite for call + reservation storage."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            async with aiosqlite.connect(self._db_path) as db:
                await db.executescript(CREATE_TABLES_SQL)
                await db.commit()
            self._initialized = True

    async def create_call_session(
        self,
        session_id: str,
        caller_name: str,
        signaling_token: str,
        token_expires_at: int,
    ) -> None:
        await self._execute(
            """
            INSERT INTO call_sessions (
                session_id, caller_name, signaling_token, token_expires_at,
                status, started_at
            ) VALUES (?, ?, ?, ?, 'active', ?)
            """,
            (
                session_id,
                caller_name,
                signaling_token,
                token_expires_at,
                datetime.utcnow().isoformat(),
            ),
        )

    async def close_call_session(
        self,
        session_id: str,
        status: str,
        last_agent_message: Optional[str],
        cost_snapshot: Dict[str, Any],
    ) -> None:
        await self._execute(
            """
            UPDATE call_sessions
            SET status = ?,
                ended_at = ?,
                last_agent_message = ?,
                total_prompt_tokens = ?,
                total_completion_tokens = ?,
                stt_seconds = ?,
                tts_characters = ?,
                estimated_cost_usd = ?
            WHERE session_id = ?
            """,
            (
                status,
                datetime.utcnow().isoformat(),
                last_agent_message,
                cost_snapshot.get("prompt_tokens", 0),
                cost_snapshot.get("completion_tokens", 0),
                cost_snapshot.get("stt_seconds", 0.0),
                cost_snapshot.get("tts_characters", 0),
                cost_snapshot.get("estimated_usd", 0.0),
                session_id,
            ),
        )

    async def record_message(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
    ) -> None:
        await self._execute(
            """
            INSERT INTO call_messages (session_id, role, content, created_at, token_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                datetime.utcnow().isoformat(),
                token_count,
            ),
        )

    async def create_reservation(
        self,
        session_id: str,
        caller_name: str,
        reservation_date: str,
        reservation_time: str,
        party_size: int,
        special_requests: Optional[str],
        event_tag: Optional[str],
        status: str,
    ) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO reservations (
                    session_id, caller_name, reservation_date, reservation_time,
                    party_size, status, special_requests, event_tag,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    caller_name,
                    reservation_date,
                    reservation_time,
                    party_size,
                    status,
                    special_requests,
                    event_tag,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def upsert_order(
        self,
        session_id: str,
        item_id: str,
        item_name: str,
        quantity: int,
        unit_price: float,
    ) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO orders (session_id, item_id, item_name, quantity, unit_price, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    item_id,
                    item_name,
                    quantity,
                    unit_price,
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM call_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    async def verify_signaling_token(self, session_id: str, token: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT signaling_token, token_expires_at FROM call_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False
            stored_token, expires_at = row
            if stored_token != token:
                return False
            return int(datetime.utcnow().timestamp()) <= int(expires_at)

    async def list_reservations_for_date(self, reservation_date: str) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM reservations WHERE reservation_date = ?",
                (reservation_date,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def count_reservations_for_slot(
        self, reservation_date: str, reservation_time: str
    ) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT COUNT(*)
                FROM reservations
                WHERE reservation_date = ?
                  AND reservation_time = ?
                  AND status != 'cancelled'
                """,
                (reservation_date, reservation_time),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def update_reservation_status(
        self, reservation_id: int, status: str
    ) -> None:
        await self._execute(
            """
            UPDATE reservations
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, datetime.utcnow().isoformat(), reservation_id),
        )

    async def _execute(self, query: str, params: tuple) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(query, params)
            await db.commit()


_database: Optional[Database] = None


def get_database() -> Database:
    global _database
    if _database is None:
        settings = get_settings()
        _database = Database(settings.SQLITE_DB_PATH)
    return _database
*** End of File