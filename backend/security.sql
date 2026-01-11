-- SQLite schema for Bella Cucina voice bot
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

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
