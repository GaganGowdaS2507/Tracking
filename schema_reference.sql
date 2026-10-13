-- SQLite schema is initialized automatically by app.py.
-- This file is informational only.
CREATE TABLE members (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 member_code TEXT UNIQUE NOT NULL,
 name TEXT NOT NULL,
 email TEXT DEFAULT '',
 notes TEXT DEFAULT '',
 embedding TEXT DEFAULT '',
 photo_path TEXT DEFAULT '',
 created_at TEXT NOT NULL
);
CREATE TABLE events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 member_id INTEGER,
 member_name TEXT NOT NULL,
 member_code TEXT DEFAULT '',
 event_type TEXT NOT NULL,
 event_time TEXT NOT NULL,
 confidence REAL DEFAULT 0,
 source TEXT DEFAULT '',
 reviewed INTEGER DEFAULT 0
);
CREATE TABLE visits (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 member_id INTEGER NOT NULL,
 entry_time TEXT NOT NULL,
 exit_time TEXT DEFAULT '',
 status TEXT NOT NULL DEFAULT 'inside'
);
