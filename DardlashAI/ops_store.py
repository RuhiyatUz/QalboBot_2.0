# -*- coding: utf-8 -*-
"""Локальный SQLite-журнал: last_seen и события (без pickle)."""
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("OPS_DB_PATH", "ops.sqlite3"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            last_seen REAL NOT NULL,
            lang TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            user_id INTEGER,
            kind TEXT NOT NULL,
            detail TEXT
        )
        """
    )
    return conn


def touch_user(user_id: int, lang: Optional[str] = None) -> None:
    now = time.time()
    with _connect() as conn:
        row = conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            conn.execute(
                "UPDATE users SET last_seen = ?, lang = COALESCE(?, lang) WHERE user_id = ?",
                (now, lang, user_id),
            )
        else:
            conn.execute(
                "INSERT INTO users (user_id, last_seen, lang, created_at) VALUES (?, ?, ?, ?)",
                (user_id, now, lang, now),
            )


def log_event(user_id: Optional[int], kind: str, detail: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (ts, user_id, kind, detail) VALUES (?, ?, ?, ?)",
            (time.time(), user_id, kind, detail[:500]),
        )
