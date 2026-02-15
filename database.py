# ──────────────────────────────────────────────
# SQLite Logging Layer — Crowd Data Persistence
# ──────────────────────────────────────────────

import sqlite3
import threading
from datetime import datetime, timezone

import config

_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the crowd_logs table if it does not exist."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crowd_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                current_count   INTEGER NOT NULL,
                density_ratio   REAL    NOT NULL,
                growth_rate     REAL    NOT NULL,
                risk_level      TEXT    NOT NULL,
                risk_score      REAL    NOT NULL,
                surge_flag      INTEGER NOT NULL,
                duration_high   REAL    NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()


def log_reading(assessment: dict):
    """Insert a single crowd reading into the database."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            """
            INSERT INTO crowd_logs
                (timestamp, current_count, density_ratio, growth_rate,
                 risk_level, risk_score, surge_flag, duration_high)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                assessment["current_count"],
                assessment["density_ratio"],
                assessment["growth_rate"],
                assessment["risk_level"],
                assessment["risk_score"],
                int(assessment["surge_flag"]),
                assessment["duration_in_high_state"],
            ),
        )
        conn.commit()
        conn.close()


def get_history(minutes: int = 2) -> list[dict]:
    """Return crowd log rows from the last N minutes."""
    with _lock:
        conn = _get_connection()
        rows = conn.execute(
            """
            SELECT timestamp, current_count, density_ratio, growth_rate,
                   risk_level, risk_score, surge_flag, duration_high
            FROM crowd_logs
            WHERE timestamp >= datetime('now', ?)
            ORDER BY id ASC
            """,
            (f"-{minutes} minutes",),
        ).fetchall()
        conn.close()

    return [dict(r) for r in rows]
