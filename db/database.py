"""
AquaGuard AI — SQLite Logging Layer
--------------------------------------
Stores every prediction made through the Data Entry Dashboard so the
Analysis Dashboard can show history, trends, and stats.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "aquaguard.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    pressure_bar REAL,
    flow_rate_lps REAL,
    temperature_c REAL,
    is_anomaly INTEGER,
    predicted_status TEXT,
    status_confidence REAL,
    impact_level TEXT,
    impact_drop_pct REAL,
    notes TEXT
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_prediction(record: dict):
    conn = get_connection()
    record = {**record, "created_at": datetime.now().isoformat(timespec="seconds")}
    cols = ", ".join(record.keys())
    placeholders = ", ".join(["?"] * len(record))
    conn.execute(
        f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
        list(record.values()),
    )
    conn.commit()
    conn.close()


def fetch_all(limit=500):
    conn = get_connection()
    cur = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    conn.close()
    return cols, rows


def fetch_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    incidents = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE predicted_status != 'normal'"
    ).fetchone()[0]
    bursts = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE predicted_status = 'burst'"
    ).fetchone()[0]
    conn.close()
    return {"total_readings": total, "incident_events": incidents, "burst_events": bursts}


def clear_all():
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
