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
    segment TEXT NOT NULL,
    station_upstream TEXT,
    station_downstream TEXT,
    pipe_diameter_in REAL,
    pipe_length_m REAL,
    pressure_upstream_psi REAL,
    pressure_downstream_psi REAL,
    pressure_drop_psi REAL,
    flow_upstream_lps REAL,
    flow_downstream_lps REAL,
    flow_diff_lps REAL,
    vibration_g REAL,
    acoustic_db REAL,
    is_anomaly INTEGER,
    predicted_severity TEXT,
    severity_confidence REAL,
    predicted_location TEXT,
    location_confidence REAL,
    estimated_water_loss_lpm REAL,
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
    leaks = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE predicted_severity != 'none'"
    ).fetchone()[0]
    total_loss = conn.execute(
        "SELECT COALESCE(SUM(estimated_water_loss_lpm),0) FROM predictions "
        "WHERE predicted_severity != 'none'"
    ).fetchone()[0]
    conn.close()
    return {"total_readings": total, "leak_events": leaks, "total_water_loss_lpm": total_loss}


def clear_all():
    conn = get_connection()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
