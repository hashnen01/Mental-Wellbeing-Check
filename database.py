import sqlite3
from datetime import datetime

from config import DATABASE_PATH


def init_db():
    """Create the assessments table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            overall_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            sleep_score REAL,
            activity_score REAL,
            nutrition_score REAL,
            social_score REAL,
            balance_score REAL
        )
    """)
    conn.commit()
    conn.close()


def save_assessment(overall_score, risk_level, dimension_scores):
    """Insert a new assessment into the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO assessments
           (date, overall_score, risk_level, sleep_score, activity_score,
            nutrition_score, social_score, balance_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            round(overall_score, 1),
            risk_level,
            dimension_scores.get("sleep", 0),
            dimension_scores.get("activity", 0),
            dimension_scores.get("nutrition", 0),
            dimension_scores.get("social", 0),
            dimension_scores.get("balance", 0),
        ),
    )
    conn.commit()
    conn.close()


def get_history():
    """Fetch all past assessments, most recent first."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM assessments ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_history():
    """Delete all assessment records."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM assessments")
    conn.commit()
    conn.close()
