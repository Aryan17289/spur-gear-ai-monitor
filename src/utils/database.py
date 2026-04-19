"""Database operations for gear history logging"""
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

class GearHistoryDB:
    """Manages gear history database operations"""
    
    def __init__(self, db_path: str = "data/gear_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gear_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                gear_type   TEXT,
                speed       REAL,
                torque      REAL,
                vibration   REAL,
                temperature REAL,
                shock       REAL,
                noise       REAL,
                max_cycles  INTEGER,
                fail_prob   REAL,
                prediction  INTEGER,
                risk_label  TEXT,
                health_score REAL,
                rul_cycles  REAL,
                rul_hours   REAL
            )
        """)
        con.commit()
        con.close()
    
    def log_reading(self, gear_type, speed, torque, vibration, temperature, 
                   shock, noise, max_cycles, fail_prob, prediction, 
                   risk_label, health_score, rul_cycles, rul_hours):
        """Log a new gear reading"""
        con = sqlite3.connect(self.db_path)
        con.execute("""
            INSERT INTO gear_log
                (timestamp, gear_type, speed, torque, vibration, temperature, 
                 shock, noise, max_cycles, fail_prob, prediction, risk_label, 
                 health_score, rul_cycles, rul_hours)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              gear_type, speed, torque, vibration, temperature, shock, noise,
              max_cycles, fail_prob, prediction, risk_label, health_score,
              rul_cycles, rul_hours))
        con.commit()
        con.close()
    
    def load_history(self) -> pd.DataFrame:
        """Load all historical readings"""
        con = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM gear_log ORDER BY timestamp DESC", con)
        con.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    
    def clear_history(self):
        """Clear all historical data"""
        con = sqlite3.connect(self.db_path)
        con.execute("DELETE FROM gear_log")
        con.commit()
        con.close()
