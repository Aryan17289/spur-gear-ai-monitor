"""Unit tests for GearHistoryDB"""
import pytest
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import GearHistoryDB

@pytest.fixture
def test_db():
    """Create temporary test database"""
    db_path = "tests/test_gear_history.db"
    db = GearHistoryDB(db_path)
    yield db
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

def test_init_db(test_db):
    """Test database initialization"""
    # Database should be created
    assert os.path.exists(test_db.db_path)

def test_log_reading(test_db):
    """Test logging a reading"""
    test_db.log_reading(
        gear_type="Spur Gear A",
        speed=1050,
        torque=110,
        vibration=1.4,
        temperature=57,
        shock=1.7,
        noise=74,
        max_cycles=1000,
        fail_prob=25.5,
        prediction=0,
        risk_label="LOW RISK",
        health_score=0.745,
        rul_cycles=745,
        rul_hours=11.9
    )
    
    # Load and verify
    df = test_db.load_history()
    assert len(df) == 1
    assert df.iloc[0]['gear_type'] == "Spur Gear A"
    assert df.iloc[0]['speed'] == 1050

def test_load_empty_history(test_db):
    """Test loading empty history"""
    df = test_db.load_history()
    assert df.empty

def test_clear_history(test_db):
    """Test clearing history"""
    # Add some data
    test_db.log_reading("Spur Gear A", 1050, 110, 1.4, 57, 1.7, 74,
                       1000, 25.5, 0, "LOW RISK", 0.745, 745, 11.9)
    
    # Clear
    test_db.clear_history()
    
    # Verify empty
    df = test_db.load_history()
    assert df.empty
