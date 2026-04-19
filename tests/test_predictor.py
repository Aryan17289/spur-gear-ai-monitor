"""Unit tests for GearPredictor"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.predictor import GearPredictor

@pytest.fixture
def predictor():
    """Create predictor instance for testing"""
    model_path = "models/spur_gear_svm_model.pkl"
    scaler_path = "models/spur_gear_scaler.pkl"
    return GearPredictor(model_path, scaler_path)

def test_predict_normal_operation(predictor):
    """Test prediction with normal operating parameters"""
    result = predictor.predict(
        speed=1050,
        torque=110,
        vibration=1.4,
        temperature=57,
        shock=1.7,
        noise=74
    )
    
    assert 'prediction' in result
    assert 'probability' in result
    assert 'probability_pct' in result
    assert 0 <= result['probability'] <= 1
    assert 0 <= result['probability_pct'] <= 100

def test_predict_high_risk(predictor):
    """Test prediction with high-risk parameters"""
    result = predictor.predict(
        speed=2800,
        torque=380,
        vibration=9.5,
        temperature=115,
        shock=5.5,
        noise=98
    )
    
    # High-risk parameters should yield high probability
    assert result['probability_pct'] > 50

def test_risk_level_classification(predictor):
    """Test risk level classification"""
    # Low risk
    label, color = predictor.get_risk_level(20)
    assert label == "LOW RISK"
    
    # Moderate risk
    label, color = predictor.get_risk_level(40)
    assert label == "MODERATE RISK"
    
    # High risk
    label, color = predictor.get_risk_level(70)
    assert label == "HIGH RISK"
    
    # Critical risk
    label, color = predictor.get_risk_level(90)
    assert label == "CRITICAL RISK"

def test_rul_calculation(predictor):
    """Test RUL calculation"""
    result = predictor.calculate_rul(
        probability=0.3,
        max_cycles=1000,
        speed=1500
    )
    
    assert 'health_score' in result
    assert 'rul_cycles' in result
    assert 'rul_hours' in result
    assert result['health_score'] == 0.7
    assert result['rul_cycles'] == 700
    assert result['rul_hours'] > 0

def test_input_validation_ranges(predictor):
    """Test that predictor handles edge cases"""
    # Minimum values
    result_min = predictor.predict(500, 50, 0.5, 30, 0.1, 50)
    assert result_min is not None
    
    # Maximum values
    result_max = predictor.predict(3000, 400, 10.0, 120, 6.0, 100)
    assert result_max is not None
