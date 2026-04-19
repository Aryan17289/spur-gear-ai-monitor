"""ML model prediction utilities"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class GearPredictor:
    """Handles gear failure prediction using trained SVM model"""
    
    def __init__(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = [
            "Speed_RPM", "Torque_Nm", "Vibration_mm_s",
            "Temperature_C", "Shock_Load_g", "Noise_dB"
        ]
    
    def predict(self, speed, torque, vibration, temperature, shock, noise):
        """Predict failure probability and classification"""
        # Create input dataframe
        input_df = pd.DataFrame(
            [[speed, torque, vibration, temperature, shock, noise]],
            columns=self.feature_names
        )
        
        # Scale input
        input_scaled = self.scaler.transform(input_df)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0][1]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'probability_pct': float(probability * 100),
            'input_scaled': input_scaled
        }
    
    def get_risk_level(self, probability_pct):
        """Determine risk level from probability"""
        if probability_pct < 30:
            return "LOW RISK", "#16a34a"
        elif probability_pct < 55:
            return "MODERATE RISK", "#d97706"
        elif probability_pct < 80:
            return "HIGH RISK", "#ea580c"
        else:
            return "CRITICAL RISK", "#dc2626"
    
    def calculate_rul(self, probability, max_cycles, speed):
        """Calculate Remaining Useful Life"""
        health_score = 1.0 - probability
        rul_cycles = max(0, health_score * max_cycles)
        rul_minutes = rul_cycles / speed if speed > 0 else 0
        rul_hours = rul_minutes / 60
        
        # Confidence band ±10%
        rul_low = max(0, rul_cycles * 0.90)
        rul_high = rul_cycles * 1.10
        
        # RUL health label
        if health_score > 0.70:
            rul_label, rul_color = "GOOD", "#16a34a"
        elif health_score > 0.45:
            rul_label, rul_color = "DEGRADING", "#d97706"
        elif health_score > 0.20:
            rul_label, rul_color = "CRITICAL", "#ea580c"
        else:
            rul_label, rul_color = "END OF LIFE", "#dc2626"
        
        return {
            'health_score': health_score,
            'rul_cycles': rul_cycles,
            'rul_hours': rul_hours,
            'rul_low': rul_low,
            'rul_high': rul_high,
            'rul_label': rul_label,
            'rul_color': rul_color
        }
