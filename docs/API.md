# API Documentation

## GearPredictor Class

### `__init__(model_path, scaler_path)`
Initialize the predictor with trained model and scaler.

**Parameters:**
- `model_path` (str): Path to pickled SVM model
- `scaler_path` (str): Path to pickled StandardScaler

### `predict(speed, torque, vibration, temperature, shock, noise)`
Predict gear failure probability.

**Parameters:**
- `speed` (float): Rotation speed in RPM (500-3000)
- `torque` (float): Applied torque in Nm (50-400)
- `vibration` (float): Vibration in mm/s (0.5-10.0)
- `temperature` (float): Temperature in °C (30-120)
- `shock` (float): Shock load in g (0.1-6.0)
- `noise` (float): Noise level in dB (50-100)

**Returns:**
```python
{
    'prediction': int,           # 0 or 1
    'probability': float,        # 0.0 to 1.0
    'probability_pct': float,    # 0 to 100
    'input_scaled': ndarray      # Scaled features
}
```

### `get_risk_level(probability_pct)`
Classify risk level from probability.

**Returns:**
```python
(risk_label: str, color: str)
# e.g., ("HIGH RISK", "#ea580c")
```

### `calculate_rul(probability, max_cycles, speed)`
Calculate Remaining Useful Life.

**Returns:**
```python
{
    'health_score': float,
    'rul_cycles': float,
    'rul_hours': float,
    'rul_low': float,
    'rul_high': float,
    'rul_label': str,
    'rul_color': str
}
```

## GearHistoryDB Class

### `__init__(db_path)`
Initialize database connection.

### `log_reading(...)`
Log a new gear reading to database.

### `load_history()`
Load all historical readings as DataFrame.

### `clear_history()`
Delete all historical data.

## PDF Report Functions

### `build_pdf_report(gear_data, prediction_data, shap_fig)`
Generate comprehensive PDF report.

**Parameters:**
- `gear_data` (dict): Gear configuration and parameters
- `prediction_data` (dict): Prediction results
- `shap_fig` (matplotlib.Figure): Optional SHAP chart

**Returns:**
- `bytes`: PDF file content

## Styling Functions

### `style_ax(ax, fig)`
Apply dark theme to matplotlib axes.

### `bar_label(ax, bars, values, fmt)`
Add value labels to bar charts.
