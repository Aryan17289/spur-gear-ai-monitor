# Architecture Documentation

## System Overview

The Spur Gear AI Failure Monitor is a Streamlit-based web application that provides real-time predictive maintenance for industrial spur gears.

## Components

### 1. Frontend (Streamlit)
- **app.py**: Main application entry point
- **src/components/**: Reusable UI components
- **assets/styles/**: CSS styling

### 2. ML Pipeline
- **src/models/predictor.py**: Prediction logic
- **models/**: Trained SVM model and scaler
- **SHAP & LIME**: Explainability engines

### 3. Data Layer
- **src/utils/database.py**: SQLite operations
- **data/**: Raw and processed datasets
- **gear_history.db**: Historical logs

### 4. Utilities
- **src/utils/pdf_report.py**: PDF generation
- **src/utils/styling.py**: Chart styling
- **config/**: Configuration management

## Data Flow

```
User Input (Sliders)
    ↓
Parameter Validation
    ↓
Feature Scaling
    ↓
SVM Model Prediction
    ↓
├─→ Failure Probability
├─→ RUL Calculation
├─→ SHAP Explanation
├─→ LIME Explanation
└─→ Database Logging
    ↓
UI Rendering (6 Tabs)
```

## Key Features

### Tab 1: Prediction & Risk
- Real-time failure probability gauge
- Risk classification
- RUL estimation with confidence bands
- PDF report generation

### Tab 2: Explainability
- SHAP feature importance (global)
- LIME local explanations
- Maintenance recommendations

### Tab 3: Trends & History
- Time-series analysis
- Breach counters
- Risk distribution
- CSV export

### Tab 4: What-If Optimizer
- Differential Evolution optimizer
- Parameter sensitivity analysis
- Before/after comparison
- Radar charts

### Tab 5: 3D Gear Model
- Three.js digital twin
- Real-time parameter coupling
- Condition-reactive effects
- Engineering metrics

### Tab 6: Maintenance Scheduler
- AI-generated task lists
- Gantt chart visualization
- Cost-benefit analysis
- Priority matrix

## Technology Stack

- **Frontend**: Streamlit, HTML/CSS/JavaScript
- **ML**: scikit-learn (SVM), SHAP, LIME
- **Visualization**: Plotly, Matplotlib, Three.js
- **Database**: SQLite
- **PDF**: ReportLab
- **Optimization**: SciPy (Differential Evolution)

## Security Considerations

- Environment variables for API keys (.env)
- Input validation on all parameters
- SQL injection prevention (parameterized queries)
- No sensitive data in logs

## Performance Optimization

- @st.cache_data for expensive computations
- Session state for persistent data
- Lazy loading of ML models
- Efficient database indexing

## Deployment

Recommended platforms:
- Streamlit Cloud
- Heroku
- AWS EC2
- Docker container
