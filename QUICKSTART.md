# 🚀 Quick Start Guide

## Installation (5 minutes)

### 1. Clone & Navigate
```bash
cd spur_gear_monitor
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit .env and add your API key
# GROQ_API_KEY=your_key_here
```

### 5. Run Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## First Use

### Sidebar Controls
1. **Gear Type**: Select gear model (A, B, or C)
2. **Operational Parameters**: Adjust 6 sliders
   - Speed (RPM)
   - Torque (Nm)
   - Vibration (mm/s)
   - Temperature (°C)
   - Shock Load (g)
   - Noise (dB)
3. **RUL Configuration**: Set max expected cycles

### Tabs Overview

**📊 Prediction & Risk**
- View current failure probability
- Check risk classification
- See RUL estimates
- Download PDF report

**🧠 Explainability**
- SHAP feature importance
- LIME local explanations
- Maintenance recommendations

**📈 Trends & History**
- Historical data (auto-logged)
- Time-series charts
- Breach counters
- Export CSV

**🔧 What-If Optimizer**
- Lock/unlock parameters
- Find safe operating point
- View sensitivity analysis

**⚙️ 3D Gear Model**
- Interactive digital twin
- Real-time effects
- Engineering metrics

**🗓 Maintenance Scheduler**
- Generate AI maintenance plan
- Gantt chart timeline
- Cost analysis

## Common Tasks

### View Current Status
1. Adjust sliders to current operating conditions
2. Check Tab 1 for failure probability and RUL

### Investigate High Risk
1. Go to Tab 2 (Explainability)
2. Check SHAP chart for risk drivers
3. Read maintenance recommendations

### Optimize Parameters
1. Go to Tab 4 (What-If Optimizer)
2. Lock fixed parameters
3. Set target probability
4. Click "Find Safe Operating Point"

### Generate Maintenance Plan
1. Go to Tab 6 (Maintenance Scheduler)
2. Configure planning horizon
3. Set crew availability
4. Click "Generate Maintenance Schedule"

### Export Data
- **PDF Report**: Tab 1 → Download button
- **CSV History**: Tab 3 → Export button

## Troubleshooting

### App won't start
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model files missing
Ensure these files exist:
- `models/spur_gear_svm_model.pkl`
- `models/spur_gear_scaler.pkl`
- `data/processed/spur_gear_svm_dataset.csv`

### Database errors
```bash
# Reset database
rm data/gear_history.db
# Restart app - database will be recreated
```

### SHAP/LIME slow
- Normal on first run (computes background)
- Cached after first computation
- Reduce background samples in code if needed

## Next Steps

1. ✅ Explore all 6 tabs
2. ✅ Adjust parameters and observe changes
3. ✅ Generate and download a PDF report
4. ✅ Run the optimizer to find safe settings
5. ✅ Create a maintenance schedule
6. ✅ Review historical trends

## Support

- 📖 Full docs: `README.md`
- 🏗️ Architecture: `docs/ARCHITECTURE.md`
- 📁 Structure: `STRUCTURE.md`
- 🔧 API: `docs/API.md`

## Development

### Run Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/
flake8 src/
```

### Add New Feature
1. Create module in `src/`
2. Write tests in `tests/`
3. Update `app.py` to use module
4. Document in `docs/`

Enjoy! 🎉
