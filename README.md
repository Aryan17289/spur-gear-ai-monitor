# ⚙️ Spur Gear AI Failure Monitor

> **Intelligent Predictive Maintenance System** for industrial spur gears — powered by SVM, SHAP/LIME explainability, a 3D digital twin, and an AI copilot.  
> Built as part of the **GearMind AI** internship project at **Elecon Engineering Company Ltd.**

---

## 📌 Overview

The **Spur Gear AI Failure Monitor** is a real-time, data-driven web application that predicts spur gear failures before they happen. It combines classical machine learning with explainable AI, live sensor simulation, and an interactive dashboard — enabling maintenance engineers to make proactive, evidence-backed decisions.

The app runs entirely in the browser via Streamlit and requires no backend deployment.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Failure Prediction** | SVM model predicts failure probability from live sensor inputs |
| 🧠 **Explainable AI** | SHAP & LIME visualizations explain every prediction |
| ⏱️ **Remaining Useful Life** | Estimates RUL in operational cycles based on degradation rate |
| 📊 **Trends & History** | SQLite-backed historical log with interactive trend charts |
| 🔬 **What-If Optimizer** | Scipy-powered optimizer finds the safest operating parameters |
| 🌐 **3D Digital Twin** | Animated Three.js gear model reacts to live sensor values |
| 🗓️ **Maintenance Scheduler** | AI-generated maintenance plans with Gantt chart visualization |
| 📄 **PDF Reports** | One-click downloadable reports via ReportLab |
| 🤖 **AI Copilot** | Groq/LLaMA-powered chatbot with live sensor context injection |
| 🎨 **Industrial UI** | Dark SCADA-style theme with animated particle background |

---

## 🖥️ Dashboard Tabs

```
Tab 1 — Prediction & Risk     : Live failure gauge, risk level, RUL, health score
Tab 2 — Explainability        : SHAP summary + waterfall, LIME feature importance
Tab 3 — Trends & History      : Historical sensor trends, session log, anomaly markers
Tab 4 — What-If Optimizer     : Differential evolution optimization for safe parameters
Tab 5 — 3D Gear Model         : Interactive Three.js digital twin driven by sensors
Tab 6 — Maintenance Scheduler : AI maintenance plan, task timeline, Gantt chart
```

---

## 🗂️ Project Structure

```
spur_gear/
├── app.py                          # Main Streamlit application (~4700 lines)
│
├── src/                            # Modular source code
│   ├── models/
│   │   └── predictor.py            # GearPredictor class (load, predict, score)
│   ├── utils/
│   │   ├── database.py             # GearHistoryDB — SQLite read/write
│   │   ├── pdf_report.py           # ReportLab PDF generator
│   │   └── styling.py              # Plotly chart theme helpers
│   └── components/
│       └── __init__.py
│
├── models/                         # Trained ML artifacts
│   ├── spur_gear_svm_model.pkl     # Trained SVM classifier
│   └── spur_gear_scaler.pkl        # StandardScaler for feature normalization
│
├── data/
│   ├── processed/
│   │   └── spur_gear_svm_dataset.csv   # Training/reference dataset (~6MB)
│   └── gear_history.db             # SQLite session history (auto-created)
│
├── config/
│   ├── config.yaml                 # Centralized parameter ranges & thresholds
│   └── settings.py                 # YAML config loader
│
├── docs/
│   ├── ARCHITECTURE.md             # System design overview
│   ├── API.md                      # Internal API reference
│   └── FOLDER_STRUCTURE_VISUAL.md  # Visual folder map
│
├── tests/
│   ├── test_predictor.py           # Unit tests for GearPredictor
│   ├── test_database.py            # Unit tests for GearHistoryDB
│   └── test.py                     # Integration tests
│
├── notebooks/
│   └── prototype.ipynb             # Model training & exploration notebook
│
├── assets/
│   ├── images/                     # Icons, screenshots
│   └── styles/                     # Custom CSS
│
├── logs/                           # Runtime application logs
├── .env.example                    # Environment variable template
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spur-gear-ai-monitor.git
cd spur-gear-ai-monitor
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Then open `.env` and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> Get a free key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🔧 Configuration

All tunable parameters live in `config/config.yaml`:

```yaml
parameters:
  speed:      { min: 500,  max: 3000, default: 1050, unit: "RPM" }
  torque:     { min: 50,   max: 400,  default: 110,  unit: "Nm"  }
  vibration:  { min: 0.5,  max: 10.0, default: 1.4,  unit: "mm/s"}
  temperature:{ min: 30,   max: 120,  default: 57,   unit: "°C"  }
  shock:      { min: 0.1,  max: 6.0,  default: 1.7,  unit: "g"   }
  noise:      { min: 50,   max: 100,  default: 74,   unit: "dB"  }

thresholds:
  danger:
    speed: 2375    # RPM above which speed is flagged
    torque: 287    # Nm
    vibration: 7.6 # mm/s
    temperature: 97 # °C
    shock: 4.4     # g
    noise: 87      # dB
```

---

## 🧠 ML Model Details

| Property | Value |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | RBF |
| Input Features | Speed, Torque, Vibration, Temperature, Shock, Noise |
| Output | Failure Probability (0–1), Binary Classification |
| Explainability | SHAP KernelExplainer + LIME TabularExplainer |
| Scaler | StandardScaler (fitted on training set) |
| Training Data | `data/processed/spur_gear_svm_dataset.csv` |

---

## 📦 Key Dependencies

```
streamlit          — Dashboard framework
scikit-learn       — SVM model & preprocessing
shap               — Model explainability (global + local)
lime               — Local surrogate explanations
plotly             — Interactive charts
scipy              — What-If parameter optimization
reportlab          — PDF report generation
openai             — AI Copilot (Groq-compatible API)
sqlite3            — Built-in Python database
python-dotenv      — Environment variable management
pyyaml             — YAML config loading
```

Full list → [`requirements.txt`](requirements.txt)

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- `test_predictor.py` — model load, predict, probability output
- `test_database.py` — insert, fetch, schema validation
- `test.py` — integration and edge-case tests

---

## 📸 Screenshots

> _Add screenshots of your dashboard here_

```
[ Prediction & Risk Tab ]    [ Explainability Tab ]
[ 3D Gear Digital Twin  ]    [ Maintenance Scheduler ]
```

---

## 🏭 About the Project

This system was developed as an **internship project at [Elecon Engineering Company Ltd.](https://www.elecon.com/)** — a global leader in industrial gear manufacturing.

It is the **spur gear module** of the broader **GearMind AI** platform, which covers:
- 🔵 Spur Gear — SVM classifier *(this repo)*
- 🟢 Helical Gear — Gradient Boosting Regressor
- 🟠 Worm Gear — Logistic Regression
- 🔴 Bevel Gear — XGBoost

---

## 📄 License

This project is for academic and internship demonstration purposes.  
© 2026 — Elecon Engineering Internship Project

---

## 👤 Author

**Aryan** — Intern, Elecon Engineering Company Ltd.  
GearMind AI — Predictive Maintenance System

---

## 🤝 Contributing

This is an internship project. For suggestions or feedback, feel free to open an issue or submit a pull request.