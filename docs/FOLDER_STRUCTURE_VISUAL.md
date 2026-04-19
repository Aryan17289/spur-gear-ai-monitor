# 📁 Visual Folder Structure

```
spur_gear_monitor/
│
├── 📄 app.py                          ← Main Streamlit app (entry point)
├── 📄 requirements.txt                ← Python dependencies
├── 📄 README.md                       ← Project overview
├── 📄 QUICKSTART.md                   ← Quick start guide
├── 📄 STRUCTURE.md                    ← Structure explanation
├── 📄 PROJECT_SUMMARY.md              ← What was done
├── 📄 .env                            ← Environment variables (secret)
├── 📄 .env.example                    ← Environment template
├── 📄 .gitignore                      ← Git ignore rules
├── 📄 pytest.ini                      ← Pytest config
│
├── 📁 src/                            ← SOURCE CODE
│   ├── 📄 __init__.py
│   │
│   ├── 📁 components/                 ← UI Components
│   │   └── 📄 __init__.py
│   │
│   ├── 📁 models/                     ← ML Models
│   │   ├── 📄 __init__.py
│   │   └── 📄 predictor.py            ← GearPredictor class
│   │
│   └── 📁 utils/                      ← Utilities
│       ├── 📄 __init__.py
│       ├── 📄 database.py             ← GearHistoryDB class
│       ├── 📄 pdf_report.py           ← PDF generation
│       └── 📄 styling.py              ← Chart styling
│
├── 📁 data/                           ← DATA STORAGE
│   ├── 📁 raw/                        ← Original datasets
│   ├── 📁 processed/                  ← Cleaned datasets
│   │   └── 📄 spur_gear_svm_dataset.csv
│   └── 📄 gear_history.db             ← SQLite database
│
├── 📁 models/                         ← TRAINED MODELS
│   ├── 📄 spur_gear_svm_model.pkl     ← SVM classifier
│   └── 📄 spur_gear_scaler.pkl        ← Feature scaler
│
├── 📁 assets/                         ← STATIC ASSETS
│   ├── 📁 images/                     ← Images, icons
│   └── 📁 styles/                     ← CSS files
│
├── 📁 config/                         ← CONFIGURATION
│   ├── 📄 config.yaml                 ← Main config
│   └── 📄 settings.py                 ← Config loader
│
├── 📁 logs/                           ← LOGS
│   └── 📄 app.log                     ← Application logs
│
├── 📁 docs/                           ← DOCUMENTATION
│   ├── 📄 ARCHITECTURE.md             ← System architecture
│   ├── 📄 API.md                      ← API docs
│   └── 📄 FOLDER_STRUCTURE_VISUAL.md  ← This file
│
├── 📁 tests/                          ← UNIT TESTS
│   ├── 📄 test_predictor.py           ← Predictor tests
│   ├── 📄 test_database.py            ← Database tests
│   └── 📄 test.py                     ← Original tests
│
└── 📁 notebooks/                      ← JUPYTER NOTEBOOKS
    └── 📄 prototype.ipynb             ← Development prototype

```

## 🎯 Purpose of Each Folder

### 📁 src/ - Source Code
**Purpose**: All reusable Python code
- **components/**: Streamlit UI components
- **models/**: ML prediction logic
- **utils/**: Helper functions (database, PDF, styling)

**Why**: Keeps code organized, modular, and testable

---

### 📁 data/ - Data Storage
**Purpose**: All data files
- **raw/**: Original, immutable datasets
- **processed/**: Cleaned, transformed data
- **gear_history.db**: SQLite database for logs

**Why**: Separates raw from processed data, easy backup

---

### 📁 models/ - Trained Models
**Purpose**: Serialized ML models
- **spur_gear_svm_model.pkl**: Trained classifier
- **spur_gear_scaler.pkl**: Feature scaler

**Why**: Version control for models, easy deployment

---

### 📁 assets/ - Static Assets
**Purpose**: Images, CSS, static files
- **images/**: Icons, logos, diagrams
- **styles/**: Custom CSS stylesheets

**Why**: Separates code from assets, easy CDN deployment

---

### 📁 config/ - Configuration
**Purpose**: Application settings
- **config.yaml**: Centralized configuration
- **settings.py**: Python config loader

**Why**: Change settings without touching code

---

### 📁 logs/ - Logs
**Purpose**: Application logs
- **app.log**: Runtime logs

**Why**: Debugging, monitoring, audit trail

---

### 📁 docs/ - Documentation
**Purpose**: Technical documentation
- **ARCHITECTURE.md**: System design
- **API.md**: API reference

**Why**: Onboarding, maintenance, collaboration

---

### 📁 tests/ - Unit Tests
**Purpose**: Automated testing
- **test_*.py**: Unit tests for each module

**Why**: Ensure code quality, prevent regressions

---

### 📁 notebooks/ - Jupyter Notebooks
**Purpose**: Exploratory analysis
- **prototype.ipynb**: Development experiments

**Why**: Data exploration, prototyping, documentation

---

## 🔄 Data Flow

```
User Input (app.py)
    ↓
src/models/predictor.py (Prediction)
    ↓
src/utils/database.py (Logging)
    ↓
data/gear_history.db (Storage)
    ↓
app.py (Visualization)
```

## 📦 Import Structure

```python
# app.py imports from src/
from src.models.predictor import GearPredictor
from src.utils.database import GearHistoryDB
from src.utils.pdf_report import build_pdf_report
from config.settings import CONFIG

# Tests import from src/
from src.models.predictor import GearPredictor
```

## 🎨 Color Legend

- 📄 = File
- 📁 = Folder
- ← = Description
- 🎯 = Purpose
- 🔄 = Flow
- 📦 = Import

## ✅ Best Practices Followed

1. ✅ **Separation of Concerns**: Each folder has one purpose
2. ✅ **DRY Principle**: Reusable modules in src/
3. ✅ **Configuration Management**: Centralized in config/
4. ✅ **Testing**: Dedicated tests/ folder
5. ✅ **Documentation**: Comprehensive docs/
6. ✅ **Version Control**: .gitignore for secrets
7. ✅ **Data Separation**: raw/ vs processed/
8. ✅ **Modularity**: Easy to add/remove features

This structure scales from prototype to production! 🚀
