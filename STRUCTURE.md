# 📁 Project Structure

```
spur_gear_monitor/
│
├── 📄 app.py                      # Main Streamlit application
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
├── 📄 .env                        # Environment variables (not in git)
├── 📄 .env.example                # Environment template
├── 📄 .gitignore                  # Git ignore rules
├── 📄 pytest.ini                  # Pytest configuration
│
├── 📁 src/                        # Source code
│   ├── 📄 __init__.py
│   │
│   ├── 📁 components/             # UI components
│   │   └── 📄 __init__.py
│   │
│   ├── 📁 models/                 # ML model utilities
│   │   ├── 📄 __init__.py
│   │   └── 📄 predictor.py        # Prediction logic
│   │
│   └── 📁 utils/                  # Helper functions
│       ├── 📄 __init__.py
│       ├── 📄 database.py         # Database operations
│       ├── 📄 styling.py          # Chart styling
│       └── 📄 pdf_report.py       # PDF generation
│
├── 📁 data/                       # Data storage
│   ├── 📁 raw/                    # Raw datasets
│   ├── 📁 processed/              # Processed datasets
│   │   └── 📄 spur_gear_svm_dataset.csv
│   └── 📄 gear_history.db         # SQLite database
│
├── 📁 models/                     # Trained ML models
│   ├── 📄 spur_gear_svm_model.pkl
│   └── 📄 spur_gear_scaler.pkl
│
├── 📁 assets/                     # Static assets
│   ├── 📁 images/                 # Images and icons
│   └── 📁 styles/                 # CSS stylesheets
│
├── 📁 config/                     # Configuration
│   ├── 📄 config.yaml             # Main config file
│   └── 📄 settings.py             # Settings loader
│
├── 📁 logs/                       # Application logs
│   └── 📄 app.log
│
├── 📁 docs/                       # Documentation
│   ├── 📄 ARCHITECTURE.md         # System architecture
│   └── 📄 API.md                  # API documentation
│
├── 📁 tests/                      # Unit tests
│   ├── 📄 test_predictor.py
│   ├── 📄 test_database.py
│   └── 📄 test.py                 # Original test file
│
└── 📁 notebooks/                  # Jupyter notebooks
    └── 📄 prototype.ipynb         # Development prototype

```

## 📋 File Descriptions

### Root Level
- **app.py**: Main entry point for the Streamlit application
- **requirements.txt**: All Python package dependencies
- **README.md**: Project overview and setup instructions
- **.env**: Environment variables (API keys, secrets)
- **.gitignore**: Files to exclude from version control

### src/ - Source Code
Organized, reusable Python modules:
- **components/**: Reusable UI components
- **models/**: ML model loading and prediction
- **utils/**: Helper functions (database, PDF, styling)

### data/ - Data Storage
- **raw/**: Original, immutable datasets
- **processed/**: Cleaned, transformed datasets
- **gear_history.db**: SQLite database for historical logs

### models/ - ML Models
Trained machine learning models:
- **spur_gear_svm_model.pkl**: Trained SVM classifier
- **spur_gear_scaler.pkl**: Feature scaler (StandardScaler)

### config/ - Configuration
- **config.yaml**: Centralized configuration (parameters, thresholds)
- **settings.py**: Python module to load config

### tests/ - Testing
Unit tests for all modules using pytest

### docs/ - Documentation
Technical documentation and architecture diagrams

### notebooks/ - Jupyter Notebooks
Exploratory data analysis and prototyping

## 🎯 Benefits of This Structure

1. **Separation of Concerns**: Each module has a single responsibility
2. **Reusability**: Components can be imported and reused
3. **Testability**: Easy to write unit tests for each module
4. **Scalability**: Easy to add new features without cluttering
5. **Maintainability**: Clear organization makes code easy to find
6. **Professional**: Follows Python best practices
7. **Version Control**: Clean .gitignore keeps repo tidy
8. **Documentation**: Comprehensive docs for onboarding

## 🚀 Next Steps

1. Move existing code from `app.py` into modular components
2. Update imports in `app.py` to use new modules
3. Write unit tests for critical functions
4. Add CI/CD pipeline (GitHub Actions)
5. Create Docker container for deployment
