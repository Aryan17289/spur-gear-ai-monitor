"""Configuration settings loader"""
import os
import yaml
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Load configuration
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

def load_config():
    """Load configuration from YAML file"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

# Load config on import
try:
    CONFIG = load_config()
except FileNotFoundError:
    print(f"Warning: Config file not found at {CONFIG_PATH}")
    CONFIG = {}

# Paths
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
ASSETS_DIR = ROOT_DIR / "assets"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
