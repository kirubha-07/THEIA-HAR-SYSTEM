"""
paths.py — centralized filesystem anchors for THEIA HAR System runtime.

Provides absolute Path objects anchored to `har_system/` regardless of
current working directory, avoiding duplicate logs/recordings when running
from repo root vs har_system/.
"""

from pathlib import Path

# Anchored to the har_system directory
BASE_DIR = Path(__file__).resolve().parent

LOGS_DIR = BASE_DIR / "logs"
RECORDINGS_DIR = BASE_DIR / "recordings"
PROFILES_PATH = BASE_DIR / "profiles.json"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
CONFIGS_DIR = CONFIG_DIR  # Convenience alias
BENCHMARKS_DIR = BASE_DIR / "benchmarks"

# Ensure all runtime output directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
