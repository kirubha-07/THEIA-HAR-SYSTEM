"""
paths.py — centralized filesystem anchors for THEIA HAR System runtime.

Provides absolute Path objects anchored to `har_system/` regardless of
current working directory, avoiding duplicate logs/recordings when running
from repo root vs har_system/.
"""

from pathlib import Path

# Anchored to the har_system directory
BASE_DIR = Path(__file__).resolve().parent

import os

env_video_dir = os.environ.get("THEIA_VIDEO_DIR")
if env_video_dir:
    VIDEOS_DIR = Path(env_video_dir).resolve()
else:
    root_videos = BASE_DIR.parent / "videos"
    har_videos = BASE_DIR / "videos"
    root_exists = root_videos.is_dir()
    har_exists = har_videos.is_dir()
    if root_exists and har_exists:
        raise RuntimeError("STOP: Both <repo root>/videos and har_system/videos exist. Please resolve ambiguity.")
    elif root_exists:
        VIDEOS_DIR = root_videos
    elif har_exists:
        VIDEOS_DIR = har_videos
    else:
        VIDEOS_DIR = root_videos

LOGS_DIR = BASE_DIR / "logs"
RECORDINGS_DIR = BASE_DIR / "recordings"
env_profiles_path = os.environ.get("THEIA_PROFILES_PATH")
if env_profiles_path:
    PROFILES_PATH = Path(env_profiles_path).resolve()
else:
    PROFILES_PATH = BASE_DIR / "profiles.json"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
CONFIGS_DIR = CONFIG_DIR  # Convenience alias
BENCHMARKS_DIR = BASE_DIR / "benchmarks"

MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiment_config.yaml"
EXPERIMENT_CONFIG_STOCK_PATH = CONFIG_DIR / "experiment_config_stock_coco.yaml"

# Ensure all runtime output directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
