"""Canonical locations for AURORA inputs and final output artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
MAPS_DIR = PROJECT_ROOT / "maps"
REGION_MAPS_DIR = MAPS_DIR / "regions"
MAP_CACHE_DIR = MAPS_DIR / "cache"
VIDEOS_DIR = PROJECT_ROOT / "videos"


def asset_path(name):
    """Return an absolute path inside the shared AURORA assets directory."""
    return ASSETS_DIR / name


def video_path(name):
    """Return an absolute path inside the shared AURORA videos directory."""
    return VIDEOS_DIR / name


def map_path(name):
    """Return an absolute path for a final full-sky map."""
    return MAPS_DIR / name


def region_map_path(name):
    """Return an absolute path for a regional map or its layout."""
    return REGION_MAPS_DIR / name


def map_cache_path(*parts):
    """Return an absolute path inside the shared map/cache hierarchy."""
    return MAP_CACHE_DIR.joinpath(*parts)


def ensure_output_directories():
    """Create the canonical final-output directories when needed."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    REGION_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
