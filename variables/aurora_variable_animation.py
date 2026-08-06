"""Render the AURORA Gaia variable-star animation at a selected resolution."""

from dataclasses import dataclass, replace
import hashlib
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
import subprocess
import sys
import time
import warnings
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from astropy.table import Table
from astropy.units import UnitsWarning
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_render_core import (
    VARIABLE_FLUX_MODES,
    VARIABLE_MODEL_CODES,
    array_signature,
    draw_gaussians_u8,
    galactic_to_hammer_pixel,
    galactic_to_region_pixel,
    galactic_region_mask,
    temperature_from_columns,
    variable_frame_parameters,
    variable_star_colors_from_columns,
)
from core.aurora_region_selection import select_sky_region
from core.aurora_paths import (
    VIDEOS_DIR,
    asset_path,
    map_path,
    region_map_path,
    video_path,
)
from core.aurora_resolution import (
    prompt_resolution,
    prompt_sky_map_background,
    prompt_sky_map_mode,
    sky_map_background_suffix,
)
from core.aurora_video_core import (
    VIDEO_CONFIG as BASE_VIDEO_CONFIG,
    build_editorial_timeline,
    build_hevc_command,
    build_hvc1_remux_command,
    build_mobile_hevc_command,
    cache_artifact_paths,
)

MEMORY = MemoryController.from_environment()


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

VARIABLE_MODE_STANDARD = "standard"
VARIABLE_MODE_LBV_TEMPERATURE_EXTREMES = "lbv_temperature_extremes"
VARIABLE_MODE_RR_LYRAE = "rr_lyrae"
VARIABLE_MODE_CEPHEIDS = "cepheids"
VARIABLE_MODE_ZZ_CETI = "zz_ceti"
VARIABLE_MODE_CATACLYSMIC = "cataclysmic"
VARIABLE_MODE_OTHER = "other"
VARIABLE_MODE_LBV_ALL = "lbv_all"
VARIABLE_MODE_BE = "be"
VARIABLE_MODE_GCAS = "gcas"
VARIABLE_MODE_SDOR = "sdor"
VARIABLE_MODE_WR = "wr"
VARIABLE_MODE_UNKNOWN_HOT = "unknown_hot_variables"
VARIABLE_MODE_LBV_SELECTED = "lbv_selected"
HOT_VARIABLE_MODES = frozenset(
    {
        VARIABLE_MODE_LBV_ALL,
        VARIABLE_MODE_BE,
        VARIABLE_MODE_GCAS,
        VARIABLE_MODE_SDOR,
        VARIABLE_MODE_WR,
        VARIABLE_MODE_UNKNOWN_HOT,
        VARIABLE_MODE_LBV_SELECTED,
    }
)
HOT_VARIABLE_CACHE_CODES = {
    VARIABLE_MODE_LBV_ALL: 9,
    VARIABLE_MODE_BE: 3,
    VARIABLE_MODE_GCAS: 4,
    VARIABLE_MODE_SDOR: 5,
    VARIABLE_MODE_WR: 6,
    VARIABLE_MODE_UNKNOWN_HOT: 7,
    VARIABLE_MODE_LBV_SELECTED: 8,
}
VARIABLE_MODE_ALIASES = {
    "1": VARIABLE_MODE_STANDARD,
    "standard": VARIABLE_MODE_STANDARD,
    "all": VARIABLE_MODE_STANDARD,
    "default": VARIABLE_MODE_STANDARD,
    "2": VARIABLE_MODE_LBV_TEMPERATURE_EXTREMES,
    "lbv": VARIABLE_MODE_LBV_ALL,
    "lbv_all": VARIABLE_MODE_LBV_ALL,
    "all_lbv": VARIABLE_MODE_LBV_ALL,
    "extremes": VARIABLE_MODE_LBV_TEMPERATURE_EXTREMES,
    "lbv_temperature_extremes": VARIABLE_MODE_LBV_TEMPERATURE_EXTREMES,
    "rr": VARIABLE_MODE_RR_LYRAE,
    "rr_lyrae": VARIABLE_MODE_RR_LYRAE,
    "cepheid": VARIABLE_MODE_CEPHEIDS,
    "cepheids": VARIABLE_MODE_CEPHEIDS,
    "zz_ceti": VARIABLE_MODE_ZZ_CETI,
    "cataclysmic": VARIABLE_MODE_CATACLYSMIC,
    "cv": VARIABLE_MODE_CATACLYSMIC,
    "other": VARIABLE_MODE_OTHER,
    "3": VARIABLE_MODE_BE,
    "be": VARIABLE_MODE_BE,
    "4": VARIABLE_MODE_GCAS,
    "gcas": VARIABLE_MODE_GCAS,
    "5": VARIABLE_MODE_SDOR,
    "sdor": VARIABLE_MODE_SDOR,
    "lbv_sdor": VARIABLE_MODE_SDOR,
    "6": VARIABLE_MODE_WR,
    "wr": VARIABLE_MODE_WR,
    "wolf_rayet": VARIABLE_MODE_WR,
    "7": VARIABLE_MODE_UNKNOWN_HOT,
    "unknown": VARIABLE_MODE_UNKNOWN_HOT,
    "unclassified": VARIABLE_MODE_UNKNOWN_HOT,
    "unknown_hot_variables": VARIABLE_MODE_UNKNOWN_HOT,
    "lbv_selected": VARIABLE_MODE_LBV_SELECTED,
    "sdor_selected": VARIABLE_MODE_LBV_SELECTED,
}
DEFAULT_ANIMATION_DAYS = 500.0
# Visual-size reference measured from
# aurora_variable_16k_lbv_temperature_extremes_2500d_hvc1.mp4.
# At 16K this gives a base Gaussian sigma of
# 3.20 px * 2.20 * 1.65 = 11.616 px for single-catalogue renders.
LBV_REFERENCE_BASE_SIZE_SCALE = np.float32(2.20)
LBV_REFERENCE_SIZE_MULTIPLIER = np.float32(1.65)
VARIABLE_MAP_SIZE_SCALE = np.float32(
    LBV_REFERENCE_BASE_SIZE_SCALE * LBV_REFERENCE_SIZE_MULTIPLIER
)
# Dedicated 16K Gaussian limits for the dense all-catalogues map:
# base sigma 3.20 px * scale 1.0, bounded to 2.20–12.50 px.
ALL_CATALOGUES_MAP_SIZE_SCALE = np.float32(1.0)
ALL_CATALOGUES_MIN_SIGMA_16K = 2.20
ALL_CATALOGUES_MAX_SIGMA_16K = 12.50
LBV_GROUP_KEYS = ("BE", "GCAS", "SDOR", "WR", "UNKNOWN_HOT")

# Rendering and timeline controls.
MAX_RENDER_WORKERS_LIMIT = 6
FRAME_CACHE_VERSION = 34
VARIABLE_BASE_SIGMA_16K = 3.20
SINGLE_CATALOGUE_MIN_SIGMA_16K = 2.20
SINGLE_CATALOGUE_MAX_SIGMA_16K = 20.0
INTRO_FRACTION = 0.0
OUTRO_FRACTION = 0.0
APPEARANCE_FRACTION = 0.20
FADE_IN_TIME = 0.04
FADE_OUT_TIME = 0.04
VIVID_TEMPERATURE_COLOURS = True

# Reproducible catalogue sampling and per-object animation variation.
CATALOG_SAMPLE_SEED = 42
APPEARANCE_SEED = 42
CLASS_VARIATION_SEED = 31_415
EVENT_TIME_SEED = 27_182
IRREGULARITY_SEED = 16_180
PULSATION_PHASE_SEED = 57_721
IRREGULARITY_SCALE_MIN = np.float32(0.75)
IRREGULARITY_SCALE_RANGE = np.float32(0.50)
EVENT_CENTER_MIN = np.float32(0.24)
EVENT_CENTER_RANGE = np.float32(0.52)

# Catalogue paths and deterministic per-class limits.
STANDARD_VARIABLE_FILES = {
    "RR_LYRAE": asset_path("rr_lyrae.fits"),
    "CEPHEIDS": asset_path("cepheids.fits"),
    "ZZ_CETI": asset_path("zz_ceti.fits"),
    "CATACLYSMIC": asset_path("cataclysmic_variables.fits"),
    "OTHER": asset_path("other_variables.fits"),
}
LBV_PARENT_FILE = asset_path("lbv.fits")
SINGLE_STANDARD_VARIABLE_FILES = {
    VARIABLE_MODE_RR_LYRAE: {
        "RR_LYRAE": STANDARD_VARIABLE_FILES["RR_LYRAE"]
    },
    VARIABLE_MODE_CEPHEIDS: {
        "CEPHEIDS": STANDARD_VARIABLE_FILES["CEPHEIDS"]
    },
    VARIABLE_MODE_ZZ_CETI: {
        "ZZ_CETI": STANDARD_VARIABLE_FILES["ZZ_CETI"]
    },
    VARIABLE_MODE_CATACLYSMIC: {
        "CATACLYSMIC": STANDARD_VARIABLE_FILES["CATACLYSMIC"]
    },
    VARIABLE_MODE_OTHER: {"OTHER": STANDARD_VARIABLE_FILES["OTHER"]},
}
LBV_TEMPERATURE_EXTREMES_FILE = asset_path("lbv_temperature_extremes.fits")
LBV_GROUP_FILES = {
    "BE": asset_path("be.fits"),
    "GCAS": asset_path("gcas.fits"),
    "SDOR": asset_path("sdor.fits"),
    "WR": asset_path("wr.fits"),
    "UNKNOWN_HOT": asset_path("be_gcas_sdor_wr_unknown.fits"),
}
ALL_CATALOGUE_VARIABLE_FILES = {
    **STANDARD_VARIABLE_FILES,
    **(
        LBV_GROUP_FILES
        if all(path.exists() for path in LBV_GROUP_FILES.values())
        else {"LBV": LBV_PARENT_FILE}
    ),
}
SINGLE_HOT_VARIABLE_GROUPS = {
    VARIABLE_MODE_BE: "BE",
    VARIABLE_MODE_GCAS: "GCAS",
    VARIABLE_MODE_SDOR: "SDOR",
    VARIABLE_MODE_LBV_SELECTED: "SDOR",
    VARIABLE_MODE_WR: "WR",
    VARIABLE_MODE_UNKNOWN_HOT: "UNKNOWN_HOT",
}
VARIABLE_LIMITS = {
    "CEPHEIDS": 3_000,
    "RR_LYRAE": 4_500,
    "LBV": 12_000,
    "BE": 12_000,
    "GCAS": 12_000,
    "SDOR": 12_000,
    "WR": 12_000,
    "UNKNOWN_HOT": 12_000,
    "ZZ_CETI": 4_500,
    "CATACLYSMIC": 4_500,
    # Applied separately to every classifier_class in other_variables.fits.
    "OTHER": 1_500,
}

# LMC/SMC sampling configuration. Gaia values remain noisy estimates.
MAGELLANIC_PARALLAX_MAX_MAS = 0.020
MAGELLANIC_TOTAL_LIMIT = 80
MAGELLANIC_SKY_REGIONS = (
    # name, Galactic longitude [deg], Galactic latitude [deg], radius [deg]
    ("LMC", 280.4652, -32.8884, 12.0),
    ("SMC", 302.8084, -44.3277, 8.0),
)
SKY_BALANCE_CENTRE_MAX_LATITUDE_DEG = 12.0
SKY_BALANCE_CENTRE_MAX_LONGITUDE_DEG = 25.0
SKY_BALANCE_DISK_MAX_LATITUDE_DEG = 15.0
SKY_BALANCE_MID_MAX_LATITUDE_DEG = 45.0
SKY_BALANCE_BASE_QUOTAS = np.array([650, 450, 250, 150], dtype=np.int32)
PARALLAX_COLUMN_NAMES = ("parallax", "parallax_mas", "plx")
VARIABLE_CLASS_COLUMN_NAMES = ("classifier_class", "variable_class")
CLASSIFIER_TYPE_ALIASES = {
    "RR": "RR_LYRAE",
    "CEP": "CEPHEIDS",
    "WD": "ZZ_CETI",
    "CV": "CATACLYSMIC",
}

# Colour precedence: teff -> reliable intrinsic BP-RP -> raw BP-RP -> class.
MIN_PLAUSIBLE_TEFF_K = 1_000.0
MAX_PLAUSIBLE_TEFF_K = 200_000.0
RELIABLE_REDDENING_QUALITIES = frozenset({"good"})
INVALID_INTRINSIC_COLOUR_FLAGS = frozenset(
    {"invalid_reddening", "uncertain_reddening"}
)
STRONG_REDDENING_MIN = 0.50
STRONG_EXTINCTION_AG_MIN = 1.00
HOT_RAW_RED_BP_RP_MIN = 0.20
HOT_CLASS_FALLBACK_TEMPERATURES = {
    "BE": np.float32(18_000.0),
    "GCAS": np.float32(18_000.0),
    "SDOR": np.float32(20_000.0),
    "LBV": np.float32(20_000.0),
    "WR": np.float32(40_000.0),
    "UNKNOWN_HOT": np.float32(15_000.0),
}

# Bounded auxiliary scaling from Gaia estimates; these never select a class.
LUMINOSITY_REFERENCE_LOG10 = np.float32(3.0)
LUMINOSITY_SCALING_SLOPE = np.float32(0.08)
LUMINOSITY_FACTOR_RANGE = (np.float32(0.88), np.float32(1.22))
MASS_REFERENCE_SOLAR = np.float32(10.0)
MASS_FACTOR_BASE = np.float32(0.98)
MASS_FACTOR_SLOPE = np.float32(0.04)
MASS_FACTOR_RANGE = (np.float32(0.96), np.float32(1.10))
CONFIDENCE_FACTOR_BASE = np.float32(0.94)
CONFIDENCE_FACTOR_RANGE = np.float32(0.06)
RADIUS_REFERENCE_SOLAR = np.float32(5.0)
RADIUS_SCALING_EXPONENT = np.float32(0.10)
RADIUS_FACTOR_RANGE = (np.float32(0.85), np.float32(1.25))

@dataclass(frozen=True)
class VariableBehavior:
    """Class rules that interpret, but never replace, per-object data."""

    light_curve: str
    flux_mode: str
    default_period_days: float
    default_amplitude_mag: float
    amplitude_floor_mag: float = 0.0
    irregularity: float = 0.0
    temperature_variation: float = 0.0
    size_scale: float = 1.0
    size_response: float = 0.08
    intensity_scale: float = 1.0
    brightness_response: float = 1.0
    amplitude_ceiling_mag: float = 5.0
    event_width: tuple[float, float] = (0.12, 0.28)


def _behavior(
    light_curve,
    flux_mode="range",
    default_period_days=5.0,
    default_amplitude_mag=0.10,
    **kwargs,
):
    return VariableBehavior(
        light_curve=light_curve,
        flux_mode=flux_mode,
        default_period_days=default_period_days,
        default_amplitude_mag=default_amplitude_mag,
        **kwargs,
    )


# Defaults are used for invalid/missing catalogue values. Valid measurements
# remain object-specific, with explicit display-amplitude floors only where a
# variability class must remain legible in the animation.
VARIABLE_BEHAVIOR = {
    "RR_LYRAE": _behavior(
        "rr_lyrae",
        default_period_days=0.55, default_amplitude_mag=0.8,
        temperature_variation=0.025, size_response=0.35,
        intensity_scale=1.10, brightness_response=1.65,
        amplitude_ceiling_mag=2.5,
    ),
    "CEPHEIDS": _behavior(
        "cepheid",
        default_period_days=5.0, default_amplitude_mag=0.7,
        temperature_variation=0.045, size_response=0.28,
        intensity_scale=1.08, brightness_response=1.40,
        amplitude_ceiling_mag=2.5,
    ),
    "ZZ_CETI": _behavior(
        "zz_ceti", default_period_days=0.01, default_amplitude_mag=0.12,
        irregularity=0.08, size_response=0.025,
        brightness_response=1.20, amplitude_ceiling_mag=0.7,
    ),
    "LBV": _behavior(
        "lbv",
        default_period_days=300.0, default_amplitude_mag=0.8,
        amplitude_floor_mag=0.8,
        irregularity=0.35, temperature_variation=-0.38,
        size_scale=2.20, size_response=1.50, intensity_scale=1.45,
        brightness_response=5.0,
        amplitude_ceiling_mag=3.0,
    ),
    "BE": _behavior(
        "stochastic_hot", flux_mode="outburst",
        default_period_days=2.0, default_amplitude_mag=0.40,
        amplitude_floor_mag=0.40, irregularity=1.0,
        temperature_variation=0.09, size_scale=1.12,
        size_response=0.95, intensity_scale=1.16,
        brightness_response=3.40, amplitude_ceiling_mag=0.65,
    ),
    "GCAS": _behavior(
        "stochastic_hot", flux_mode="outburst",
        default_period_days=60.0, default_amplitude_mag=0.50,
        amplitude_floor_mag=0.50, irregularity=1.0,
        temperature_variation=0.11, size_scale=1.25,
        size_response=1.05, intensity_scale=1.22,
        brightness_response=3.50, amplitude_ceiling_mag=0.70,
    ),
    "SDOR": _behavior(
        "lbv", default_period_days=300.0,
        default_amplitude_mag=0.8, amplitude_floor_mag=0.8,
        irregularity=0.35,
        temperature_variation=-0.38, size_scale=2.20,
        size_response=1.50, intensity_scale=1.45,
        brightness_response=5.0, amplitude_ceiling_mag=3.0,
    ),
    "WR": _behavior(
        "stochastic_hot", flux_mode="outburst",
        default_period_days=20.0, default_amplitude_mag=0.40,
        amplitude_floor_mag=0.40, irregularity=1.0,
        temperature_variation=0.12, size_scale=1.35,
        size_response=0.95, intensity_scale=1.55,
        brightness_response=3.40, amplitude_ceiling_mag=0.55,
    ),
    "UNKNOWN_HOT": _behavior(
        "stochastic_hot", flux_mode="outburst",
        default_period_days=10.0, default_amplitude_mag=0.35,
        amplitude_floor_mag=0.35, irregularity=1.0,
        temperature_variation=0.09, size_scale=1.05,
        size_response=0.85, intensity_scale=1.05,
        brightness_response=3.20, amplitude_ceiling_mag=0.60,
    ),
    "CATACLYSMIC": _behavior(
        "cataclysmic", flux_mode="outburst", default_period_days=30.0,
        default_amplitude_mag=3.0, temperature_variation=0.12,
        size_response=0.35, intensity_scale=1.10,
        brightness_response=1.25,
        amplitude_ceiling_mag=6.0,
    ),
    "ACV|CP|MCP|ROAM|ROAP|SXARI": _behavior(
        "rotational", default_period_days=3.0, default_amplitude_mag=0.08,
        size_response=0.08, amplitude_ceiling_mag=0.8,
    ),
    "ACYG": _behavior(
        "gentle_pulsation", default_period_days=10.0,
        default_amplitude_mag=0.15, irregularity=0.15,
        size_response=0.12, amplitude_ceiling_mag=1.0,
    ),
    "AGN": _behavior(
        "agn_noise", flux_mode="signed", default_period_days=120.0,
        default_amplitude_mag=0.3, irregularity=0.65,
        size_response=0.04, amplitude_ceiling_mag=1.5,
    ),
    "BCEP": _behavior(
        "gentle_pulsation", default_period_days=0.18,
        default_amplitude_mag=0.12, temperature_variation=0.008,
        size_response=0.08,
        amplitude_ceiling_mag=0.8,
    ),
    "DSCT|GDOR|SXPHE": _behavior(
        "multimode", default_period_days=0.10,
        default_amplitude_mag=0.18, temperature_variation=0.006,
        irregularity=0.08,
        size_response=0.09, amplitude_ceiling_mag=1.0,
    ),
    "ECL": _behavior(
        "eclipsing", flux_mode="dimming",
        default_period_days=1.0, default_amplitude_mag=0.6,
        size_response=0.0, amplitude_ceiling_mag=5.0,
    ),
    "ELL": _behavior(
        "ellipsoidal", default_period_days=2.0,
        default_amplitude_mag=0.15, size_response=0.08,
        amplitude_ceiling_mag=0.8,
    ),
    "EP": _behavior(
        "planet_transit", flux_mode="dimming",
        default_period_days=3.0, default_amplitude_mag=0.02,
        size_response=0.0, amplitude_ceiling_mag=0.15,
    ),
    "LPV": _behavior(
        "long_period",
        default_period_days=250.0, default_amplitude_mag=1.5,
        irregularity=0.40, temperature_variation=0.035,
        size_response=0.24, brightness_response=1.30,
        amplitude_ceiling_mag=5.0,
    ),
    "MICROLENSING": _behavior(
        "microlensing_event", flux_mode="outburst",
        default_period_days=30.0, default_amplitude_mag=1.0,
        size_response=0.18, amplitude_ceiling_mag=5.0,
        event_width=(0.08, 0.20),
    ),
    "RCB": _behavior(
        "rcb_decline", flux_mode="dimming",
        default_period_days=200.0, default_amplitude_mag=4.0,
        temperature_variation=-0.08, size_response=0.06,
        amplitude_ceiling_mag=8.0, event_width=(0.20, 0.42),
    ),
    "RS": _behavior(
        "rotational", default_period_days=8.0,
        default_amplitude_mag=0.18, irregularity=0.16,
        size_response=0.09, amplitude_ceiling_mag=1.2,
    ),
    "S": _behavior(
        "conservative", default_period_days=5.0,
        default_amplitude_mag=0.10, size_response=0.06,
        amplitude_ceiling_mag=0.8,
    ),
    "SDB": _behavior(
        "gentle_pulsation", default_period_days=0.08,
        default_amplitude_mag=0.08, size_response=0.05,
        amplitude_ceiling_mag=0.5,
    ),
    "SN": _behavior(
        "supernova_event", flux_mode="supernova",
        default_period_days=40.0, default_amplitude_mag=4.0,
        temperature_variation=0.30, size_response=0.25,
        intensity_scale=1.08, amplitude_ceiling_mag=7.0,
        event_width=(0.10, 0.22),
    ),
    "SOLAR_LIKE": _behavior(
        "solar_like", default_period_days=5.0,
        default_amplitude_mag=0.01, size_response=0.0,
        amplitude_ceiling_mag=0.05,
    ),
    "SPB": _behavior(
        "gentle_pulsation", default_period_days=2.0,
        default_amplitude_mag=0.08, temperature_variation=0.005,
        size_response=0.05,
        amplitude_ceiling_mag=0.5,
    ),
    "SYST": _behavior(
        "conservative", default_period_days=20.0,
        default_amplitude_mag=0.15, size_response=0.08,
        amplitude_ceiling_mag=1.0,
    ),
    "YSO": _behavior(
        "yso_irregular", flux_mode="signed",
        default_period_days=20.0, default_amplitude_mag=0.7,
        irregularity=0.70, temperature_variation=-0.06,
        size_response=0.10, amplitude_ceiling_mag=3.0,
    ),
    "OTHER": _behavior(
        "conservative", default_period_days=5.0,
        default_amplitude_mag=0.10, size_response=0.06,
        amplitude_ceiling_mag=0.8,
    ),
}


def get_variable_mode(value, default=VARIABLE_MODE_STANDARD):
    """Normalize the catalogue choice used by the variable animation."""
    text = str(default if value is None else value).strip().casefold()
    if not text:
        text = default
    try:
        return VARIABLE_MODE_ALIASES[text]
    except KeyError as error:
        raise ValueError(
            "choose standard, lbv_temperature_extremes, be, gcas, sdor, "
            "wr, unknown_hot_variables, or a named standard catalogue"
        ) from error


def _format_optional_number(value, digits=1):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"


def read_variable_catalogue(path):
    """Read FITS data without noisy non-standard Gaia unit warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*did not parse as fits unit.*",
            category=UnitsWarning,
        )
        return Table.read(path)


def _load_lbv_menu_rows(path=None):
    """Load compact LBV/SDOR menu information without changing the catalogue."""
    path = asset_path("sdor.fits") if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LBV/SDOR catalogue not found: {path}")
    table = read_variable_catalogue(path)
    if "source_id" not in table.colnames:
        raise KeyError(f"{path} is missing source_id")
    rows = []
    for index, row in enumerate(table, start=1):
        rows.append(
            (
                index,
                int(row["source_id"]),
                _format_optional_number(
                    row["teff_gspphot"]
                    if "teff_gspphot" in table.colnames
                    else np.nan,
                    0,
                ),
                _format_optional_number(
                    row["phot_g_mean_mag"]
                    if "phot_g_mean_mag" in table.colnames
                    else np.nan,
                    2,
                ),
                _format_optional_number(
                    row["classification_score"]
                    if "classification_score" in table.colnames
                    else np.nan,
                    3,
                ),
            )
        )
    return rows


def parse_lbv_source_selection(value, menu_rows):
    """Resolve comma-separated menu numbers or Gaia source_id values."""
    text = str(value).strip().casefold()
    if text in {"all", "all stars", "wszystkie", "*"}:
        return None
    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    if not tokens:
        raise ValueError("choose at least one LBV star or enter all")
    by_number = {number: source_id for number, source_id, *_ in menu_rows}
    valid_source_ids = {source_id for _, source_id, *_ in menu_rows}
    selected = []
    for token in tokens:
        try:
            number = int(token)
        except ValueError as error:
            raise ValueError(f"invalid LBV selection: {token}") from error
        if number in by_number:
            source_id = by_number[number]
        elif number in valid_source_ids:
            source_id = number
        else:
            raise ValueError(f"unknown LBV menu number/source_id: {number}")
        if source_id not in selected:
            selected.append(source_id)
    return tuple(selected)


def parse_lbv_group_selection(value):
    """Resolve one or more LBV subgroup numbers/names in catalogue order."""
    text = str(value).strip().casefold()
    if text in {"", "6", "all", "all lbv", "wszystkie", "*"}:
        return LBV_GROUP_KEYS
    aliases = {
        "1": "BE",
        "be": "BE",
        "2": "GCAS",
        "gcas": "GCAS",
        "3": "SDOR",
        "sdor": "SDOR",
        "lbv/sdor": "SDOR",
        "lbv_sdor": "SDOR",
        "4": "WR",
        "wr": "WR",
        "wolf-rayet": "WR",
        "wolf_rayet": "WR",
        "5": "UNKNOWN_HOT",
        "unknown": "UNKNOWN_HOT",
        "unclassified": "UNKNOWN_HOT",
        "unknown_hot": "UNKNOWN_HOT",
    }
    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    selected = set()
    for token in tokens:
        if token in {"6", "all", "wszystkie", "*"}:
            return LBV_GROUP_KEYS
        try:
            selected.add(aliases[token])
        except KeyError as error:
            raise ValueError(f"unknown LBV catalogue selection: {token}") from error
    if not selected:
        raise ValueError("choose at least one LBV catalogue")
    return tuple(group for group in LBV_GROUP_KEYS if group in selected)


def _prompt_lbv_mode():
    """Choose all hot-variable LBV groups or any subgroup combination."""
    single_modes = {
        "BE": VARIABLE_MODE_BE,
        "GCAS": VARIABLE_MODE_GCAS,
        "SDOR": VARIABLE_MODE_SDOR,
        "WR": VARIABLE_MODE_WR,
        "UNKNOWN_HOT": VARIABLE_MODE_UNKNOWN_HOT,
    }
    while True:
        try:
            response = console.prompt(
                "Choose one or more LBV catalogues (e.g. 1,2,3,4) "
                "[1 = BE, 2 = GCAS, 3 = LBV / SDOR, 4 = Wolf-Rayet, "
                "5 = unclassified hot variables, 6 = all LBV] "
                "(Enter = 6)"
            ).strip()
        except EOFError:
            response = ""
        try:
            selected = parse_lbv_group_selection(response)
        except ValueError as error:
            console.warning(str(error))
            continue
        if selected == LBV_GROUP_KEYS:
            os.environ.pop("AURORA_LBV_GROUPS", None)
            mode = VARIABLE_MODE_LBV_ALL
        elif len(selected) == 1:
            os.environ.pop("AURORA_LBV_GROUPS", None)
            mode = single_modes[selected[0]]
        else:
            os.environ["AURORA_LBV_GROUPS"] = ",".join(selected)
            mode = VARIABLE_MODE_LBV_ALL
        os.environ.pop("AURORA_LBV_SOURCE_IDS", None)
        return mode


def prompt_variable_mode(default=VARIABLE_MODE_STANDARD):
    """Choose all catalogues or one catalogue through a hierarchical menu."""
    default_mode = get_variable_mode(default)
    is_main_process = mp.current_process().name == "MainProcess"
    env_value = os.environ.get("AURORA_VARIABLE_MODE")
    if not is_main_process:
        return get_variable_mode(env_value, default_mode)
    if env_value:
        mode = get_variable_mode(env_value, default_mode)
        os.environ["AURORA_VARIABLE_MODE"] = mode
        return mode
    if not sys.stdin.isatty():
        os.environ["AURORA_VARIABLE_MODE"] = default_mode
        return default_mode
    while True:
        try:
            response = console.prompt(
                "Variable-star animation "
                "[1 = all catalogues, 2 = one catalogue] (Enter = 1)"
            ).strip()
        except EOFError:
            response = ""
        if response in {"", "1", "all", "wszystkie"}:
            mode = VARIABLE_MODE_STANDARD
            os.environ["AURORA_VARIABLE_MODE"] = mode
            return mode
        if response not in {"2", "one", "single", "pojedynczy"}:
            console.warning("Enter 1 or 2")
            continue

        single_modes = {
            "1": VARIABLE_MODE_RR_LYRAE,
            "2": VARIABLE_MODE_CEPHEIDS,
            "3": VARIABLE_MODE_ZZ_CETI,
            "4": VARIABLE_MODE_CATACLYSMIC,
            "5": VARIABLE_MODE_OTHER,
        }
        while True:
            try:
                catalogue = console.prompt(
                    "Choose one catalogue "
                    "[1 = RR Lyrae, 2 = Cepheids, 3 = ZZ Ceti, "
                    "4 = cataclysmic, 5 = other, 6 = LBV]"
                ).strip()
            except EOFError:
                catalogue = ""
            if catalogue == "6":
                mode = _prompt_lbv_mode()
                break
            try:
                mode = single_modes[catalogue]
            except KeyError:
                console.warning("Enter a catalogue number from 1 to 6")
                continue
            break
        os.environ["AURORA_VARIABLE_MODE"] = mode
        return mode


def _parse_animation_days(value):
    try:
        days = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("animation days must be a number") from error
    if not np.isfinite(days) or days <= 0.0:
        raise ValueError("animation days must be greater than zero")
    return days


def prompt_animation_days():
    """Ask globally for the simulated span immediately after resolution."""
    default_days = DEFAULT_ANIMATION_DAYS
    is_main_process = mp.current_process().name == "MainProcess"
    env_value = os.environ.get("AURORA_ANIMATION_DAYS")
    if env_value:
        days = _parse_animation_days(env_value)
        os.environ["AURORA_ANIMATION_DAYS"] = f"{days:g}"
        return days
    if not is_main_process or not sys.stdin.isatty():
        os.environ["AURORA_ANIMATION_DAYS"] = f"{default_days:g}"
        return default_days

    while True:
        try:
            response = console.prompt(
                "Animation time span in simulated days "
                f"(e.g. 500 or 3600; Enter = {default_days:g})"
            ).strip()
        except EOFError:
            response = ""
        try:
            days = _parse_animation_days(response or default_days)
        except ValueError:
            console.warning("Enter a positive number of days, e.g. 500 or 3600")
            continue
        os.environ["AURORA_ANIMATION_DAYS"] = f"{days:g}"
        return days


def _animation_days_tag(days):
    return f"{days:g}".replace(".", "p")


def _selected_lbv_source_ids_from_environment(variable_mode):
    if variable_mode != VARIABLE_MODE_LBV_SELECTED:
        return ()
    raw_value = os.environ.get("AURORA_LBV_SOURCE_IDS", "")
    menu_rows = _load_lbv_menu_rows()
    selected = parse_lbv_source_selection(raw_value, menu_rows)
    if selected is None:
        raise ValueError(
            "lbv_selected requires explicit AURORA_LBV_SOURCE_IDS; "
            "use mode sdor for all LBV / SDOR stars"
        )
    return selected


def _lbv_selection_tag(source_ids):
    payload = ",".join(map(str, source_ids)).encode("ascii")
    return hashlib.blake2b(payload, digest_size=5).hexdigest()


def _lbv_groups_from_environment(variable_mode):
    if variable_mode != VARIABLE_MODE_LBV_ALL:
        return ()
    raw_value = os.environ.get("AURORA_LBV_GROUPS", "")
    selected = parse_lbv_group_selection(raw_value)
    os.environ["AURORA_LBV_GROUPS"] = ",".join(selected)
    return selected

RESOLUTION = prompt_resolution()
ANIMATION_DAYS = prompt_animation_days()
RESOLUTION_TAG = RESOLUTION.tag
VIDEO_CONFIG = replace(
    BASE_VIDEO_CONFIG,
    width=RESOLUTION.width,
    height=RESOLUTION.height,
)

# ``full`` uses the Hammer all-sky map. ``region`` loads the PNG and layout
# written by main/aurora_sky_region_render.py. An environment value skips the
# prompt for scheduled and batch renders.
SKY_MAP_MODE = prompt_sky_map_mode()
SKY_MAP_BACKGROUND = (
    prompt_sky_map_background() if SKY_MAP_MODE == "full" else "plain"
)
VARIABLE_MODE = prompt_variable_mode()
LBV_GROUP_SELECTION = _lbv_groups_from_environment(VARIABLE_MODE)
LBV_SELECTED_SOURCE_IDS = _selected_lbv_source_ids_from_environment(
    VARIABLE_MODE
)
MAP_FILE = map_path(
    RESOLUTION.hammer_background_map_name(SKY_MAP_BACKGROUND)
)
REGION_MAP_FILE = Path(
    os.environ.get(
        "AURORA_REGION_MAP",
        str(region_map_path(RESOLUTION.region_map_name)),
    )
)
REGION_LAYOUT_FILE = Path(
    os.environ.get(
        "AURORA_REGION_LAYOUT",
        str(region_map_path(RESOLUTION.region_layout_name)),
    )
)
FIELD_SUFFIX = "_region" if SKY_MAP_MODE == "region" else ""
BACKGROUND_SUFFIX = sky_map_background_suffix(SKY_MAP_BACKGROUND)
VARIABLE_MODE_BASE_SUFFIX = (
    "_lbv_temperature_extremes"
    if VARIABLE_MODE == VARIABLE_MODE_LBV_TEMPERATURE_EXTREMES
    else
    (
        f"_sdor_selected_{_lbv_selection_tag(LBV_SELECTED_SOURCE_IDS)}"
        if VARIABLE_MODE == VARIABLE_MODE_LBV_SELECTED
        else {
        VARIABLE_MODE_STANDARD: "",
        VARIABLE_MODE_RR_LYRAE: "_rr_lyrae",
        VARIABLE_MODE_CEPHEIDS: "_cepheids",
        VARIABLE_MODE_ZZ_CETI: "_zz_ceti",
        VARIABLE_MODE_CATACLYSMIC: "_cataclysmic",
        VARIABLE_MODE_OTHER: "_other",
        VARIABLE_MODE_LBV_ALL: (
            "_lbv_all"
            if LBV_GROUP_SELECTION == LBV_GROUP_KEYS
            else "_lbv_" + "_".join(
                group.casefold() for group in LBV_GROUP_SELECTION
            )
        ),
        VARIABLE_MODE_BE: "_be",
        VARIABLE_MODE_GCAS: "_gcas",
        VARIABLE_MODE_SDOR: "_sdor",
        VARIABLE_MODE_WR: "_wr",
        VARIABLE_MODE_UNKNOWN_HOT: "_unknown_hot",
        }[VARIABLE_MODE]
    )
)
VARIABLE_MODE_SUFFIX = (
    f"{VARIABLE_MODE_BASE_SUFFIX}_"
    f"{_animation_days_tag(ANIMATION_DAYS)}d"
)
ARTIFACT_SUFFIX = f"{VARIABLE_MODE_SUFFIX}{FIELD_SUFFIX}{BACKGROUND_SUFFIX}"
VIDEO = video_path(
    f"aurora_variable_{RESOLUTION_TAG}{ARTIFACT_SUFFIX}_animation.mp4"
)
HVC1_VIDEO = video_path(
    f"aurora_variable_{RESOLUTION_TAG}{ARTIFACT_SUFFIX}_hvc1.mp4"
)
MOBILE_VIDEO = video_path(
    f"aurora_variable_{RESOLUTION_TAG}{ARTIFACT_SUFFIX}_mobile.mp4"
)

FPS = VIDEO_CONFIG.fps
FRAMES = VIDEO_CONFIG.frame_count
VIDEO_W = VIDEO_CONFIG.width
VIDEO_H = VIDEO_CONFIG.height
MAX_RENDER_WORKERS = min(MAX_RENDER_WORKERS_LIMIT, os.cpu_count() or 1)
SAVE_DEBUG_FRAMES = VIDEO_CONFIG.save_debug_frames
VARIABLE_MIN_SIGMA_16K = (
    ALL_CATALOGUES_MIN_SIGMA_16K
    if VARIABLE_MODE == VARIABLE_MODE_STANDARD
    else SINGLE_CATALOGUE_MIN_SIGMA_16K
)
VARIABLE_MAX_SIGMA_16K = (
    ALL_CATALOGUES_MAX_SIGMA_16K
    if VARIABLE_MODE == VARIABLE_MODE_STANDARD
    else SINGLE_CATALOGUE_MAX_SIGMA_16K
)
PNG_CACHE, OVERLAY_CACHE_DIR = cache_artifact_paths(
    f"aurora_variable_{RESOLUTION_TAG}{ARTIFACT_SUFFIX}",
    f"frames_variable_overlay_{RESOLUTION_TAG}{ARTIFACT_SUFFIX}",
    FRAME_CACHE_VERSION,
    VIDEO_CONFIG,
)

if VARIABLE_MODE == VARIABLE_MODE_LBV_TEMPERATURE_EXTREMES:
    VARIABLE_FILES = {"LBV": LBV_TEMPERATURE_EXTREMES_FILE}
elif VARIABLE_MODE == VARIABLE_MODE_LBV_ALL:
    VARIABLE_FILES = {
        group: LBV_GROUP_FILES[group]
        for group in (LBV_GROUP_SELECTION or LBV_GROUP_KEYS)
    }
elif VARIABLE_MODE in SINGLE_HOT_VARIABLE_GROUPS:
    HOT_GROUP = SINGLE_HOT_VARIABLE_GROUPS[VARIABLE_MODE]
    VARIABLE_FILES = {HOT_GROUP: LBV_GROUP_FILES[HOT_GROUP]}
elif VARIABLE_MODE in SINGLE_STANDARD_VARIABLE_FILES:
    VARIABLE_FILES = SINGLE_STANDARD_VARIABLE_FILES[VARIABLE_MODE]
else:
    VARIABLE_FILES = ALL_CATALOGUE_VARIABLE_FILES


# ─────────────────────────────────────────────────────────────
# Catalogue loading and deterministic limiting
# ─────────────────────────────────────────────────────────────

def _column(table, name, default, dtype=np.float32):
    if name not in table.colnames:
        return np.full(len(table), default, dtype=dtype)
    column = table[name]
    if hasattr(column, "filled"):
        column = column.filled(default)
    return np.asarray(column, dtype=dtype)


def _find_column(table, candidates):
    column_names = {
        str(name).casefold(): name
        for name in table.colnames
    }
    for candidate in candidates:
        name = column_names.get(candidate.casefold())
        if name is not None:
            return name
    return None


def _text_column(table, name, default, width=40):
    if name not in table.colnames:
        return np.full(len(table), default, dtype=f"U{width}")
    column = table[name]
    if hasattr(column, "filled"):
        column = column.filled(default)
    values = np.char.strip(np.asarray(column).astype(f"U{width}"))
    values[values == ""] = default
    return values


def _catalog_variable_types(table, catalogue_type):
    if catalogue_type != "OTHER":
        return np.full(len(table), catalogue_type, dtype="U40")

    class_column = _find_column(table, VARIABLE_CLASS_COLUMN_NAMES)
    if class_column is None:
        console.print(
            "  ! No classifier_class/variable_class column; "
            "using fallback OTHER style"
        )
        return np.full(len(table), "OTHER", dtype="U40")

    raw_types = _text_column(table, class_column, "OTHER")
    variable_types = raw_types.copy()
    for classifier_name, aurora_name in CLASSIFIER_TYPE_ALIASES.items():
        variable_types[raw_types == classifier_name] = aurora_name

    known = np.isin(variable_types, tuple(VARIABLE_BEHAVIOR))
    unknown_types = np.unique(variable_types[~known])
    if len(unknown_types):
        console.print(
            "  ! Unknown classifier classes mapped to OTHER: "
            + ", ".join(unknown_types)
        )
        variable_types[~known] = "OTHER"
    console.print(f"  → Using variability classes from column: {class_column}")
    return variable_types


def _magellanic_sky_mask(lon, lat):
    """Locate the LMC/SMC when a catalogue has no parallax column."""
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    result = np.zeros(len(lon), dtype=bool)
    finite = np.isfinite(lon) & np.isfinite(lat)

    for _, centre_lon, centre_lat, radius in MAGELLANIC_SKY_REGIONS:
        centre_lon = np.radians(centre_lon)
        centre_lat = np.radians(centre_lat)
        cosine_distance = (
            np.sin(lat) * np.sin(centre_lat)
            + np.cos(lat)
            * np.cos(centre_lat)
            * np.cos(lon - centre_lon)
        )
        result |= finite & (cosine_distance >= np.cos(np.radians(radius)))
    return result


def _magellanic_membership_mask(lon, lat, parallax=None):
    """Identify MC objects by sky footprint or a compatible parallax.

    Gaia parallaxes at Magellanic-Cloud distances are noisy.  Using parallax
    alone incorrectly labels many objects projected directly on the LMC/SMC as
    Milky Way foreground, allowing them to bypass MAGELLANIC_TOTAL_LIMIT.
    """
    sky_mask = _magellanic_sky_mask(lon, lat)
    if parallax is None:
        return sky_mask
    parallax = np.asarray(parallax)
    distance_mask = (
        np.isfinite(parallax)
        & (parallax >= 0.0)
        & (parallax <= MAGELLANIC_PARALLAX_MAX_MAS)
    )
    return sky_mask | distance_mask


def _random_sample(candidates, limit, rng):
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) <= limit:
        return candidates
    return np.sort(rng.choice(candidates, size=limit, replace=False))


def _sample_sky_balanced_indices(lon, lat, candidates, limit, rng):
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) <= limit:
        return candidates

    count = len(lon)
    available = np.ones(count, dtype=bool)
    available[:] = False
    available[candidates] = True
    selected = []
    centre_distance = np.abs((lon + 180.0) % 360.0 - 180.0)
    centre = np.flatnonzero(
        (np.abs(lat) <= SKY_BALANCE_CENTRE_MAX_LATITUDE_DEG)
        & (centre_distance <= SKY_BALANCE_CENTRE_MAX_LONGITUDE_DEG)
    )

    def pick(candidates, amount):
        candidates = candidates[available[candidates]]
        if len(candidates) > amount:
            candidates = np.sort(
                rng.choice(candidates, size=amount, replace=False)
            )
        available[candidates] = False
        selected.append(candidates)

    # Preserve the previous 650/450/250/150 sky-distribution ratio at every
    # requested catalogue size, allocating any rounding remainder centrally.
    base_quotas = SKY_BALANCE_BASE_QUOTAS
    quotas = (base_quotas * limit) // base_quotas.sum()
    quotas[0] += limit - quotas.sum()

    pick(centre, int(quotas[0]))
    for zone, amount in zip(("disk", "mid", "halo"), quotas[1:]):
        abs_lat = np.abs(lat)
        if zone == "disk":
            mask = abs_lat <= SKY_BALANCE_DISK_MAX_LATITUDE_DEG
        elif zone == "mid":
            mask = (
                (abs_lat > SKY_BALANCE_DISK_MAX_LATITUDE_DEG)
                & (abs_lat <= SKY_BALANCE_MID_MAX_LATITUDE_DEG)
            )
        else:
            mask = abs_lat > SKY_BALANCE_MID_MAX_LATITUDE_DEG
        pick(np.flatnonzero(mask), int(amount))

    indices = np.concatenate(selected) if selected else np.empty(0, dtype=int)
    if len(indices) < limit:
        pick(np.flatnonzero(available), limit - len(indices))
        indices = np.concatenate(selected)
    return indices[:limit]


def _sample_indices(lon, lat, parallax, limit):
    if len(lon) != len(lat):
        raise ValueError("lon and lat must have equal lengths")
    if parallax is not None and len(parallax) != len(lon):
        raise ValueError("parallax must have the same length as lon and lat")
    if limit < 0:
        raise ValueError("sample limit cannot be negative")

    if parallax is None:
        magellanic_mask = _magellanic_membership_mask(lon, lat)
        valid_position = np.isfinite(lon) & np.isfinite(lat)
        magellanic_candidates = np.flatnonzero(magellanic_mask)
        milky_way_candidates = np.flatnonzero(
            valid_position & ~magellanic_mask
        )
    else:
        parallax = np.asarray(parallax)
        magellanic_mask = _magellanic_membership_mask(lon, lat, parallax)
        magellanic_candidates = np.flatnonzero(magellanic_mask)
        milky_way_candidates = np.flatnonzero(
            ~magellanic_mask
            & np.isfinite(parallax)
            & (parallax > MAGELLANIC_PARALLAX_MAX_MAS)
        )

    rng = np.random.default_rng(CATALOG_SAMPLE_SEED)

    # Each bucket receives the complete per-class limit. Values immediately
    # above the LMC/SMC threshold remain eligible for the Milky Way draw.
    magellanic = _random_sample(
        magellanic_candidates,
        limit,
        rng,
    )
    milky_way = _sample_sky_balanced_indices(
        lon,
        lat,
        milky_way_candidates,
        limit,
        rng,
    )
    return np.concatenate((magellanic, milky_way))


def load_variable_stars(sky_region=None):
    console.print("\n[AURORA] Loading variable-star catalogues")
    console.print("─" * 45)
    groups = []

    for catalogue_number, (variable_type, path) in enumerate(
        VARIABLE_FILES.items()
    ):
        if not path.exists():
            console.print(f"  ! Catalogue not found, skipping: {path}")
            continue
        console.print(f"  → Loading {variable_type}: {path}")
        table = read_variable_catalogue(path)
        if "l" not in table.colnames or "b" not in table.colnames:
            raise KeyError(f"{path} is missing l or b")

        lon = _column(table, "l", np.nan)
        lat = _column(table, "b", np.nan)
        parallax_name = _find_column(table, PARALLAX_COLUMN_NAMES)
        if parallax_name is None:
            parallax = None
            console.print(
                "  ! No parallax column; identifying LMC/SMC by "
                "Galactic sky regions"
            )
        else:
            parallax = _column(table, parallax_name, np.nan)
            console.print(f"  → Using parallax column: {parallax_name}")
        catalogue_types = _catalog_variable_types(table, variable_type)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if sky_region is not None:
            valid &= galactic_region_mask(lon, lat, sky_region)
        valid_indices = np.flatnonzero(valid)
        if VARIABLE_MODE == VARIABLE_MODE_LBV_SELECTED:
            catalogue_source_id = _column(
                table, "source_id", -1, dtype=np.int64
            )
            requested = np.asarray(
                LBV_SELECTED_SOURCE_IDS, dtype=np.int64
            )
            selected_mask = np.isin(
                catalogue_source_id[valid_indices], requested
            )
            valid_indices = valid_indices[selected_mask]
            found = set(
                catalogue_source_id[valid_indices].astype(np.int64).tolist()
            )
            missing_requested = [
                source_id
                for source_id in LBV_SELECTED_SOURCE_IDS
                if source_id not in found
            ]
            if missing_requested:
                console.warning(
                    "Selected LBV source_id values unavailable in this field: "
                    + ", ".join(map(str, missing_requested))
                )
            console.print(
                f"  → Explicit LBV selection retained "
                f"{len(valid_indices):,}/{len(LBV_SELECTED_SOURCE_IDS):,} stars"
            )
        limit = VARIABLE_LIMITS[variable_type]
        if variable_type == "OTHER":
            selected_parts = []
            valid_types = catalogue_types[valid_indices]
            for subtype in np.unique(valid_types):
                subtype_indices = valid_indices[valid_types == subtype]
                limited = _sample_indices(
                    lon[subtype_indices],
                    lat[subtype_indices],
                    (
                        parallax[subtype_indices]
                        if parallax is not None
                        else None
                    ),
                    limit,
                )
                selected_parts.append(subtype_indices[limited])
            indices = (
                np.concatenate(selected_parts)
                if selected_parts
                else np.empty(0, dtype=np.int64)
            )
            console.print(
                f"  → OTHER contains {len(np.unique(valid_types))} "
                f"styled variability classes"
            )
        else:
            limited = _sample_indices(
                lon[valid_indices],
                lat[valid_indices],
                (
                    parallax[valid_indices]
                    if parallax is not None
                    else None
                ),
                limit,
            )
            indices = valid_indices[limited]
        selected_magellanic = _magellanic_membership_mask(
            lon[indices],
            lat[indices],
            parallax[indices] if parallax is not None else None,
        )
        magellanic_count = np.count_nonzero(selected_magellanic)
        milky_way_count = len(indices) - magellanic_count

        period = _column(table, "period", np.nan)
        if "alternate_period" in table.colnames:
            alternate_period = _column(table, "alternate_period", np.nan)
            missing_period = ~np.isfinite(period) | (period <= 0.0)
            usable_alternate = np.isfinite(alternate_period) & (
                alternate_period > 0.0
            )
            use_alternate = missing_period & usable_alternate
            period[use_alternate] = alternate_period[use_alternate]
        source_id = _column(table, "source_id", -1, dtype=np.int64)
        missing_source_id = source_id < 0
        if np.any(missing_source_id):
            # A stable catalogue/row key is used only when source_id is absent.
            source_id[missing_source_id] = (
                np.int64(catalogue_number + 1) * np.int64(10**12)
                + np.flatnonzero(missing_source_id).astype(np.int64)
            )

        amplitude = _column(table, "amplitude", np.nan)
        if variable_type == "LBV" or variable_type in LBV_GROUP_KEYS:
            # The resolved catalogues carry Gaia summary statistics rather
            # than a dedicated SOS amplitude.  Prefer the robust trimmed range
            # and fall back to the full range before class defaults are used,
            # in both single- and all-catalogue modes.
            for amplitude_name in (
                "trimmed_range_mag_g_fov",
                "range_mag_g_fov",
            ):
                candidate = _column(table, amplitude_name, np.nan)
                missing_amplitude = ~np.isfinite(amplitude) | (amplitude < 0.0)
                usable = np.isfinite(candidate) & (candidate >= 0.0)
                amplitude[missing_amplitude & usable] = candidate[
                    missing_amplitude & usable
                ]

        groups.append(
            {
                "l": lon[indices],
                "b": lat[indices],
                "source_id": source_id[indices],
                "period": period[indices],
                "amplitude": amplitude[indices],
                "phase": _column(table, "phase", np.nan)[indices],
                "teff": _column(table, "teff_gspphot", np.nan)[indices],
                "bp_rp": _column(table, "bp_rp", np.nan)[indices],
                "bp_rp_intrinsic": _column(
                    table, "bp_rp_intrinsic", np.nan
                )[indices],
                "reddening_quality": _text_column(
                    table, "reddening_quality", "missing", width=9
                )[indices],
                "extinction_flags": _text_column(
                    table, "extinction_flags", "", width=128
                )[indices],
                "ag_gspphot": _column(table, "ag_gspphot", np.nan)[indices],
                "ebpminrp_gspphot": _column(
                    table, "ebpminrp_gspphot", np.nan
                )[indices],
                "luminosity": _column(table, "lum_flame", np.nan)[indices],
                "radius": _column(table, "radius_gspphot", np.nan)[indices],
                "mass": _column(table, "mass_flame", np.nan)[indices],
                "classification_score": _column(
                    table, "classification_score", np.nan
                )[indices],
                "type": catalogue_types[indices],
                "magellanic": selected_magellanic,
            }
        )
        limit_description = (
            "limit per class/bucket"
            if variable_type == "OTHER"
            else "limit per bucket"
        )
        console.print(
            f"  ✓ Selected {len(indices):,}/{len(table):,} objects "
            f"(LMC/SMC: {magellanic_count:,}, "
            f"Milky Way: {milky_way_count:,}, "
            f"{limit_description}: {limit:,})"
        )

    if not groups:
        raise RuntimeError("No variable-star catalogues found")

    combined = {
        name: np.concatenate([group[name] for group in groups])
        for name in groups[0]
    }
    if not len(combined["l"]):
        raise RuntimeError(
            "No selected variable stars fall inside the configured sky region"
        )
    magellanic_indices = np.flatnonzero(combined["magellanic"])
    if len(magellanic_indices) > MAGELLANIC_TOTAL_LIMIT:
        rng = np.random.default_rng(CATALOG_SAMPLE_SEED)
        retained_magellanic = np.sort(
            rng.choice(
                magellanic_indices,
                size=MAGELLANIC_TOTAL_LIMIT,
                replace=False,
            )
        )
        milky_way_indices = np.flatnonzero(~combined["magellanic"])
        retained = np.sort(
            np.concatenate((retained_magellanic, milky_way_indices))
        )
        removed = len(magellanic_indices) - MAGELLANIC_TOTAL_LIMIT
        combined = {
            name: values[retained]
            for name, values in combined.items()
        }
        console.print(
            f"  ✓ Global LMC/SMC limit: retained "
            f"{MAGELLANIC_TOTAL_LIMIT:,}, removed {removed:,}"
        )
    else:
        console.print(
            f"  ✓ Global LMC/SMC limit: retained "
            f"{len(magellanic_indices):,}/{MAGELLANIC_TOTAL_LIMIT:,}"
        )
    l = np.radians(combined["l"]).astype(np.float32)
    b = np.radians(combined["b"]).astype(np.float32)
    l = -np.where(l > np.pi, l - 2.0 * np.pi, l).astype(np.float32)
    console.print(f"  ✓ Total variable stars prepared: {len(l):,}")
    console.print("=== AURORA VARIABLE STAR CATALOG READY ===")
    return (
        l,
        b,
        combined["period"].astype(np.float32),
        combined["amplitude"].astype(np.float32),
        combined["phase"].astype(np.float32),
        combined["type"],
        combined["teff"].astype(np.float32),
        combined["bp_rp"].astype(np.float32),
        combined["bp_rp_intrinsic"].astype(np.float32),
        combined["reddening_quality"],
        combined["extinction_flags"],
        combined["ag_gspphot"].astype(np.float32),
        combined["ebpminrp_gspphot"].astype(np.float32),
        combined["source_id"].astype(np.int64),
        combined["luminosity"].astype(np.float32),
        combined["radius"].astype(np.float32),
        combined["mass"].astype(np.float32),
        combined["classification_score"].astype(np.float32),
    )


def prepare_animation_colour_inputs(
    teff,
    bp_rp_intrinsic,
    reddening_quality,
    bp_rp,
    variable_types,
    extinction_flags=None,
    ag_gspphot=None,
    ebpminrp_gspphot=None,
):
    """Resolve teff -> reliable intrinsic -> raw colour -> class fallback.

    Gaia extinction and reddening values are treated as uncertain estimates.
    They protect WR and strongly extinguished SDOR/LBV objects from a falsely
    cool interpretation of red observed BP-RP; they never establish a class.
    """
    teff = np.asarray(teff, dtype=np.float32).copy()
    intrinsic = np.asarray(bp_rp_intrinsic, dtype=np.float32)
    raw = np.asarray(bp_rp, dtype=np.float32)
    quality = np.char.lower(np.asarray(reddening_quality).astype("U9"))
    variable_types = np.asarray(variable_types).astype("U40")
    count = len(teff)
    flags = (
        np.full(count, "", dtype="U128")
        if extinction_flags is None
        else np.char.lower(np.asarray(extinction_flags).astype("U128"))
    )
    ag = (
        np.full(count, np.nan, dtype=np.float32)
        if ag_gspphot is None
        else np.asarray(ag_gspphot, dtype=np.float32)
    )
    reddening = (
        np.full(count, np.nan, dtype=np.float32)
        if ebpminrp_gspphot is None
        else np.asarray(ebpminrp_gspphot, dtype=np.float32)
    )
    if not all(
        len(values) == count
        for values in (
            intrinsic,
            raw,
            quality,
            variable_types,
            flags,
            ag,
            reddening,
        )
    ):
        raise ValueError("colour input arrays must have equal lengths")

    valid_teff = (
        np.isfinite(teff)
        & (teff >= MIN_PLAUSIBLE_TEFF_K)
        & (teff <= MAX_PLAUSIBLE_TEFF_K)
    )
    teff[~valid_teff] = np.nan
    invalid_intrinsic_flags = np.zeros(count, dtype=bool)
    for invalid_flag in INVALID_INTRINSIC_COLOUR_FLAGS:
        invalid_intrinsic_flags |= np.char.find(flags, invalid_flag) >= 0
    reliable_intrinsic = (
        np.isfinite(intrinsic)
        & np.isin(quality, tuple(RELIABLE_REDDENING_QUALITIES))
        & ~invalid_intrinsic_flags
    )
    selected_bp_rp = np.where(reliable_intrinsic, intrinsic, raw).astype(
        np.float32
    )
    strong_extinction = (
        (np.char.find(flags, "strong_extinction") >= 0)
        | (np.isfinite(ag) & (ag >= STRONG_EXTINCTION_AG_MIN))
        | (
            np.isfinite(reddening)
            & (reddening >= STRONG_REDDENING_MIN)
        )
    )

    # A red observed WR colour can be entirely caused by dust.  Without a
    # reliable temperature or dereddened colour, use a safely hot class colour
    # instead of interpreting raw BP-RP as a cool photosphere.
    wr_safe_fallback = (
        (variable_types == "WR") & ~valid_teff & ~reliable_intrinsic
    )
    sdor_extinction_fallback = (
        np.isin(variable_types, ("SDOR", "LBV"))
        & ~valid_teff
        & ~reliable_intrinsic
        & strong_extinction
        & np.isfinite(raw)
        & (raw > HOT_RAW_RED_BP_RP_MIN)
    )
    extinction_protected = wr_safe_fallback | sdor_extinction_fallback
    selected_bp_rp[extinction_protected] = np.nan

    missing_physical_colour = ~valid_teff & ~np.isfinite(selected_bp_rp)
    class_fallback_count = 0
    for class_name, fallback_temperature in (
        HOT_CLASS_FALLBACK_TEMPERATURES.items()
    ):
        use_fallback = missing_physical_colour & (variable_types == class_name)
        teff[use_fallback] = fallback_temperature
        class_fallback_count += int(np.count_nonzero(use_fallback))

    return teff, selected_bp_rp, {
        "teff": int(np.count_nonzero(valid_teff)),
        "intrinsic": int(np.count_nonzero(~valid_teff & reliable_intrinsic)),
        "raw": int(
            np.count_nonzero(
                ~valid_teff
                & ~reliable_intrinsic
                & np.isfinite(selected_bp_rp)
            )
        ),
        "class_fallback": class_fallback_count,
        "extinction_protected": int(np.count_nonzero(extinction_protected)),
    }


def apply_catalog_physical_scaling(
    behavior,
    luminosity,
    radius,
    mass,
    classification_score,
):
    """Apply bounded, auxiliary Gaia scaling without changing object types."""
    intensity = behavior["intensity_scale"].copy()
    size = behavior["size_scale"].copy()

    luminosity = np.asarray(luminosity, dtype=np.float32)
    valid_luminosity = np.isfinite(luminosity) & (luminosity > 0.0)
    luminosity_factor = np.ones(len(intensity), dtype=np.float32)
    luminosity_factor[valid_luminosity] = np.clip(
        1.0
        + LUMINOSITY_SCALING_SLOPE
        * (
            np.log10(luminosity[valid_luminosity])
            - LUMINOSITY_REFERENCE_LOG10
        ),
        *LUMINOSITY_FACTOR_RANGE,
    )

    mass = np.asarray(mass, dtype=np.float32)
    valid_mass = np.isfinite(mass) & (mass > 0.0)
    mass_factor = np.ones(len(intensity), dtype=np.float32)
    mass_factor[valid_mass] = np.clip(
        MASS_FACTOR_BASE
        + MASS_FACTOR_SLOPE
        * np.log1p(mass[valid_mass] / MASS_REFERENCE_SOLAR),
        *MASS_FACTOR_RANGE,
    )

    classification_score = np.asarray(classification_score, dtype=np.float32)
    valid_score = np.isfinite(classification_score)
    confidence_factor = np.ones(len(intensity), dtype=np.float32)
    confidence_factor[valid_score] = (
        CONFIDENCE_FACTOR_BASE
        + CONFIDENCE_FACTOR_RANGE
        * np.clip(classification_score[valid_score], 0.0, 1.0)
    )

    radius = np.asarray(radius, dtype=np.float32)
    valid_radius = np.isfinite(radius) & (radius > 0.0)
    radius_factor = np.ones(len(size), dtype=np.float32)
    radius_factor[valid_radius] = np.clip(
        np.power(
            radius[valid_radius] / RADIUS_REFERENCE_SOLAR,
            RADIUS_SCALING_EXPONENT,
        ),
        *RADIUS_FACTOR_RANGE,
    )

    behavior["intensity_scale"] = (
        intensity * luminosity_factor * mass_factor * confidence_factor
    ).astype(np.float32)
    behavior["size_scale"] = (size * radius_factor).astype(np.float32)
    return {
        "luminosity": int(np.count_nonzero(valid_luminosity)),
        "radius": int(np.count_nonzero(valid_radius)),
        "mass": int(np.count_nonzero(valid_mass)),
        "classification_score": int(np.count_nonzero(valid_score)),
    }


def apply_reference_variable_star_size(behavior, variable_mode):
    """Use a density-aware size on both full-sky and region maps."""
    count = len(behavior["size_scale"])
    size_scale = (
        ALL_CATALOGUES_MAP_SIZE_SCALE
        if variable_mode == VARIABLE_MODE_STANDARD
        else VARIABLE_MAP_SIZE_SCALE
    )
    behavior["size_scale"] = np.full(
        count,
        size_scale,
        dtype=np.float32,
    )
    return behavior


def prepare_behavior_arrays(variable_types, period, amplitude, phase):
    """Resolve class interpretation while preserving every valid star value."""
    count = len(variable_types)
    model = np.full(count, -1, dtype=np.int8)
    flux_mode = np.full(count, -1, dtype=np.int8)
    size_scale = np.full(count, np.nan, dtype=np.float32)
    size_response = np.full(count, np.nan, dtype=np.float32)
    intensity_scale = np.full(count, np.nan, dtype=np.float32)
    brightness_response = np.full(count, np.nan, dtype=np.float32)
    temperature_response = np.zeros(count, dtype=np.float32)
    irregularity = np.zeros(count, dtype=np.float32)
    period_default = np.full(count, np.nan, dtype=np.float32)
    amplitude_default = np.full(count, np.nan, dtype=np.float32)
    amplitude_floor = np.zeros(count, dtype=np.float32)
    amplitude_ceiling = np.full(count, np.nan, dtype=np.float32)
    event_width_min = np.full(count, np.nan, dtype=np.float32)
    event_width_max = np.full(count, np.nan, dtype=np.float32)
    model_groups = []

    for name, behavior in VARIABLE_BEHAVIOR.items():
        indices = np.flatnonzero(variable_types == name)
        if not len(indices):
            continue
        model[indices] = VARIABLE_MODEL_CODES[behavior.light_curve]
        flux_mode[indices] = VARIABLE_FLUX_MODES[behavior.flux_mode]
        size_scale[indices] = behavior.size_scale
        size_response[indices] = behavior.size_response
        intensity_scale[indices] = behavior.intensity_scale
        brightness_response[indices] = behavior.brightness_response
        temperature_response[indices] = behavior.temperature_variation
        irregularity[indices] = behavior.irregularity
        period_default[indices] = behavior.default_period_days
        amplitude_default[indices] = behavior.default_amplitude_mag
        amplitude_floor[indices] = behavior.amplitude_floor_mag
        amplitude_ceiling[indices] = behavior.amplitude_ceiling_mag
        event_width_min[indices] = behavior.event_width[0]
        event_width_max[indices] = behavior.event_width[1]
        model_groups.append(
            (
                VARIABLE_MODEL_CODES[behavior.light_curve],
                VARIABLE_FLUX_MODES[behavior.flux_mode],
                indices.astype(np.int32, copy=False),
            )
        )

    missing = (
        (model < 0)
        | (flux_mode < 0)
        | ~np.isfinite(period_default)
        | ~np.isfinite(amplitude_default)
        | ~np.isfinite(brightness_response)
    )
    if np.any(missing):
        names = ", ".join(np.unique(variable_types[missing]))
        raise KeyError(f"Missing variable behavior for classes: {names}")

    period = np.asarray(period, dtype=np.float32).copy()
    missing_period = ~np.isfinite(period) | (period <= 0.0)
    period[missing_period] = period_default[missing_period]
    period = period.astype(np.float32, copy=False)

    amplitude = np.asarray(amplitude, dtype=np.float32).copy()
    missing_amplitude = ~np.isfinite(amplitude) | (amplitude < 0.0)
    amplitude[missing_amplitude] = amplitude_default[missing_amplitude]
    amplitude = np.maximum(amplitude, amplitude_floor)
    amplitude = np.minimum(amplitude, amplitude_ceiling).astype(np.float32)

    phase = np.asarray(phase, dtype=np.float32).copy()
    phase = np.remainder(phase, np.float32(2.0 * np.pi))
    model_groups = tuple(model_groups)
    return {
        "period": period,
        "amplitude": amplitude,
        "phase": phase,
        "model": model,
        "model_groups": model_groups,
        "flux_mode": flux_mode,
        "size_scale": size_scale,
        "size_response": size_response,
        "intensity_scale": intensity_scale,
        "brightness_response": brightness_response,
        "temperature_response": temperature_response,
        "irregularity": irregularity,
        "event_width_min": event_width_min,
        "event_width_max": event_width_max,
        "missing_period_count": int(np.count_nonzero(missing_period)),
        "missing_amplitude_count": int(np.count_nonzero(missing_amplitude)),
    }


def deterministic_uniform(source_id, seed):
    """Stable per-source values in [0, 1), independent of worker scheduling."""
    value = np.asarray(source_id, dtype=np.uint64) ^ np.uint64(seed)
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return ((value >> np.uint64(40)).astype(np.float32) / np.float32(2**24))


def prepare_simulation_phases(catalog_phase, source_id):
    """Combine catalogue phases with stable, independent simulation offsets.

    The catalogue phase remains part of every valid value.  The additional
    source-specific offset represents the arbitrary start epoch of this movie
    and prevents equal or zero catalogue phases from synchronising the sky.
    Missing catalogue phases use only that deterministic offset.
    """
    catalog_phase = np.asarray(catalog_phase, dtype=np.float32)
    source_id = np.asarray(source_id, dtype=np.int64)
    if catalog_phase.shape != source_id.shape:
        raise ValueError("Catalogue phase and source_id shapes must match")

    valid_catalog_phase = np.isfinite(catalog_phase)
    base_phase = np.where(valid_catalog_phase, catalog_phase, 0.0).astype(
        np.float32,
        copy=False,
    )
    phase_offset = (
        np.float32(2.0 * np.pi)
        * deterministic_uniform(source_id, PULSATION_PHASE_SEED)
    ).astype(np.float32)
    simulation_phase = np.remainder(
        base_phase + phase_offset,
        np.float32(2.0 * np.pi),
    ).astype(np.float32)
    return simulation_phase, valid_catalog_phase


# ─────────────────────────────────────────────────────────────
# Sparse frame cache
# ─────────────────────────────────────────────────────────────

def _cache_path(frame):
    return OVERLAY_CACHE_DIR / f"overlay_{frame:04d}.npz"


def frame_cache_signature(img_w, img_h, *arrays):
    config = np.asarray(
        [
            FRAME_CACHE_VERSION,
            FRAMES,
            img_w,
            img_h,
            ANIMATION_DAYS,
            INTRO_FRACTION,
            OUTRO_FRACTION,
            APPEARANCE_FRACTION,
            FADE_IN_TIME,
            FADE_OUT_TIME,
        ],
        dtype=np.float64,
    )
    if VARIABLE_MODE in HOT_VARIABLE_MODES:
        arrays = (
            *arrays,
            np.asarray(
                [HOT_VARIABLE_CACHE_CODES[VARIABLE_MODE]],
                dtype=np.uint8,
            ),
        )
    return array_signature(config, *arrays)


def load_frame_cache_safe(
    path,
    img_w,
    img_h,
    signature,
    star_count,
    verbose=True,
):
    path = Path(path)
    if verbose:
        console.print(f"  → Loading overlay cache: {path}")
    required = {
        "version",
        "img_w",
        "img_h",
        "star_count",
        "signature",
        "indices",
        "sigma",
        "alpha",
        "color_indices",
        "colors",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                raise KeyError("missing sparse-cache fields")
            if int(data["version"]) != FRAME_CACHE_VERSION:
                raise ValueError("unsupported cache version")
            if int(data["img_w"]) != img_w or int(data["img_h"]) != img_h:
                raise ValueError("cache resolution differs from the current map")
            if int(data["star_count"]) != star_count:
                raise ValueError("cache star count differs from current catalogues")
            if str(data["signature"].item()) != signature:
                raise ValueError("cache inputs differ from current catalogues")
            arrays = (
                np.asarray(data["indices"], dtype=np.int32),
                np.asarray(data["sigma"], dtype=np.float32),
                np.asarray(data["alpha"], dtype=np.float32),
                np.asarray(data["color_indices"], dtype=np.int32),
                np.asarray(data["colors"], dtype=np.float32),
            )
            if len(arrays[0]) != len(arrays[1]) or len(arrays[0]) != len(arrays[2]):
                raise ValueError("invalid sparse-cache shapes")
            if arrays[4].shape != (len(arrays[3]), 3):
                raise ValueError("invalid sparse dynamic-colour shape")
            if np.any(arrays[0] < 0) or np.any(arrays[0] >= star_count):
                raise ValueError("invalid sparse-cache star index")
            if np.any(arrays[3] < 0) or np.any(arrays[3] >= star_count):
                raise ValueError("invalid sparse-cache colour index")
            if len(arrays[3]) and not np.all(np.isin(arrays[3], arrays[0])):
                raise ValueError("dynamic colours refer to invisible stars")
            if not all(np.all(np.isfinite(array)) for array in arrays):
                raise ValueError("non-finite sparse-cache values")
            return arrays
    except (
        BadZipFile,
        EOFError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        console.print(f"  ! Corrupted overlay cache detected: {path} ({error})")
        try:
            path.unlink()
            console.print(f"  ✓ Removed corrupted overlay cache: {path}")
        except FileNotFoundError:
            pass
        except OSError as remove_error:
            console.print(f"  ! Could not remove cache: {path} ({remove_error})")
        return None


def save_frame_cache_safe(
    path,
    parameters,
    img_w,
    img_h,
    signature,
    star_count,
    verbose=True,
):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    if verbose:
        console.print(f"  → Saving overlay cache atomically: {path}")
    indices, sigma, alpha, color_indices, colors = parameters
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.int16(FRAME_CACHE_VERSION),
                img_w=np.int32(img_w),
                img_h=np.int32(img_h),
                star_count=np.int32(star_count),
                signature=np.str_(signature),
                indices=np.asarray(indices, dtype=np.int32),
                sigma=np.asarray(sigma, dtype=np.float32),
                alpha=np.asarray(alpha, dtype=np.float32),
                color_indices=np.asarray(color_indices, dtype=np.int32),
                colors=np.asarray(colors, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


# ─────────────────────────────────────────────────────────────
# Worker state and frame parameter generation
# ─────────────────────────────────────────────────────────────

_worker_data = None


def _init_worker(*worker_data):
    global _worker_data
    _worker_data = worker_data


def render_frame_parameters(frame):
    (
        px,
        py,
        period,
        amplitude,
        phase,
        model_groups,
        flux_mode,
        size_scale,
        size_response,
        intensity_scale,
        brightness_response,
        temperature_base,
        temperature_response,
        vivid_temperature_colors,
        irregularity,
        mode_phase_1,
        mode_phase_2,
        event_center,
        event_width,
        event_shape,
        appearance_delay,
        fade_delay,
        frame_times,
        active_progress,
        frame_opacity,
        img_h,
    ) = _worker_data

    return frame, variable_frame_parameters(
        elapsed=frame_times[frame],
        timeline_progress=active_progress[frame],
        timeline_opacity=frame_opacity[frame],
        period=period,
        amplitude=amplitude,
        phase=phase,
        model_groups=model_groups,
        flux_mode=flux_mode,
        size_scale=size_scale,
        size_response=size_response,
        intensity_scale=intensity_scale,
        brightness_response=brightness_response,
        temperature_base=temperature_base,
        temperature_response=temperature_response,
        vivid_temperature_colors=vivid_temperature_colors,
        irregularity=irregularity,
        mode_phase_1=mode_phase_1,
        mode_phase_2=mode_phase_2,
        event_center=event_center,
        event_width=event_width,
        event_shape=event_shape,
        appearance_delay=appearance_delay,
        fade_delay=fade_delay,
        image_height=img_h,
        base_sigma_16k=VARIABLE_BASE_SIGMA_16K,
        min_sigma_16k=VARIABLE_MIN_SIGMA_16K,
        max_sigma_16k=VARIABLE_MAX_SIGMA_16K,
        intro_fraction=INTRO_FRACTION,
        outro_fraction=OUTRO_FRACTION,
        fade_in_time=FADE_IN_TIME,
        fade_out_time=FADE_OUT_TIME,
    )


def draw_cached_frame(canvas, parameters, px, py, colors):
    indices, sigma, alpha, color_indices, dynamic_colors = parameters
    if not len(color_indices):
        draw_gaussians_u8(
            canvas, px[indices], py[indices], sigma, alpha, colors[indices]
        )
        return

    color_positions = np.searchsorted(indices, color_indices)
    if (
        np.any(color_positions >= len(indices))
        or not np.array_equal(indices[color_positions], color_indices)
    ):
        raise ValueError("dynamic colour indices are inconsistent with frame indices")
    static = np.ones(len(indices), dtype=bool)
    static[color_positions] = False
    draw_gaussians_u8(
        canvas,
        px[indices[static]],
        py[indices[static]],
        sigma[static],
        alpha[static],
        colors[indices[static]],
    )
    draw_gaussians_u8(
        canvas,
        px[color_indices],
        py[color_indices],
        sigma[color_positions],
        alpha[color_positions],
        dynamic_colors,
    )


# ─────────────────────────────────────────────────────────────
# Background and FFmpeg
# ─────────────────────────────────────────────────────────────

def load_background(region_selection=None):
    if region_selection is None:
        map_path = MAP_FILE
        sky_region = None
    else:
        map_path = region_selection.map_path
        sky_region = region_selection.region
    if not map_path.exists():
        raise FileNotFoundError(f"Missing ready map: {map_path}")
    with Image.open(map_path) as image:
        image = image.convert("RGB")
        if sky_region is not None and image.size != (
            sky_region.width,
            sky_region.height,
        ):
            raise ValueError(
                f"Region map dimensions {image.width} × {image.height} do not "
                f"match layout {sky_region.width} × {sky_region.height}"
            )
        if image.size != (VIDEO_W, VIDEO_H):
            console.print(
                f"  → Resizing sky map once: {image.width} × {image.height} "
                f"→ {VIDEO_W} × {VIDEO_H}"
            )
            image = image.resize(
                (VIDEO_W, VIDEO_H),
                resample=Image.Resampling.LANCZOS,
            )
        background = np.asarray(image, dtype=np.uint8).copy()
    console.print(
        f"  ✓ Sky map loaded: {map_path} "
        f"({background.shape[1]} × {background.shape[0]} px)"
    )
    return np.ascontiguousarray(background)


def build_ffmpeg_command(width, height):
    return build_hevc_command(
        VIDEO,
        width,
        height,
        VIDEO_CONFIG,
    )


def configure_region_artifact_paths(region_selection):
    """Give every selected region isolated videos, preview, and frame cache."""
    global VIDEO, HVC1_VIDEO, MOBILE_VIDEO, PNG_CACHE, OVERLAY_CACHE_DIR

    label = region_selection.map_path.stem.casefold()
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    prefix = f"aurora_sky_region_rect_pic1_{RESOLUTION_TAG}_"
    if label.startswith(prefix):
        label = label[len(prefix):]
    elif label == prefix.rstrip("_"):
        label = "default"
    label = label or "region"

    VIDEO = video_path(
        f"aurora_variable_{RESOLUTION_TAG}_region_{label}"
        f"{VARIABLE_MODE_SUFFIX}_animation.mp4"
    )
    HVC1_VIDEO = video_path(
        f"aurora_variable_{RESOLUTION_TAG}_region_{label}"
        f"{VARIABLE_MODE_SUFFIX}_hvc1.mp4"
    )
    MOBILE_VIDEO = video_path(
        f"aurora_variable_{RESOLUTION_TAG}_region_{label}"
        f"{VARIABLE_MODE_SUFFIX}_mobile.mp4"
    )
    PNG_CACHE, OVERLAY_CACHE_DIR = cache_artifact_paths(
        f"aurora_variable_{RESOLUTION_TAG}_region_{label}{VARIABLE_MODE_SUFFIX}",
        f"frames_variable_overlay_{RESOLUTION_TAG}_region_{label}"
        f"{VARIABLE_MODE_SUFFIX}",
        FRAME_CACHE_VERSION,
        VIDEO_CONFIG,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    start_time = time.perf_counter()
    console.section("VIDEO RENDER — Variable stars")
    console.detail(f"Resolution: {RESOLUTION_TAG} ({VIDEO_W} × {VIDEO_H} px)")
    console.detail(f"Frame rate: {FPS} fps")
    console.detail(f"Frames: {FRAMES:,}")
    console.detail(f"Duration: {FRAMES / FPS:.2f} s")
    console.detail(f"Sky-map mode: {SKY_MAP_MODE}")
    console.detail(f"Background map: {SKY_MAP_BACKGROUND}")
    console.detail(f"Variable-star mode: {VARIABLE_MODE}")
    console.detail(f"Simulated time span: {ANIMATION_DAYS:g} days")
    console.detail(f"Output directory: {VIDEOS_DIR}")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"  ✓ Video output directory ready: {VIDEOS_DIR}")
    region_selection = (
        select_sky_region(REGION_MAP_FILE, REGION_LAYOUT_FILE)
        if SKY_MAP_MODE == "region"
        else None
    )
    if region_selection is not None:
        configure_region_artifact_paths(region_selection)
    OVERLAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"  ✓ Cache directory ready: {OVERLAY_CACHE_DIR}")
    console.detail(f"Video file: {HVC1_VIDEO}")
    sky_region = region_selection.region if region_selection else None
    background = load_background(region_selection)
    img_h, img_w = background.shape[:2]

    (
        l,
        b,
        period,
        amplitude,
        phase,
        variable_types,
        teff,
        bp_rp,
        bp_rp_intrinsic,
        reddening_quality,
        extinction_flags,
        ag_gspphot,
        ebpminrp_gspphot,
        source_id,
        luminosity,
        radius,
        mass,
        classification_score,
    ) = load_variable_stars(sky_region)
    colour_teff, colour_bp_rp, colour_counts = prepare_animation_colour_inputs(
        teff,
        bp_rp_intrinsic,
        reddening_quality,
        bp_rp,
        variable_types,
        extinction_flags,
        ag_gspphot,
        ebpminrp_gspphot,
    )
    temperature_base = temperature_from_columns(colour_teff, colour_bp_rp)
    # Variable stars need a display palette with enough chroma to remain
    # visibly blue/yellow after alpha blending and 16K downsampling.
    vivid_temperature_colors = VIVID_TEMPERATURE_COLOURS
    base_colors = variable_star_colors_from_columns(
        colour_teff, colour_bp_rp
    )
    console.print(
        "  ✓ Colour inputs selected in priority order "
        f"(teff={colour_counts['teff']:,}, "
        f"intrinsic BP-RP={colour_counts['intrinsic']:,}, "
        f"raw BP-RP={colour_counts['raw']:,}, "
        f"class fallback={colour_counts['class_fallback']:,}, "
        f"extinction-protected={colour_counts['extinction_protected']:,})"
    )
    behavior = prepare_behavior_arrays(
        variable_types,
        period,
        amplitude,
        phase,
    )
    if np.any(np.isin(variable_types, tuple(HOT_CLASS_FALLBACK_TEMPERATURES))):
        sdor_count = int(np.count_nonzero(variable_types == "SDOR"))
        stochastic_count = int(
            np.count_nonzero(
                np.isin(
                    variable_types,
                    ("BE", "GCAS", "WR", "UNKNOWN_HOT"),
                )
            )
        )
        console.print(
            "  ✓ Hot-variable timing models: "
            f"SDOR periodic/strong={sdor_count:,}, "
            f"aperiodic random brightenings={stochastic_count:,}"
        )
    apply_reference_variable_star_size(behavior, VARIABLE_MODE)
    if VARIABLE_MODE in HOT_VARIABLE_MODES:
        physical_counts = apply_catalog_physical_scaling(
            behavior,
            luminosity,
            radius,
            mass,
            classification_score,
        )
        console.print(
            "  ✓ Applied bounded Gaia physical scaling "
            f"(luminosity={physical_counts['luminosity']:,}, "
            f"radius={physical_counts['radius']:,}, "
            f"mass={physical_counts['mass']:,}, "
            f"confidence={physical_counts['classification_score']:,})"
        )
    applied_size_scale = (
        ALL_CATALOGUES_MAP_SIZE_SCALE
        if VARIABLE_MODE == VARIABLE_MODE_STANDARD
        else VARIABLE_MAP_SIZE_SCALE
    )
    console.print(
        "  ✓ Applied density-aware variable-star size "
        f"(base scale={float(applied_size_scale):.3f}, "
        f"sigma range={VARIABLE_MIN_SIGMA_16K:g}–"
        f"{VARIABLE_MAX_SIGMA_16K:g} px at 16K, full-sky and region)"
    )
    console.print(
        "  ✓ Vivid temperature palette enabled for all variable stars "
        "(saturated red/yellow/blue hues)"
    )
    period = behavior["period"]
    amplitude = behavior["amplitude"]
    phase, valid_catalog_phase = prepare_simulation_phases(
        behavior["phase"],
        source_id,
    )
    console.print(
        f"  ✓ Prepared physical behaviors and base colours for "
        f"{len(np.unique(variable_types))} variability classes"
    )
    console.print(
        "  ✓ Independent deterministic pulsation phases prepared: "
        f"catalogue phase used for {np.count_nonzero(valid_catalog_phase):,}/"
        f"{len(phase):,} stars"
    )
    if behavior["missing_period_count"] or behavior["missing_amplitude_count"]:
        console.print(
            "  → Applied class fallbacks only to invalid catalogue values: "
            f"period={behavior['missing_period_count']:,}, "
            f"amplitude={behavior['missing_amplitude_count']:,}"
        )

    appearance_rng = np.random.default_rng(APPEARANCE_SEED)
    appearance_delay = appearance_rng.uniform(
        0.0,
        APPEARANCE_FRACTION,
        len(l),
    ).astype(np.float32)
    max_fade_delay = max(OUTRO_FRACTION - FADE_OUT_TIME, 0.0)
    fade_delay = appearance_rng.uniform(
        0.0,
        max_fade_delay,
        len(l),
    ).astype(np.float32)
    unit_1 = deterministic_uniform(source_id, CLASS_VARIATION_SEED)
    unit_2 = deterministic_uniform(source_id, IRREGULARITY_SEED)
    unit_3 = deterministic_uniform(source_id, EVENT_TIME_SEED)
    mode_phase_1 = (2.0 * np.pi * unit_1).astype(np.float32)
    mode_phase_2 = (2.0 * np.pi * unit_2).astype(np.float32)
    irregularity = (
        behavior["irregularity"]
        * (IRREGULARITY_SCALE_MIN + IRREGULARITY_SCALE_RANGE * unit_3)
    ).astype(np.float32)
    event_center = (
        EVENT_CENTER_MIN + EVENT_CENTER_RANGE * unit_3
    ).astype(np.float32)
    event_fraction = deterministic_uniform(
        source_id,
        EVENT_TIME_SEED ^ IRREGULARITY_SEED,
    )
    event_width = (
        behavior["event_width_min"]
        + event_fraction
        * (behavior["event_width_max"] - behavior["event_width_min"])
    ).astype(np.float32)
    event_shape = deterministic_uniform(
        source_id,
        EVENT_TIME_SEED ^ CLASS_VARIATION_SEED,
    )
    console.print("  ✓ Deterministic per-object modes and event timing prepared")

    if sky_region is None:
        px, py = galactic_to_hammer_pixel(l, b, img_w, img_h)
        geometry_signature = np.asarray([0.0], dtype=np.float64)
    else:
        px, py, inside_region = galactic_to_region_pixel(
            l,
            b,
            img_w,
            img_h,
            sky_region,
        )
        if not np.all(inside_region):
            raise RuntimeError(
                "Variable-star region filter and pixel projection disagree"
            )
        geometry_signature = sky_region.signature_array()
        console.print(
            f"  ✓ Variable stars inside video field: {len(px):,}"
        )
    frame_times, frame_opacity, active_progress = build_editorial_timeline(
        0.0,
        ANIMATION_DAYS,
        VIDEO_CONFIG,
    )
    console.print(
        f"  ✓ Video timeline: {VIDEO_CONFIG.pre_roll_seconds:g}s map + "
        f"{VIDEO_CONFIG.active_duration_seconds:g}s variables + "
        f"{VIDEO_CONFIG.post_roll_seconds:g}s map"
    )
    worker_args = (
        px,
        py,
        period,
        amplitude,
        phase,
        behavior["model_groups"],
        behavior["flux_mode"],
        behavior["size_scale"],
        behavior["size_response"],
        behavior["intensity_scale"],
        behavior["brightness_response"],
        temperature_base,
        behavior["temperature_response"],
        vivid_temperature_colors,
        irregularity,
        mode_phase_1,
        mode_phase_2,
        event_center,
        event_width,
        event_shape,
        appearance_delay,
        fade_delay,
        frame_times,
        active_progress,
        frame_opacity,
        img_h,
    )
    _init_worker(*worker_args)
    signature = frame_cache_signature(
        img_w,
        img_h,
        geometry_signature,
        px,
        py,
        period,
        amplitude,
        phase,
        behavior["model"],
        behavior["flux_mode"],
        behavior["size_scale"],
        behavior["size_response"],
        behavior["intensity_scale"],
        behavior["brightness_response"],
        temperature_base,
        behavior["temperature_response"],
        np.asarray([vivid_temperature_colors], dtype=np.uint8),
        irregularity,
        mode_phase_1,
        mode_phase_2,
        event_center,
        event_width,
        event_shape,
        appearance_delay,
        fade_delay,
        frame_times,
        active_progress,
        frame_opacity,
    )

    missing_frames = []
    if SAVE_DEBUG_FRAMES:
        for frame in range(FRAMES):
            MEMORY.throttle()
            path = _cache_path(frame)
            parameters = (
                load_frame_cache_safe(
                    path,
                    img_w,
                    img_h,
                    signature,
                    len(px),
                    verbose=False,
                )
                if path.exists()
                else None
            )
            if parameters is None:
                missing_frames.append(frame)
    else:
        missing_frames = list(range(FRAMES))
    console.print(f"  ✓ Frames to render: {len(missing_frames)}/{FRAMES}")

    if missing_frames:
        render_workers = 1 if MEMORY.throttle() else MAX_RENDER_WORKERS
        if render_workers > 1 and len(missing_frames) > 1:
            chunk_size = max(
                1,
                len(missing_frames) // (render_workers * 4),
            )
            with mp.Pool(
                processes=render_workers,
                initializer=_init_worker,
                initargs=worker_args,
            ) as pool:
                results = pool.imap(
                    render_frame_parameters,
                    missing_frames,
                    chunksize=chunk_size,
                )
                for frame, parameters in console.progress(
                    results,
                    total=len(missing_frames),
                    desc="Rendering frames",
                    unit="frame",
                ):
                    MEMORY.throttle()
                    if SAVE_DEBUG_FRAMES:
                        save_frame_cache_safe(
                            _cache_path(frame),
                            parameters,
                            img_w,
                            img_h,
                            signature,
                            len(px),
                            verbose=False,
                        )
        else:
            for frame in console.progress(
                missing_frames,
                desc="Rendering frames",
                unit="frame",
            ):
                MEMORY.throttle()
                _, parameters = render_frame_parameters(frame)
                if SAVE_DEBUG_FRAMES:
                    save_frame_cache_safe(
                        _cache_path(frame),
                        parameters,
                        img_w,
                        img_h,
                        signature,
                        len(px),
                        verbose=False,
                    )

    console.print("\n[AURORA] Preparing preview")
    console.print("─" * 45)
    if PNG_CACHE.exists():
        console.print(f"  ✓ Preview already exists: {PNG_CACHE}")
    else:
        preview_frame = FRAMES // 2
        parameters = (
            load_frame_cache_safe(
                _cache_path(preview_frame),
                img_w,
                img_h,
                signature,
                len(px),
                verbose=False,
            )
            if SAVE_DEBUG_FRAMES
            else render_frame_parameters(preview_frame)[1]
        )
        if parameters is None:
            parameters = render_frame_parameters(preview_frame)[1]
        preview = background.copy()
        draw_cached_frame(preview, parameters, px, py, base_colors)
        Image.fromarray(preview).save(PNG_CACHE, compress_level=3)
        console.print(f"  ✓ Preview saved: {PNG_CACHE}")

    console.print("\n[AURORA] Rendering animation")
    console.print("─" * 45)
    ffmpeg_process = subprocess.Popen(
        build_ffmpeg_command(img_w, img_h),
        stdin=subprocess.PIPE,
    )
    if ffmpeg_process.stdin is None:
        raise RuntimeError("FFmpeg stdin pipe could not be created")

    # Reuse one selected-resolution frame buffer instead of allocating a new
    # large array on every iteration.
    frame8 = np.empty_like(background)
    try:
        for frame in console.progress(range(FRAMES), desc="Encoding video", unit="frame"):
            MEMORY.throttle()
            parameters = (
                load_frame_cache_safe(
                    _cache_path(frame),
                    img_w,
                    img_h,
                    signature,
                    len(px),
                    verbose=False,
                )
                if SAVE_DEBUG_FRAMES
                else None
            )
            if parameters is None:
                console.print(
                    f"  ! Re-rendering frame {frame:04d} after "
                    "cache miss or corruption"
                )
                _, parameters = render_frame_parameters(frame)
                if SAVE_DEBUG_FRAMES:
                    save_frame_cache_safe(
                        _cache_path(frame),
                        parameters,
                        img_w,
                        img_h,
                        signature,
                        len(px),
                        verbose=False,
                    )
            np.copyto(frame8, background)
            draw_cached_frame(frame8, parameters, px, py, base_colors)
            ffmpeg_process.stdin.write(memoryview(frame8).cast("B"))
    except Exception:
        ffmpeg_process.stdin.close()
        ffmpeg_process.terminate()
        ffmpeg_process.wait()
        raise
    else:
        ffmpeg_process.stdin.close()
        return_code = ffmpeg_process.wait()

    if return_code != 0 or not VIDEO.exists() or VIDEO.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg encoding failed with exit code {return_code}")
    console.print(f"  ✓ MP4 created (raw {RESOLUTION_TAG} stream)")

    console.print("  → Fixing HEVC container compatibility (hvc1 tag, stream copy)")
    subprocess.run(
        build_hvc1_remux_command(VIDEO, HVC1_VIDEO),
        check=True,
    )
    console.print(f"  ✓ hvc1 version created: {HVC1_VIDEO}")

    if console.confirm("Create 8K mobile version"):
        console.print("  → Creating 8K mobile version (HEVC hvc1, 7680×4320)")
        subprocess.run(
            build_mobile_hevc_command(
                VIDEO,
                MOBILE_VIDEO,
                VIDEO_CONFIG,
            ),
            check=True,
        )
        console.print(f"  ✓ Mobile version created: {MOBILE_VIDEO}")

    console.print("  ✓ Output file verified")
    elapsed = time.perf_counter() - start_time
    console.complete("VIDEO RENDER — Variable stars")
    console.success(f"Output: {HVC1_VIDEO}")
    console.detail(
        f"Runtime: {elapsed / 3600.0:.2f} h "
        f"({elapsed / 60.0:.1f} min, {elapsed:.1f} s)"
    )
    '''
    console.warning("Shutting down the machine in 1 minute...")
    subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
    '''

if __name__ == "__main__":
    main()
