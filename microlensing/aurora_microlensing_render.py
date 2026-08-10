"""
Render a high-resolution AURORA gravitational-microlensing animation.

The expensive full-frame work is deliberately kept in the parent process.
Workers calculate only compact event parameters for a frame, so 16K
background images and overlays never cross process boundaries.
"""

from dataclasses import replace
import os
import re
import sys

# Keep worker processes from oversubscribing numerical-library threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
import subprocess
import time
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from astropy.table import Table
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_render_core import (
    array_signature,
    bp_rp_to_temperature,
    draw_gaussians_u8,
    galactic_to_hammer_pixel,
    galactic_to_region_pixel,
    galactic_region_mask,
    microlensing_visual_parameters,
    paczynski_amplification,
    temperature_to_rgb,
)
from core.aurora_region_selection import select_sky_region
from core.aurora_paths import (
    MAPS_DIR,
    VIDEOS_DIR,
    asset_path,
    map_path,
    region_map_path,
    video_path,
)
from core.aurora_resolution import (
    available_sky_map_backgrounds,
    get_resolution,
    prompt_sky_map_background,
    prompt_sky_map_mode,
    sky_map_background_suffix,
)
from core.aurora_video_core import (
    VIDEO_CONFIG as BASE_VIDEO_CONFIG,
    build_editorial_timeline as build_shared_editorial_timeline,
    build_hevc_command,
    build_hvc1_remux_command,
    build_mobile_hevc_command,
    cache_artifact_paths,
)

MEMORY = MemoryController.from_environment()


# ─────────────────────────────────────────────────────────────
# Program settings
# ─────────────────────────────────────────────────────────────

MICROLENS_FILE = asset_path("aurora_microlensing.fits")
MAX_RENDER_WORKERS_LIMIT = 6
MIN_VISIBLE_PEAK_EXCESS = 0.05
EVENT_TIME_MARGIN_MULTIPLIER = 1.0
EVENT_TIME_CAP_PERCENTILE = 95.0
FRAME_CACHE_VERSION = 9
BASE_STAR_SIGMA_16K = 2.0
BASE_STAR_ALPHA = 0.2
DEFAULT_LENS_MASS_SOLAR = 1.0

RESOLUTION = get_resolution(16)
RESOLUTION_TAG = RESOLUTION.tag
VIDEO_CONFIG = replace(
    BASE_VIDEO_CONFIG,
    width=RESOLUTION.width,
    height=RESOLUTION.height,
)
SKY_MAP_MODE = prompt_sky_map_mode()
AVAILABLE_SKY_MAP_BACKGROUNDS = available_sky_map_backgrounds(
    RESOLUTION,
    MAPS_DIR,
)
if SKY_MAP_MODE == "full" and not AVAILABLE_SKY_MAP_BACKGROUNDS:
    raise FileNotFoundError(
        f"No ready {RESOLUTION_TAG} full-sky maps found in {MAPS_DIR}; "
        "run main/aurora_sky_render.py first"
    )
SKY_MAP_BACKGROUND = (
    prompt_sky_map_background(
        available_backgrounds=AVAILABLE_SKY_MAP_BACKGROUNDS
    )
    if SKY_MAP_MODE == "full"
    else "plain"
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
ARTIFACT_SUFFIX = f"{FIELD_SUFFIX}{BACKGROUND_SUFFIX}"
VIDEO = video_path(f"aurora_microlensing{ARTIFACT_SUFFIX}_animation.mp4")
HVC1_VIDEO = video_path(
    f"aurora_microlensing_{RESOLUTION_TAG}{ARTIFACT_SUFFIX}_hvc1.mp4"
)
MOBILE_VIDEO = video_path(
    f"aurora_microlensing_8k{ARTIFACT_SUFFIX}_mobile.mp4"
)

FPS = VIDEO_CONFIG.fps
FRAMES = VIDEO_CONFIG.frame_count
VIDEO_W = VIDEO_CONFIG.width
VIDEO_H = VIDEO_CONFIG.height
MAX_RENDER_WORKERS = min(MAX_RENDER_WORKERS_LIMIT, os.cpu_count() or 1)

# Shared editorial timeline: fixed clean-map holds around a scalable activity
# interval.  The project-wide duration lives in core/aurora_video_core.py.
PRE_ROLL_SECONDS = VIDEO_CONFIG.pre_roll_seconds
POST_ROLL_SECONDS = VIDEO_CONFIG.post_roll_seconds

# Meaningful events define the astronomical time range.  This excludes fits
# whose maximum blended excess is visually negligible, while a capped one-tE
# margin lets the first curve grow after the pre-roll and the last one fade
# before the shared post-roll.
# Same debug/cache switch and console-message convention as the variable-star
# renderer. The cache contains sparse drawing parameters, not 16K canvases.
SAVE_DEBUG_FRAMES = VIDEO_CONFIG.save_debug_frames
PNG_CACHE, OVERLAY_CACHE_DIR = cache_artifact_paths(
    f"aurora_microlensing{ARTIFACT_SUFFIX}",
    f"frames_micro_overlay{ARTIFACT_SUFFIX}",
    FRAME_CACHE_VERSION,
    VIDEO_CONFIG,
)

# At rest an event should look like one of the stars already present on the
# 16K sky map.  Its full Paczynski amplification then grows from that point
# instead of making a large overlay suddenly appear in the middle of the
# light curve.
# ─────────────────────────────────────────────────────────────
# Small, validated, atomic frame cache
# ─────────────────────────────────────────────────────────────

def _cache_path(frame):
    return OVERLAY_CACHE_DIR / f"overlay_{frame:04d}.npz"


def load_frame_cache_safe(path, img_w, img_h, signature):
    """Load sparse frame parameters or remove an invalid/old cache file."""
    path = Path(path)
    required = {
        "version",
        "img_w",
        "img_h",
        "signature",
        "px",
        "py",
        "sigma",
        "alpha",
        "color",
    }

    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                raise KeyError("missing sparse-cache fields")
            if int(data["version"]) != FRAME_CACHE_VERSION:
                raise ValueError("unsupported cache version")
            if int(data["img_w"]) != img_w or int(data["img_h"]) != img_h:
                raise ValueError("cache resolution differs from the current map")
            if str(data["signature"].item()) != signature:
                raise ValueError("cache inputs differ from the current catalogue")

            arrays = tuple(
                np.asarray(data[name], dtype=np.float32)
                for name in ("px", "py", "sigma", "alpha", "color")
            )
            lengths = {len(array) for array in arrays}
            if len(lengths) != 1 or not all(
                np.all(np.isfinite(array)) for array in arrays
            ):
                raise ValueError("invalid sparse-cache arrays")
            if arrays[-1].ndim != 2 or arrays[-1].shape[1] != 3:
                raise ValueError("invalid sparse-cache RGB colours")
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
            console.print(
                f"  ! Could not remove corrupted overlay cache: "
                f"{path} ({remove_error})"
            )
        return None


def save_frame_cache_safe(path, params, img_w, img_h, signature):
    """Atomically save sparse parameters so partial archives are never visible."""
    path = Path(path)
    temp_path = path.with_name(f"{path.name}.tmp")
    px, py, sigma, alpha, color = params

    try:
        with temp_path.open("wb") as temp_file:
            np.savez_compressed(
                temp_file,
                version=np.int16(FRAME_CACHE_VERSION),
                img_w=np.int32(img_w),
                img_h=np.int32(img_h),
                signature=np.str_(signature),
                px=np.asarray(px, dtype=np.float32),
                py=np.asarray(py, dtype=np.float32),
                sigma=np.asarray(sigma, dtype=np.float32),
                alpha=np.asarray(alpha, dtype=np.float32),
                color=np.asarray(color, dtype=np.float32),
            )
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


# ─────────────────────────────────────────────────────────────
# Worker state: initialized once, tasks contain only a frame number
# ─────────────────────────────────────────────────────────────

_worker_data = None


def _init_worker(
    px,
    py,
    tmax,
    te,
    u0,
    mass,
    source_fraction,
    color,
    frame_times,
    frame_opacity,
    img_h,
):
    global _worker_data
    _worker_data = (
        px,
        py,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        color,
        frame_times,
        frame_opacity,
        img_h,
    )


def render_frame_parameters(frame):
    """Calculate all event spots for one frame without allocating a canvas."""
    (
        px,
        py,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        color,
        frame_times,
        frame_opacity,
        img_h,
    ) = _worker_data
    t = frame_times[frame]
    sigma, alpha = microlensing_visual_parameters(
        t,
        tmax,
        te,
        u0,
        mass,
        img_h,
        source_fraction=source_fraction,
        base_sigma_16k=BASE_STAR_SIGMA_16K,
        base_alpha=BASE_STAR_ALPHA,
    )
    alpha *= frame_opacity[frame]

    return frame, (
        px.astype(np.float32, copy=False),
        py.astype(np.float32, copy=False),
        sigma,
        alpha,
        color.astype(np.float32, copy=False),
    )


def frame_cache_signature(img_w, img_h, *arrays):
    """Fingerprint all inputs that influence sparse frame parameters."""
    config = np.asarray(
        [FRAME_CACHE_VERSION, FRAMES, img_w, img_h],
        dtype=np.int64,
    )
    return array_signature(config, *arrays)


# ─────────────────────────────────────────────────────────────
# Input preparation
# ─────────────────────────────────────────────────────────────

def _table_column_f32(table, name, default=np.nan):
    """Read an Astropy column while converting masked values to a default."""
    if name not in table.colnames:
        return np.full(len(table), default, dtype=np.float32)
    column = table[name]
    if hasattr(column, "filled"):
        column = column.filled(default)
    return np.asarray(column, dtype=np.float32)


def _microlensing_source_colors(
    teff,
    bp_rp,
    ebpminrp,
    abp,
    arp,
    level1_bp0,
    level1_rp0,
    level1_fs_bp,
    level1_fs_rp,
    level1_model_valid,
):
    """Recover intrinsic source colours with a documented fallback chain."""
    source_colour_valid = (
        level1_model_valid
        & np.isfinite(level1_bp0)
        & np.isfinite(level1_rp0)
        & np.isfinite(level1_fs_bp)
        & np.isfinite(level1_fs_rp)
        & (level1_fs_bp > 0.0)
        & (level1_fs_rp > 0.0)
    )
    source_bp_rp = np.full_like(bp_rp, np.nan, dtype=np.float32)
    source_bp_rp[source_colour_valid] = (
        level1_bp0[source_colour_valid]
        - level1_rp0[source_colour_valid]
        - 2.5
        * np.log10(
            level1_fs_bp[source_colour_valid]
            / level1_fs_rp[source_colour_valid]
        )
    )

    band_reddening = abp - arp
    band_reddening_valid = np.isfinite(band_reddening) & (
        band_reddening >= 0.0
    )
    catalog_reddening_valid = np.isfinite(ebpminrp) & (
        ebpminrp >= 0.0
    )
    reddening = np.where(
        band_reddening_valid,
        band_reddening,
        np.where(catalog_reddening_valid, ebpminrp, np.nan),
    ).astype(np.float32, copy=False)

    source_intrinsic = source_bp_rp - reddening
    catalog_intrinsic = bp_rp - reddening

    temperature = np.full_like(teff, np.nan, dtype=np.float32)
    use_source_intrinsic = np.isfinite(source_intrinsic)
    temperature[use_source_intrinsic] = bp_rp_to_temperature(
        source_intrinsic[use_source_intrinsic]
    )

    missing = ~np.isfinite(temperature)
    use_teff = missing & np.isfinite(teff)
    temperature[use_teff] = teff[use_teff]

    missing = ~np.isfinite(temperature)
    use_catalog_intrinsic = missing & np.isfinite(catalog_intrinsic)
    temperature[use_catalog_intrinsic] = bp_rp_to_temperature(
        catalog_intrinsic[use_catalog_intrinsic]
    )

    missing = ~np.isfinite(temperature)
    use_source_observed = missing & np.isfinite(source_bp_rp)
    temperature[use_source_observed] = bp_rp_to_temperature(
        source_bp_rp[use_source_observed]
    )

    missing = ~np.isfinite(temperature)
    use_catalog_observed = missing & np.isfinite(bp_rp)
    temperature[use_catalog_observed] = bp_rp_to_temperature(
        bp_rp[use_catalog_observed]
    )

    solar_fallback = ~np.isfinite(temperature)
    temperature[solar_fallback] = 5772.0
    colors = temperature_to_rgb(temperature)
    statistics = {
        "teff": np.count_nonzero(use_teff),
        "source_intrinsic": np.count_nonzero(use_source_intrinsic),
        "catalog_intrinsic": np.count_nonzero(use_catalog_intrinsic),
        "source_observed": np.count_nonzero(use_source_observed),
        "catalog_observed": np.count_nonzero(use_catalog_observed),
        "solar": np.count_nonzero(solar_fallback),
        "source_colour": np.count_nonzero(source_colour_valid),
        "reddening": np.count_nonzero(np.isfinite(reddening)),
    }
    return colors, statistics


def load_events(sky_region=None):
    console.print("\n[AURORA] Loading microlensing catalogue")
    console.print("─" * 45)
    console.print(f"  → Loading FITS: {MICROLENS_FILE}")

    required = [
        "l",
        "b",
        "paczynski0_tmax",
        "paczynski0_te",
        "paczynski0_u0",
        "teff_gspphot",
        "bp_rp",
    ]
    table = Table.read(MICROLENS_FILE)
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise KeyError(f"Missing FITS columns: {', '.join(missing)}")
    console.print(f"  ✓ FITS loaded: {len(table):,} objects")

    # Reading directly from the Astropy table avoids a full pandas copy.
    l = _table_column_f32(table, "l")
    b = _table_column_f32(table, "b")
    tmax0 = _table_column_f32(table, "paczynski0_tmax")
    te0 = _table_column_f32(table, "paczynski0_te")
    u00 = _table_column_f32(table, "paczynski0_u0")
    tmax1 = _table_column_f32(table, "paczynski1_tmax")
    te1 = _table_column_f32(table, "paczynski1_te")
    u01 = _table_column_f32(table, "paczynski1_u0")
    fs_g1 = _table_column_f32(table, "paczynski1_fs_g")
    bp01 = _table_column_f32(table, "paczynski1_bp0")
    rp01 = _table_column_f32(table, "paczynski1_rp0")
    fs_bp1 = _table_column_f32(table, "paczynski1_fs_bp")
    fs_rp1 = _table_column_f32(table, "paczynski1_fs_rp")
    teff = _table_column_f32(table, "teff_gspphot")
    bp_rp = _table_column_f32(table, "bp_rp")
    ebpminrp = _table_column_f32(table, "ebpminrp_gspphot")
    abp = _table_column_f32(table, "abp_gspphot")
    arp = _table_column_f32(table, "arp_gspphot")

    level1_valid = (
        np.isfinite(tmax1)
        & np.isfinite(te1)
        & np.isfinite(u01)
        & np.isfinite(fs_g1)
        & (te1 > 0.0)
        & (fs_g1 > 0.0)
    )
    tmax = np.where(level1_valid, tmax1, tmax0).astype(
        np.float32,
        copy=False,
    )
    te = np.where(level1_valid, te1, te0).astype(np.float32, copy=False)
    u0 = np.where(level1_valid, u01, u00).astype(np.float32, copy=False)
    source_fraction = np.where(
        level1_valid,
        np.clip(fs_g1, 0.0, 1.0),
        1.0,
    ).astype(np.float32, copy=False)

    mass = np.ones(len(table), dtype=np.float32)
    for column in ("paczynski0_mass", "paczynski_mass", "lens_mass", "mass"):
        if column in table.colnames:
            mass = _table_column_f32(
                table,
                column,
                default=DEFAULT_LENS_MASS_SOLAR,
            )
            console.print(f"  ✓ Lens mass column: {column}")
            break
    good = (
        np.isfinite(l)
        & np.isfinite(b)
        & np.isfinite(tmax)
        & np.isfinite(te)
        & np.isfinite(u0)
        & np.isfinite(mass)
        & (te > 0.0)
    )
    if sky_region is not None:
        region_mask = galactic_region_mask(l, b, sky_region)
        good &= region_mask
        console.print(
            f"  ✓ Catalogue events inside regional field: "
            f"{np.count_nonzero(good):,}"
        )
    (
        l,
        b,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        teff,
        bp_rp,
        ebpminrp,
        abp,
        arp,
        bp01,
        rp01,
        fs_bp1,
        fs_rp1,
        level1_valid,
    ) = (
        array[good]
        for array in (
            l,
            b,
            tmax,
            te,
            u0,
            mass,
            source_fraction,
            teff,
            bp_rp,
            ebpminrp,
            abp,
            arp,
            bp01,
            rp01,
            fs_bp1,
            fs_rp1,
            level1_valid,
        )
    )
    del table
    if not len(tmax):
        scope = " inside the configured sky region" if sky_region else ""
        raise RuntimeError(
            "No valid microlensing events remain after filtering" + scope
        )

    color, color_statistics = _microlensing_source_colors(
        teff,
        bp_rp,
        ebpminrp,
        abp,
        arp,
        bp01,
        rp01,
        fs_bp1,
        fs_rp1,
        level1_valid,
    )

    l = np.radians(l).astype(np.float32)
    b = np.radians(b).astype(np.float32)
    l = -np.where(l > np.pi, l - 2.0 * np.pi, l).astype(np.float32)

    start_mjd = float(np.min(tmax))
    end_mjd = float(np.max(tmax))
    tmax = (tmax - start_mjd).astype(np.float32)
    console.print(f"  ✓ Valid events: {len(l):,}")
    console.print(
        f"  ✓ Level-1 blended Paczynski model: "
        f"{np.count_nonzero(level1_valid):,}/{len(l):,}"
    )
    console.print("  → Temperature and colour statistics")
    console.print(
        f"    Gaia teff_gspphot: {color_statistics['teff']:,}"
    )
    console.print(
        f"    blend-separated BP-RP available: "
        f"{color_statistics['source_colour']:,}"
    )
    console.print(
        f"    extinction/reddening available: "
        f"{color_statistics['reddening']:,}"
    )
    console.print(
        f"    dereddened BP-RP estimates: "
        f"{color_statistics['source_intrinsic'] + color_statistics['catalog_intrinsic']:,}"
    )
    console.print(
        f"    observed BP-RP estimates: "
        f"{color_statistics['source_observed'] + color_statistics['catalog_observed']:,}"
    )
    console.print(
        f"    solar-temperature fallback: {color_statistics['solar']:,}"
    )
    console.print(f"  ✓ Gaia time range: {start_mjd:g}–{end_mjd:g}")
    console.print("=== AURORA MICROLENSING CATALOG READY ===")
    return (
        l,
        b,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        color,
        start_mjd,
        end_mjd,
    )


def build_editorial_timeline(tmax, te, u0, source_fraction):
    """Build a fixed 3 s map + 35 s events + 2 s map timeline."""
    peak_amplification = paczynski_amplification(
        tmax,
        tmax,
        te,
        u0,
    )
    peak_excess = source_fraction * np.clip(
        peak_amplification - 1.0,
        0.0,
        None,
    )
    meaningful = (
        np.isfinite(peak_excess)
        & (peak_excess >= MIN_VISIBLE_PEAK_EXCESS)
    )
    if not np.any(meaningful):
        meaningful = np.ones_like(tmax, dtype=bool)

    duration_cap = float(
        np.percentile(te[meaningful], EVENT_TIME_CAP_PERCENTILE)
    )
    event_margin = (
        np.minimum(te[meaningful], duration_cap)
        * EVENT_TIME_MARGIN_MULTIPLIER
    )
    activity_start = float(
        np.min(tmax[meaningful] - event_margin)
    )
    activity_end = float(
        np.max(tmax[meaningful] + event_margin)
    )
    if not activity_end > activity_start:
        raise RuntimeError("Microlensing activity timeline has zero duration")

    frame_times, frame_opacity, _ = build_shared_editorial_timeline(
        activity_start,
        activity_end,
        VIDEO_CONFIG,
    )

    return (
        frame_times,
        frame_opacity.astype(np.float32, copy=False),
        activity_start,
        activity_end,
        int(np.count_nonzero(meaningful)),
    )


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
                f"  → Resizing sky map once: "
                f"{image.width} × {image.height} → {VIDEO_W} × {VIDEO_H}"
            )
            image = image.resize(
                (VIDEO_W, VIDEO_H),
                resample=Image.Resampling.LANCZOS,
            )
        bg = np.asarray(image, dtype=np.uint8).copy()

    img_h, img_w = bg.shape[:2]
    console.print(f"  ✓ Sky map loaded: {map_path} ({img_w} × {img_h} px)")
    pad_h = img_h & 1
    pad_w = img_w & 1
    if pad_h or pad_w:
        console.print("  → Padding canvas to even dimensions")
        bg = np.pad(bg, ((0, pad_h), (0, pad_w), (0, 0)))
    return np.ascontiguousarray(bg)


def build_ffmpeg_command(img_w, img_h):
    return build_hevc_command(
        VIDEO,
        img_w,
        img_h,
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
        f"aurora_microlensing_{RESOLUTION_TAG}_region_{label}_animation.mp4"
    )
    HVC1_VIDEO = video_path(
        f"aurora_microlensing_{RESOLUTION_TAG}_region_{label}_hvc1.mp4"
    )
    MOBILE_VIDEO = video_path(
        f"aurora_microlensing_8k_region_{label}_mobile.mp4"
    )
    PNG_CACHE, OVERLAY_CACHE_DIR = cache_artifact_paths(
        f"aurora_microlensing_{RESOLUTION_TAG}_region_{label}",
        f"frames_micro_overlay_{RESOLUTION_TAG}_region_{label}",
        FRAME_CACHE_VERSION,
        VIDEO_CONFIG,
    )


# ─────────────────────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────────────────────

def main():
    start_time = time.perf_counter()
    console.section("VIDEO RENDER — Gravitational microlensing")
    console.detail(f"Resolution: {RESOLUTION_TAG} ({VIDEO_W} × {VIDEO_H} px)")
    console.detail(f"Frame rate: {FPS} fps")
    console.detail(f"Frames: {FRAMES:,}")
    console.detail(f"Duration: {FRAMES / FPS:.2f} s")
    console.detail(f"Sky-map mode: {SKY_MAP_MODE}")
    console.detail(f"Background map: {SKY_MAP_BACKGROUND}")
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
    bg = load_background(region_selection)
    img_h, img_w = bg.shape[:2]

    (
        l,
        b,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        color,
        start_mjd,
        end_mjd,
    ) = load_events(sky_region)
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
                "Microlensing region filter and pixel projection disagree"
            )
        geometry_signature = sky_region.signature_array()
        console.print(f"  ✓ Events inside video field: {len(px):,}")

    (
        frame_times,
        frame_opacity,
        activity_start,
        activity_end,
        meaningful_events,
    ) = build_editorial_timeline(
        tmax,
        te,
        u0,
        source_fraction,
    )
    console.print(
        f"  ✓ Video timeline: {PRE_ROLL_SECONDS:g}s map + "
        f"{FRAMES / FPS - PRE_ROLL_SECONDS - POST_ROLL_SECONDS:g}s events "
        f"+ {POST_ROLL_SECONDS:g}s map"
    )
    console.print(
        f"  ✓ Activity range: {activity_start + start_mjd:g}–"
        f"{activity_end + start_mjd:g} MJD "
        f"({meaningful_events}/{len(tmax)} visible events set the range)"
    )
    worker_args = (
        px,
        py,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        color,
        frame_times,
        frame_opacity,
        img_h,
    )
    _init_worker(*worker_args)
    cache_signature = frame_cache_signature(
        img_w,
        img_h,
        geometry_signature,
        px,
        py,
        tmax,
        te,
        u0,
        mass,
        source_fraction,
        color,
        frame_times,
        frame_opacity,
    )
    console.print("  ✓ Paczynski parameters prepared")

    valid_cache = {}
    missing_frames = []
    saved_cache_frames = 0
    if SAVE_DEBUG_FRAMES:
        for frame in range(FRAMES):
            MEMORY.throttle()
            path = _cache_path(frame)
            params = (
                load_frame_cache_safe(
                    path,
                    img_w,
                    img_h,
                    cache_signature,
                )
                if path.exists()
                else None
            )
            if params is None:
                missing_frames.append(frame)
            else:
                valid_cache[frame] = params
    else:
        missing_frames = list(range(FRAMES))
    console.print(f"  ✓ Frames to render: {len(missing_frames)}/{FRAMES}")

    if missing_frames:
        render_workers = 1 if MEMORY.throttle() else MAX_RENDER_WORKERS
        if render_workers > 1 and len(missing_frames) > 1:
            chunk_size = max(1, len(missing_frames) // (render_workers * 4))
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
                for frame, params in console.progress(
                    results,
                    total=len(missing_frames),
                    description="Frames",
                    unit="frame",
                ):
                    MEMORY.throttle()
                    valid_cache[frame] = params
                    if SAVE_DEBUG_FRAMES:
                        save_frame_cache_safe(
                            _cache_path(frame),
                            params,
                            img_w,
                            img_h,
                            cache_signature,
                        )
                        saved_cache_frames += 1
        else:
            for frame in console.progress(
                missing_frames,
                description="Frames",
                unit="frame",
            ):
                MEMORY.throttle()
                _, params = render_frame_parameters(frame)
                valid_cache[frame] = params
                if SAVE_DEBUG_FRAMES:
                    save_frame_cache_safe(
                        _cache_path(frame),
                        params,
                        img_w,
                        img_h,
                        cache_signature,
                    )
                    saved_cache_frames += 1

    if saved_cache_frames:
        console.print(f"  ✓ Saved overlay caches: {saved_cache_frames}")

    console.print("\n[AURORA] Preparing preview")
    console.print("─" * 45)
    if PNG_CACHE.exists():
        console.print(f"  ✓ Preview already exists: {PNG_CACHE}")
    else:
        preview_frame = FRAMES // 2
        preview = bg.copy()
        draw_gaussians_u8(preview, *valid_cache[preview_frame])
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

    # Reuse one fixed-16K frame buffer. Allocating and freeing ~384 MiB for
    # every frame creates avoidable allocator and memory-pressure overhead.
    frame8 = np.empty_like(bg)
    draw_seconds = 0.0
    pipe_seconds = 0.0
    encoding_progress = console.progress(
        range(FRAMES),
        description="Encoding",
        unit="frame",
    )
    try:
        for frame in encoding_progress:
            MEMORY.throttle()
            params = valid_cache.get(frame)
            if params is None and SAVE_DEBUG_FRAMES:
                params = load_frame_cache_safe(
                    _cache_path(frame),
                    img_w,
                    img_h,
                    cache_signature,
                )
            if params is None:
                console.print(
                    f"  ! Re-rendering frame {frame:04d} after "
                    "cache miss or corruption"
                )
                _, params = render_frame_parameters(frame)
                if SAVE_DEBUG_FRAMES:
                    save_frame_cache_safe(
                        _cache_path(frame),
                        params,
                        img_w,
                        img_h,
                        cache_signature,
                    )

            draw_t0 = time.perf_counter()
            np.copyto(frame8, bg)
            draw_gaussians_u8(frame8, *params)
            draw_seconds += time.perf_counter() - draw_t0
            pipe_t0 = time.perf_counter()
            ffmpeg_process.stdin.write(memoryview(frame8).cast("B"))
            pipe_seconds += time.perf_counter() - pipe_t0
    except Exception:
        encoding_progress.close()
        console.print()
        ffmpeg_process.stdin.close()
        ffmpeg_process.terminate()
        ffmpeg_process.wait()
        raise
    else:
        encoding_progress.close()
        ffmpeg_process.stdin.close()
        return_code = ffmpeg_process.wait()

    if return_code != 0 or not VIDEO.exists() or VIDEO.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg encoding failed with exit code {return_code}")
    console.print(
        f"  ✓ Frame preparation: {draw_seconds:.1f}s; "
        f"pipe wait: {pipe_seconds:.1f}s; "
        "remaining time was spent inside ffmpeg"
    )
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
    console.complete("VIDEO RENDER — Gravitational microlensing")
    console.success(f"Output: {HVC1_VIDEO}")
    console.detail(
        f"Runtime: {elapsed / 3600.0:.2f} h "
        f"({elapsed / 60.0:.1f} min, {elapsed:.1f} s)"
    )


if __name__ == "__main__":
    main()
