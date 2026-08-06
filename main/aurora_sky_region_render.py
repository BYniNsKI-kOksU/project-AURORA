"""
Render a rectangular, projection-free AURORA map of a selected sky region.

The stellar flux, temperature colour, PSF and tone mapping intentionally match
aurora_sky_render.py. Galactic longitude and latitude are mapped directly to
the output rectangle; no Hammer or other cartographic projection is applied.
"""

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_constellations import (
    CONSTELLATION_INDEX_FILE,
    CONSTELLATIONS_SOURCE_FILE,
    DEFAULT_CONSTELLATION_MODE,
    load_constellation_region_polylines,
    prompt_constellation_overlay,
)
from core.aurora_sky_guides import (
    DEFAULT_GUIDE_MODE,
    SKY_GUIDES_SOURCE_FILE,
    add_reference_overlays_to_png,
    load_region_guide_layers,
    prompt_coordinate_grid_overlay,
    prompt_poland_limits_overlay,
)
from core.aurora_render_core import (
    accumulate_sorted_histograms,
    apply_psf,
    combined_psf_kernel,
    find_fits_table_hdu,
    sample_finite_column_windows,
    temperature_from_columns,
    temperature_to_rgb_channels,
)
from core.aurora_paths import (
    REGION_MAPS_DIR,
    asset_path,
    map_cache_path,
    region_map_path,
)
from core.aurora_resolution import (
    prompt_resolution,
    sky_map_background_from_flags,
    sky_map_background_suffix,
)


# ─────────────────────────────────────────────────────────────
# Selected sky region
# ─────────────────────────────────────────────────────────────

# Centre and angular size of the rectangular Galactic-coordinate field.
# Longitude may use either the [-180, 180] or [0, 360) convention.
REGION_L_CENTER_DEG = 3.23
REGION_B_CENTER_DEG = 0.59
REGION_L_WIDTH_DEG = 112.70
REGION_B_HEIGHT_DEG = 56.38

RESOLUTION = prompt_resolution()
RESOLUTION_TAG = RESOLUTION.tag

# Keep the pixel aspect ratio equal to the angular aspect ratio to avoid
# stretching. The selected profile controls only the raster dimensions.
OUTPUT_W = RESOLUTION.width
OUTPUT_H = RESOLUTION.height

INPUT_FILE = asset_path("aurora_gaia_catalog_900m.fits")
MAP_FILE = region_map_path(RESOLUTION.region_map_name)
CACHE_DIR = map_cache_path("regions", "default", RESOLUTION_TAG)
LAYOUT_FILE = region_map_path(RESOLUTION.region_layout_name)
RGB_CACHE = CACHE_DIR / RESOLUTION.region_rgb_cache_name
REGION_PROFILE_NAME = "default"

CATALOG_CHUNK_ROWS = 1_000_000
TONE_MAP_TILE_ROWS = 128
PERCENTILE_SAMPLE_SIZE = 2_000_000
PNG_BIT_DEPTH = 8
PNG_COMPRESSION = 6
SAVE_DEBUG_FRAMES = True
DEBUG_MAX_DIM = 4096
MEMORY = MemoryController.from_environment()


# ─────────────────────────────────────────────────────────────
# Configuration and diagnostics
# ─────────────────────────────────────────────────────────────

def _normalize_region_profile_name(value):
    text = str(value or "").strip().casefold()
    if not text or text in {"default", "rect_pic1"}:
        return "default"
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        raise ValueError("Region name must contain at least one letter or digit")
    return text


def configure_region_output_paths(profile_name):
    """Select isolated output and cache paths for one named sky region."""
    global REGION_PROFILE_NAME
    global MAP_FILE, CACHE_DIR, LAYOUT_FILE, RGB_CACHE

    REGION_PROFILE_NAME = _normalize_region_profile_name(profile_name)
    if REGION_PROFILE_NAME == "default":
        MAP_FILE = region_map_path(RESOLUTION.region_map_name)
        CACHE_DIR = map_cache_path("regions", "default", RESOLUTION_TAG)
        LAYOUT_FILE = region_map_path(RESOLUTION.region_layout_name)
        RGB_CACHE = CACHE_DIR / RESOLUTION.region_rgb_cache_name
        return

    base_name = (
        f"aurora_sky_region_rect_pic1_{RESOLUTION_TAG}_"
        f"{REGION_PROFILE_NAME}"
    )
    MAP_FILE = region_map_path(f"{base_name}.png")
    CACHE_DIR = map_cache_path(
        "regions", REGION_PROFILE_NAME, RESOLUTION_TAG
    )
    LAYOUT_FILE = region_map_path(f"{base_name}_layout.npz")
    RGB_CACHE = CACHE_DIR / (
        f"aurora_sky_region_rgb_{RESOLUTION_TAG}_{REGION_PROFILE_NAME}.npy"
    )


def resolve_region_profile_name(argument_value):
    if argument_value is not None:
        return _normalize_region_profile_name(argument_value)
    if not sys.stdin.isatty():
        return "default"
    response = console.prompt(
        "Region profile name (for example galactic_center or lmc; "
        "Enter = default)"
    )
    return _normalize_region_profile_name(response)


def configure_region_geometry(args):
    """Apply optional command-line geometry without editing source constants."""
    global REGION_L_CENTER_DEG, REGION_B_CENTER_DEG
    global REGION_L_WIDTH_DEG, REGION_B_HEIGHT_DEG

    if args.l_center is not None:
        REGION_L_CENTER_DEG = float(args.l_center)
    if args.b_center is not None:
        REGION_B_CENTER_DEG = float(args.b_center)
    if args.l_width is not None:
        REGION_L_WIDTH_DEG = float(args.l_width)
    if args.b_height is not None:
        REGION_B_HEIGHT_DEG = float(args.b_height)


def validate_configuration():
    if not np.isfinite(
        [
            REGION_L_CENTER_DEG,
            REGION_B_CENTER_DEG,
            REGION_L_WIDTH_DEG,
            REGION_B_HEIGHT_DEG,
        ]
    ).all():
        raise ValueError("Region coordinates and dimensions must be finite")
    if not 0.0 < REGION_L_WIDTH_DEG <= 360.0:
        raise ValueError("REGION_L_WIDTH_DEG must be in the range (0, 360]")
    if not 0.0 < REGION_B_HEIGHT_DEG <= 180.0:
        raise ValueError("REGION_B_HEIGHT_DEG must be in the range (0, 180]")
    if not -90.0 <= REGION_B_CENTER_DEG <= 90.0:
        raise ValueError("REGION_B_CENTER_DEG must be in the range [-90, 90]")

    b_min = REGION_B_CENTER_DEG - REGION_B_HEIGHT_DEG * 0.5
    b_max = REGION_B_CENTER_DEG + REGION_B_HEIGHT_DEG * 0.5
    if b_min < -90.0 or b_max > 90.0:
        raise ValueError("Selected latitude range extends beyond [-90, 90]")
    if OUTPUT_W < 1 or OUTPUT_H < 1:
        raise ValueError("OUTPUT_W and OUTPUT_H must be positive")

    angular_aspect = REGION_L_WIDTH_DEG / REGION_B_HEIGHT_DEG
    pixel_aspect = OUTPUT_W / OUTPUT_H
    if not np.isclose(pixel_aspect, angular_aspect, rtol=0.01):
        console.print(
            "  ! Output aspect ratio differs from the angular field; "
            "the map will be stretched"
        )


def _save_debug_npz(path, **arrays):
    if not SAVE_DEBUG_FRAMES:
        return
    debug_arrays = {}
    for name, array in arrays.items():
        array = np.asarray(array)
        if array.ndim >= 2:
            step = max(1, max(array.shape[:2]) // DEBUG_MAX_DIM)
            array = array[::step, ::step]
        debug_arrays[name] = array
    np.savez_compressed(path, **debug_arrays)
    console.print(f"  ✓ Debug cache saved: {path}")


def _layout_values():
    return {
        "projection": np.array(["rectangular"]),
        "dimensions": np.array([OUTPUT_W, OUTPUT_H], dtype=np.int32),
        "l_center_deg": np.float64(REGION_L_CENTER_DEG),
        "b_center_deg": np.float64(REGION_B_CENTER_DEG),
        "l_width_deg": np.float64(REGION_L_WIDTH_DEG),
        "b_height_deg": np.float64(REGION_B_HEIGHT_DEG),
    }


def _layout_storage_values():
    return {
        **_layout_values(),
        "region_profile": np.array([REGION_PROFILE_NAME]),
        "map_filename": np.array([MAP_FILE.name]),
    }


def _layout_matches_configuration():
    if not LAYOUT_FILE.exists():
        return False
    expected = _layout_values()
    try:
        with np.load(LAYOUT_FILE, allow_pickle=False) as layout:
            if set(expected) - set(layout.files):
                return False
            return all(
                np.array_equal(layout[name], value)
                for name, value in expected.items()
            )
    except (OSError, ValueError):
        return False


def _png_matches_output_format(path):
    try:
        with Path(path).open("rb") as png:
            header = png.read(26)
    except OSError:
        return False
    return (
        len(header) == 26
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR"
        and int.from_bytes(header[16:20], "big") == OUTPUT_W
        and int.from_bytes(header[20:24], "big") == OUTPUT_H
        and header[24] == PNG_BIT_DEPTH
        and header[25] == 2
    )


def _reference_map_is_current(
    output_path,
    *,
    include_constellations,
    include_coordinate_grid,
    include_poland_limits,
):
    if not _png_matches_output_format(output_path):
        return False
    try:
        dependencies = [MAP_FILE, LAYOUT_FILE]
        if include_constellations:
            dependencies.extend(
                (CONSTELLATION_INDEX_FILE, CONSTELLATIONS_SOURCE_FILE)
            )
        if include_coordinate_grid or include_poland_limits:
            dependencies.append(SKY_GUIDES_SOURCE_FILE)
        output_time = Path(output_path).stat().st_mtime_ns
        return all(
            output_time >= path.stat().st_mtime_ns
            for path in dependencies
        )
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────
# Projection-free rectangular histogram
# ─────────────────────────────────────────────────────────────

def build_region_histograms():
    console.print("\n[AURORA] Building rectangular Gaia histograms")
    console.print("─" * 45)
    console.print(f"  → Opening memory-mapped FITS: {INPUT_FILE}")

    with fits.open(INPUT_FILE, memmap=True, lazy_load_hdus=True) as hdul:
        table_hdu = find_fits_table_hdu(hdul)
        data = table_hdu.data
        names = {name.lower(): name for name in data.names}
        required = {"l", "b", "phot_g_mean_mag"}
        missing = required - names.keys()
        if missing:
            raise KeyError(f"Missing FITS columns: {', '.join(sorted(missing))}")

        row_count = len(data)
        console.print(f"  ✓ FITS table ready: {row_count:,} records")

        mag_column = data[names["phot_g_mean_mag"]]
        console.print(
            "  → Sampling magnitudes for bright-source clipping "
            f"({PERCENTILE_SAMPLE_SIZE:,} rows max)"
        )
        sample_t0 = time.perf_counter()
        mag_sample = sample_finite_column_windows(
            mag_column,
            row_count,
            PERCENTILE_SAMPLE_SIZE,
        )
        if not len(mag_sample):
            raise RuntimeError("No finite Gaia magnitudes found")
        bright_mag = np.percentile(mag_sample, 0.1)
        flux_max = np.float32(10.0 ** (-0.4 * bright_mag))
        console.print(
            f"  ✓ Magnitude sample ready: {len(mag_sample):,} values "
            f"in {time.perf_counter() - sample_t0:.1f}s"
        )
        del mag_sample

        shape = (OUTPUT_H, OUTPUT_W)
        gib = np.prod(shape) * np.dtype(np.float32).itemsize * 3 / 1024**3
        console.print(f"  → Allocating histogram buffers: {gib:.2f} GiB")
        hist_flux = np.zeros(shape, dtype=np.float32)
        hist_temp_flux = np.zeros(shape, dtype=np.float32)
        hist_count = np.zeros(shape, dtype=np.float32)
        histograms = (hist_flux, hist_temp_flux, hist_count)
        console.print("  ✓ Histogram buffers ready")

        teff_name = names.get("teff_gspphot")
        bp_rp_name = names.get("bp_rp")
        if teff_name is None and bp_rp_name is None:
            raise KeyError("Neither teff_gspphot nor bp_rp exists in FITS")

        half_l = REGION_L_WIDTH_DEG * 0.5
        b_min = REGION_B_CENTER_DEG - REGION_B_HEIGHT_DEG * 0.5
        b_max = REGION_B_CENTER_DEG + REGION_B_HEIGHT_DEG * 0.5

        selected_total = 0
        teff_total = 0
        bp_rp_total = 0
        missing_temperature_total = 0
        ranges = range(0, row_count, CATALOG_CHUNK_ROWS)
        iterator = console.progress(
            ranges,
            total=(row_count + CATALOG_CHUNK_ROWS - 1)
            // CATALOG_CHUNK_ROWS,
            description="Catalogue",
            unit="chunk",
        )
        for start in iterator:
            MEMORY.throttle()
            stop = min(start + CATALOG_CHUNK_ROWS, row_count)
            lon_deg = np.asarray(
                data[names["l"]][start:stop],
                dtype=np.float32,
            )
            lat_deg = np.asarray(
                data[names["b"]][start:stop],
                dtype=np.float32,
            )
            magnitude = np.asarray(
                mag_column[start:stop],
                dtype=np.float32,
            )
            teff = (
                np.asarray(data[teff_name][start:stop], dtype=np.float32)
                if teff_name
                else np.full(stop - start, np.nan, dtype=np.float32)
            )
            bp_rp = (
                np.asarray(data[bp_rp_name][start:stop], dtype=np.float32)
                if bp_rp_name
                else np.full(stop - start, np.nan, dtype=np.float32)
            )

            delta_l = (
                lon_deg - REGION_L_CENTER_DEG + 180.0
            ) % 360.0 - 180.0
            selected = (
                np.isfinite(lon_deg)
                & np.isfinite(lat_deg)
                & np.isfinite(magnitude)
                & (np.abs(delta_l) <= half_l)
                & (lat_deg >= b_min)
                & (lat_deg <= b_max)
            )
            if not np.any(selected):
                continue

            delta_l = delta_l[selected]
            lat_deg = lat_deg[selected]
            magnitude = magnitude[selected]
            teff = teff[selected]
            bp_rp = bp_rp[selected]

            selected_total += len(delta_l)
            teff_total += np.count_nonzero(np.isfinite(teff))
            bp_rp_total += np.count_nonzero(np.isfinite(bp_rp))
            missing_temperature_total += np.count_nonzero(
                ~np.isfinite(teff) & ~np.isfinite(bp_rp)
            )

            # Longitude decreases from left to right, matching the orientation
            # of aurora_sky_render.py. Latitude decreases from top to bottom.
            x_index = np.floor(
                (half_l - delta_l) * (OUTPUT_W / REGION_L_WIDTH_DEG)
            ).astype(np.int64)
            y_index = np.floor(
                (b_max - lat_deg) * (OUTPUT_H / REGION_B_HEIGHT_DEG)
            ).astype(np.int64)
            np.clip(x_index, 0, OUTPUT_W - 1, out=x_index)
            np.clip(y_index, 0, OUTPUT_H - 1, out=y_index)
            flat_index = y_index * OUTPUT_W + x_index

            flux = np.power(
                np.float32(10.0),
                np.float32(-0.4) * magnitude,
            ).astype(np.float32, copy=False)
            np.minimum(flux, flux_max, out=flux)
            temperature = temperature_from_columns(teff, bp_rp)
            accumulate_sorted_histograms(
                histograms,
                flat_index,
                flux,
                temperature,
            )

        console.print(f"  ✓ Selected catalogue rows: {selected_total:,}")
        console.print(f"  ✓ teff_gspphot available: {teff_total:,}")
        console.print(f"  ✓ bp_rp available: {bp_rp_total:,}")
        console.print(
            f"  ✓ No temperature or colour data: "
            f"{missing_temperature_total:,}"
        )

    if selected_total == 0:
        raise RuntimeError("No catalogue sources fall inside the selected region")

    console.print(
        f"  ✓ Nonzero flux cells: {np.count_nonzero(hist_flux):,} "
        f"/ {hist_flux.size:,}"
    )
    _save_debug_npz(
        CACHE_DIR / f"region_histogram_raw_{RESOLUTION_TAG}.npz",
        hist_flux=hist_flux,
        hist_temp_flux=hist_temp_flux,
        hist_count=hist_count,
    )
    return hist_flux, hist_temp_flux, hist_count


# ─────────────────────────────────────────────────────────────
# PSF and tone mapping matching aurora_sky_render.py
# ─────────────────────────────────────────────────────────────

def build_rgb_cache(hist_flux, hist_temp_flux, hist_count):
    console.print("\n[AURORA] Tone-mapping rectangular stellar field")
    console.print("─" * 45)

    flux_sample = np.log1p(
        np.asarray(hist_flux).reshape(-1)[
            :: max(1, hist_flux.size // PERCENTILE_SAMPLE_SIZE)
        ]
    )
    flux_p1, flux_p2 = np.percentile(flux_sample, [0.1, 99.99])
    count_sample = np.log1p(
        np.asarray(hist_count).reshape(-1)[
            :: max(1, hist_count.size // PERCENTILE_SAMPLE_SIZE)
        ]
    )
    count_p1, count_p2 = np.percentile(count_sample, [1.0, 99.85])
    del flux_sample, count_sample

    rgb = np.lib.format.open_memmap(
        RGB_CACHE,
        mode="w+",
        dtype=np.float16,
        shape=(OUTPUT_H, OUTPUT_W, 3),
    )
    rgb_max = 0.0

    for y0 in console.progress(
        range(0, OUTPUT_H, TONE_MAP_TILE_ROWS),
        description="Tone map",
        unit="tile",
    ):
        MEMORY.throttle()
        y1 = min(y0 + TONE_MAP_TILE_ROWS, OUTPUT_H)
        flux = hist_flux[y0:y1]
        temperature = np.clip(
            hist_temp_flux[y0:y1] / np.maximum(flux, 1e-8),
            2500.0,
            25000.0,
        )
        r, g, b = temperature_to_rgb_channels(temperature)

        brightness = np.log1p(flux)
        if flux_p2 > flux_p1:
            brightness = np.clip(
                (brightness - flux_p1) / (flux_p2 - flux_p1),
                0.0,
                1.0,
            )
        else:
            brightness.fill(0.0)
        brightness = np.arcsinh(2.5 * brightness) / np.arcsinh(2.5)
        np.power(brightness, 1.5, out=brightness)

        detail = np.log1p(hist_count[y0:y1])
        if count_p2 > count_p1:
            detail = np.clip(
                (detail - count_p1) / (count_p2 - count_p1),
                0.0,
                1.0,
            )
        else:
            detail.fill(0.0)
        detail = np.arcsinh(4.5 * detail) / np.arcsinh(4.5)
        np.power(detail, 0.95, out=detail)
        detail = 0.85 + 0.15 * detail

        tile = np.empty((y1 - y0, OUTPUT_W, 3), dtype=np.float32)
        tile[:, :, 0] = r * brightness * detail
        tile[:, :, 1] = g * brightness * detail
        tile[:, :, 2] = b * brightness * detail

        luminance = tile.mean(axis=2, keepdims=True)
        tile = np.clip(luminance + 3.5 * (tile - luminance), 0.0, 1.0)
        rgb_max = max(rgb_max, float(tile.max()))
        rgb[y0:y1] = tile

    if rgb_max > 0.0:
        for y0 in console.progress(
            range(0, OUTPUT_H, TONE_MAP_TILE_ROWS),
            description="Stretch",
            unit="tile",
        ):
            MEMORY.throttle()
            y1 = min(y0 + TONE_MAP_TILE_ROWS, OUTPUT_H)
            tile = np.asarray(rgb[y0:y1], dtype=np.float32)
            np.divide(tile, rgb_max, out=tile)
            np.clip(tile, 0.0, 1.0, out=tile)
            np.power(tile, 0.45, out=tile)
            rgb[y0:y1] = tile
    rgb.flush()

    console.print("  → RGB statistics")
    console.print("    min: 0.0")
    console.print(f"    max before normalization: {rgb_max}")
    console.print(f"    mean: {float(np.mean(rgb[::64, ::64]))}")
    console.print(f"  ✓ RGB cache saved: {RGB_CACHE}")

    if SAVE_DEBUG_FRAMES:
        preview_step = max(1, max(OUTPUT_W, OUTPUT_H) // DEBUG_MAX_DIM)
        preview = np.clip(
            np.asarray(
                rgb[::preview_step, ::preview_step],
                dtype=np.float32,
            )
            * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        debug_path = CACHE_DIR / f"aurora_sky_region_preview_{RESOLUTION_TAG}.png"
        cv2.imwrite(
            str(debug_path),
            cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
        console.print(f"  ✓ Debug RGB preview saved: {debug_path}")
    return rgb


# ─────────────────────────────────────────────────────────────
# Direct rectangular PNG output
# ─────────────────────────────────────────────────────────────

def save_rectangular_png(rgb):
    console.print("\n[AURORA] Writing projection-free rectangular map")
    console.print("─" * 45)

    temp_map = CACHE_DIR / f".aurora_sky_region_bgr_u8_{RESOLUTION_TAG}.dat"
    bgr = np.memmap(
        temp_map,
        mode="w+",
        dtype=np.uint8,
        shape=(OUTPUT_H, OUTPUT_W, 3),
    )

    for y0 in console.progress(
        range(0, OUTPUT_H, TONE_MAP_TILE_ROWS),
        description="RGB output",
        unit="tile",
    ):
        MEMORY.throttle()
        y1 = min(y0 + TONE_MAP_TILE_ROWS, OUTPUT_H)
        tile = np.asarray(rgb[y0:y1], dtype=np.float32)
        tile = np.clip(tile * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        bgr[y0:y1] = tile[:, :, ::-1]
    bgr.flush()

    temp_output = MAP_FILE.with_name(f".{MAP_FILE.stem}.tmp.png")
    console.print(
        f"  → Encoding {PNG_BIT_DEPTH}-bit PNG "
        f"(compression {PNG_COMPRESSION}): {MAP_FILE}"
    )
    if not cv2.imwrite(
        str(temp_output),
        bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    ):
        raise RuntimeError(f"OpenCV could not save {temp_output}")
    temp_output.replace(MAP_FILE)

    del bgr
    try:
        temp_map.unlink()
    except FileNotFoundError:
        pass
    console.print(f"  ✓ Rectangular sky map saved: {MAP_FILE}")


def save_png_from_rgb_cache():
    rgb = np.load(RGB_CACHE, mmap_mode="r")
    expected = (OUTPUT_H, OUTPUT_W, 3)
    if rgb.shape != expected:
        raise ValueError(f"RGB cache shape {rgb.shape} does not match {expected}")
    console.print(f"  ✓ RGB cache loaded: {rgb.shape}")
    save_rectangular_png(rgb)


def build_and_save_region_map():
    hist_flux, hist_temp_flux, hist_count = build_region_histograms()
    kernel = combined_psf_kernel()

    console.print("  → Applying combined PSF: flux")
    smoothed_flux = apply_psf(hist_flux, kernel)
    console.print("  ✓ PSF complete: flux")
    del hist_flux
    gc.collect()
    console.print("  → Applying combined PSF: temperature flux")
    smoothed_temp_flux = apply_psf(
        hist_temp_flux,
        kernel,
    )
    console.print("  ✓ PSF complete: temperature flux")
    del hist_temp_flux
    gc.collect()
    console.print("  → Applying combined PSF: source count")
    smoothed_count = apply_psf(hist_count, kernel)
    console.print("  ✓ PSF complete: source count")
    del hist_count
    gc.collect()

    if SAVE_DEBUG_FRAMES:
        _save_debug_npz(
            CACHE_DIR / f"region_histogram_after_psf_{RESOLUTION_TAG}.npz",
            hist_flux=smoothed_flux,
            hist_temp_flux=smoothed_temp_flux,
            hist_count=smoothed_count,
        )

    rgb = build_rgb_cache(
        smoothed_flux,
        smoothed_temp_flux,
        smoothed_count,
    )
    del smoothed_flux, smoothed_temp_flux, smoothed_count
    gc.collect()

    np.savez(LAYOUT_FILE, **_layout_storage_values())
    save_rectangular_png(rgb)


# ─────────────────────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────────────────────

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Render a rectangular AURORA Galactic sky region."
    )
    parser.add_argument(
        "--region-name",
        help=(
            "Unique region profile name, for example galactic_center or lmc; "
            "omit for an interactive prompt"
        ),
    )
    parser.add_argument(
        "--l-center",
        type=float,
        help="Galactic longitude at the region centre in degrees",
    )
    parser.add_argument(
        "--b-center",
        type=float,
        help="Galactic latitude at the region centre in degrees",
    )
    parser.add_argument(
        "--l-width",
        type=float,
        help="Region longitude width in degrees",
    )
    parser.add_argument(
        "--b-height",
        type=float,
        help="Region latitude height in degrees",
    )
    parser.add_argument(
        "--constellations",
        choices=("ask", "yes", "no"),
        default=DEFAULT_CONSTELLATION_MODE,
        help=(
            "Overlay constellation stick figures from assets/index.json; "
            "default: ask in an interactive terminal"
        ),
    )
    parser.add_argument(
        "--coordinate-grid",
        choices=("ask", "yes", "no"),
        default=DEFAULT_GUIDE_MODE,
        help="Overlay a subtle Galactic coordinate grid; default: ask",
    )
    parser.add_argument(
        "--poland-limits",
        choices=("ask", "yes", "no"),
        default=DEFAULT_GUIDE_MODE,
        help="Overlay both Poland visibility limits; default: ask",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    configure_region_geometry(args)
    configure_region_output_paths(
        resolve_region_profile_name(args.region_name)
    )
    start_time = time.perf_counter()
    console.section("MAP RENDER — Gaia sky region")
    console.detail("Projection: rectangular Galactic coordinates")
    console.detail(
        f"Resolution: {RESOLUTION_TAG} ({OUTPUT_W} × {OUTPUT_H} px)"
    )
    console.detail(f"Region profile: {REGION_PROFILE_NAME}")
    console.detail(f"Output directory: {REGION_MAPS_DIR}")
    validate_configuration()
    MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"  ✓ Cache directory ready: {CACHE_DIR}")
    console.print(
        f"  ✓ Galactic field: l={REGION_L_CENTER_DEG:g}°, "
        f"b={REGION_B_CENTER_DEG:g}°, "
        f"{REGION_L_WIDTH_DEG:g}° × {REGION_B_HEIGHT_DEG:g}°"
    )
    console.print(f"  ✓ Output dimensions: {OUTPUT_W} × {OUTPUT_H} px")
    include_constellations = prompt_constellation_overlay(
        args.constellations
    )
    include_coordinate_grid = prompt_coordinate_grid_overlay(
        args.coordinate_grid
    )
    include_poland_limits = prompt_poland_limits_overlay(
        args.poland_limits
    )
    console.detail(
        "Selected overlays: "
        f"constellations={'yes' if include_constellations else 'no'}, "
        f"coordinates={'yes' if include_coordinate_grid else 'no'}, "
        f"Poland limits={'yes' if include_poland_limits else 'no'}"
    )

    layout_matches = _layout_matches_configuration()
    if layout_matches and _png_matches_output_format(MAP_FILE):
        console.print(f"  ✓ Cached rectangular map already exists: {MAP_FILE}")
    elif layout_matches and RGB_CACHE.exists():
        console.print("  → Rebuilding PNG from the matching RGB cache")
        save_png_from_rgb_cache()
    else:
        build_and_save_region_map()

    final_map = MAP_FILE
    include_any_overlay = (
        include_constellations
        or include_coordinate_grid
        or include_poland_limits
    )
    if include_any_overlay:
        background = sky_map_background_from_flags(
            constellations=include_constellations,
            coordinates=include_coordinate_grid,
            poland_limits=include_poland_limits,
        )
        final_map = MAP_FILE.with_name(
            f"{MAP_FILE.stem}{sky_map_background_suffix(background)}.png"
        )
        if _reference_map_is_current(
            final_map,
            include_constellations=include_constellations,
            include_coordinate_grid=include_coordinate_grid,
            include_poland_limits=include_poland_limits,
        ):
            console.success(
                f"Cached reference map already exists: {final_map}"
            )
        else:
            console.info(
                "Projecting selected static map overlays"
            )
            polylines = (
                load_constellation_region_polylines(
                    OUTPUT_W,
                    OUTPUT_H,
                    l_center_deg=REGION_L_CENTER_DEG,
                    b_center_deg=REGION_B_CENTER_DEG,
                    l_width_deg=REGION_L_WIDTH_DEG,
                    b_height_deg=REGION_B_HEIGHT_DEG,
                )
                if include_constellations
                else []
            )
            guide_layers = load_region_guide_layers(
                OUTPUT_W,
                OUTPUT_H,
                l_center_deg=REGION_L_CENTER_DEG,
                b_center_deg=REGION_B_CENTER_DEG,
                l_width_deg=REGION_L_WIDTH_DEG,
                b_height_deg=REGION_B_HEIGHT_DEG,
                include_coordinate_grid=include_coordinate_grid,
                include_poland_limits=include_poland_limits,
            )
            add_reference_overlays_to_png(
                MAP_FILE,
                final_map,
                polylines,
                guide_layers,
                png_compression=PNG_COMPRESSION,
            )
            visible_segment_count = len(polylines) + sum(
                len(layer.polylines) for layer in guide_layers
            )
            console.success(
                f"Reference-overlay map saved: {final_map} "
                f"({visible_segment_count:,} visible line segments)"
            )

    elapsed = time.perf_counter() - start_time
    console.complete("MAP RENDER — Gaia sky region")
    console.success(f"Output: {final_map}")
    console.detail(
        f"Runtime: {elapsed / 3600.0:.2f} h "
        f"({elapsed / 60.0:.1f} min, {elapsed:.1f} s)"
    )


if __name__ == "__main__":
    main()
