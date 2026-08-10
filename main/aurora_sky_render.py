"""
Generate the high-resolution AURORA Gaia all-sky map.

The pipeline is designed for a very large Gaia catalogue and selectable
8K/16K/32K/64K outputs:
FITS rows are streamed in bounded chunks, PSF operations use overlap-add
convolution, tone mapping is tiled, and the Hammer projection is rasterised
without constructing multi-gigabyte Matplotlib coordinate meshes.
"""

import argparse
import gc
import hashlib
import sys
import time
from pathlib import Path

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
    load_constellation_polylines,
)
from core.aurora_sky_guides import (
    DEFAULT_GUIDE_MODE,
    SKY_GUIDES_SOURCE_FILE,
    add_reference_overlays_to_png,
    load_full_sky_guide_layers,
    prompt_reference_overlays,
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
from core.aurora_paths import MAPS_DIR, asset_path, map_cache_path, map_path
from core.aurora_resolution import (
    prompt_resolution,
    sky_map_background_from_flags,
)


# ─────────────────────────────────────────────────────────────
# Program settings
# ─────────────────────────────────────────────────────────────

RESOLUTION = prompt_resolution()
RESOLUTION_TAG = RESOLUTION.tag

INPUT_FILE = asset_path("aurora_gaia_catalog_1_8mld.fits")
MAP_FILE = map_path(RESOLUTION.hammer_map_name)

NPZ_DIR = map_cache_path("full_sky", RESOLUTION_TAG)
MAP_LAYOUT_FILE = NPZ_DIR / RESOLUTION.hammer_layout_name
PROJECTED_RGB_CACHE = NPZ_DIR / RESOLUTION.hammer_rgb_cache_name
# Version 1 is the original temperature-weighted colour pipeline.  Keeping
# it separate from the former RGB-channel cache prevents stale colours from
# being reused after this compatibility restoration.
RENDER_PIPELINE_VERSION = 2

BINS_L = RESOLUTION.bins_l
BINS_B = RESOLUTION.bins_b
CATALOG_CHUNK_ROWS = 1_000_000
TONE_MAP_TILE_ROWS = 128
PROJECTION_TILE_ROWS = 128
PERCENTILE_SAMPLE_SIZE = 2_000_000
# The tone-mapped map is display-ready, so 16-bit output only doubles the
# amount of data without preserving useful catalogue precision.  Eight-bit RGB
# also matches what the animation renderers consume.
PNG_BIT_DEPTH = 8
PNG_COMPRESSION = 6
DEBUG_MAX_DIM = 4096

# Same diagnostic switch and console convention as
# aurora_variable_animation.py.
SAVE_DEBUG_FRAMES = True
MEMORY = MemoryController.from_environment()


# ─────────────────────────────────────────────────────────────
# Diagnostics and small helpers
# ─────────────────────────────────────────────────────────────

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
    console.print(f"  → Saving debug cache: {path}")
    np.savez_compressed(path, **debug_arrays)


def _png_matches_output_format(path):
    """Check PNG dimensions and bit depth without decoding the huge image."""
    try:
        with Path(path).open("rb") as png:
            header = png.read(26)
    except OSError:
        return False
    return (
        len(header) == 26
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR"
        and int.from_bytes(header[16:20], "big") == BINS_L
        and int.from_bytes(header[20:24], "big") == BINS_B
        and header[24] == PNG_BIT_DEPTH
        and header[25] == 2  # true-colour RGB
    )


def _file_sha256(path):
    """Return a streaming digest without loading a 16K PNG into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_map_layout():
    """Store renderer metadata and the exact clean base-map fingerprint."""
    np.savez(
        MAP_LAYOUT_FILE,
        projection=np.array(["hammer"]),
        dimensions=np.array([BINS_L, BINS_B], dtype=np.int32),
        render_pipeline_version=np.int32(RENDER_PIPELINE_VERSION),
        base_map_filename=np.array([MAP_FILE.name]),
        base_map_sha256=np.array([_file_sha256(MAP_FILE)]),
    )


def _layout_matches_renderer():
    if not MAP_LAYOUT_FILE.exists():
        return False
    try:
        with np.load(MAP_LAYOUT_FILE, allow_pickle=False) as layout:
            metadata_matches = (
                "render_pipeline_version" in layout.files
                and int(layout["render_pipeline_version"])
                == RENDER_PIPELINE_VERSION
                and np.array_equal(
                    layout["dimensions"],
                    np.array([BINS_L, BINS_B], dtype=np.int32),
                )
                and str(layout["projection"][0]) == "hammer"
                and "base_map_filename" in layout.files
                and str(layout["base_map_filename"][0]) == MAP_FILE.name
                and "base_map_sha256" in layout.files
            )
            if not metadata_matches or not MAP_FILE.is_file():
                return False
            return str(layout["base_map_sha256"][0]) == _file_sha256(MAP_FILE)
    except (OSError, KeyError, ValueError):
        return False


def _reference_map_is_current(
    output_path: Path,
    *,
    include_constellations: bool,
    include_coordinate_grid: bool,
    include_poland_limits: bool,
) -> bool:
    if not _png_matches_output_format(output_path):
        return False
    try:
        dependencies = [MAP_FILE]
        if include_constellations:
            dependencies.extend(
                (CONSTELLATION_INDEX_FILE, CONSTELLATIONS_SOURCE_FILE)
            )
        if include_coordinate_grid or include_poland_limits:
            dependencies.append(SKY_GUIDES_SOURCE_FILE)
        output_time = output_path.stat().st_mtime_ns
        return all(
            output_time >= path.stat().st_mtime_ns
            for path in dependencies
        )
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────
# Streaming catalogue histogram
# ─────────────────────────────────────────────────────────────

def build_catalog_histograms():
    console.print("\n[AURORA] Building Gaia histograms")
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

        # Bright-source clipping is derived from a bounded magnitude sample.
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
        if not mag_sample.size:
            raise RuntimeError("No finite phot_g_mean_mag values found")
        bright_mag = np.percentile(mag_sample, 0.1)
        flux_max = np.float32(10.0 ** (-0.4 * bright_mag))
        console.print(
            f"  ✓ Magnitude sample ready: {len(mag_sample):,} values "
            f"in {time.perf_counter() - sample_t0:.1f}s"
        )
        del mag_sample

        shape = (BINS_B, BINS_L)
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

        valid_total = 0
        teff_total = 0
        bp_rp_total = 0
        missing_temperature_total = 0
        ranges = range(0, row_count, CATALOG_CHUNK_ROWS)
        for start in console.progress(
            ranges,
            total=(row_count + CATALOG_CHUNK_ROWS - 1)
            // CATALOG_CHUNK_ROWS,
            desc="Histogramming catalogue",
            unit="chunk",
        ):
            MEMORY.throttle()
            stop = min(start + CATALOG_CHUNK_ROWS, row_count)
            lon_deg = np.asarray(data[names["l"]][start:stop], dtype=np.float32)
            lat_deg = np.asarray(data[names["b"]][start:stop], dtype=np.float32)
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

            valid = (
                np.isfinite(lon_deg)
                & np.isfinite(lat_deg)
                & np.isfinite(magnitude)
                & (lat_deg >= -90.0)
                & (lat_deg <= 90.0)
            )
            if not np.any(valid):
                continue
            lon_deg = lon_deg[valid]
            lat_deg = lat_deg[valid]
            magnitude = magnitude[valid]
            teff = teff[valid]
            bp_rp = bp_rp[valid]
            valid_total += len(lon_deg)
            teff_total += np.count_nonzero(np.isfinite(teff))
            bp_rp_total += np.count_nonzero(np.isfinite(bp_rp))
            missing_temperature_total += np.count_nonzero(
                ~np.isfinite(teff) & ~np.isfinite(bp_rp)
            )

            # Gaia longitude is [0, 360). Center, negate, and convert directly
            # to bins without allocating edge arrays or calling searchsorted.
            centered = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)
            centered = -centered
            x_index = np.floor(
                (centered + 180.0) * (BINS_L / 360.0)
            ).astype(np.int64)
            y_index = np.floor(
                (lat_deg + 90.0) * (BINS_B / 180.0)
            ).astype(np.int64)
            np.clip(x_index, 0, BINS_L - 1, out=x_index)
            np.clip(y_index, 0, BINS_B - 1, out=y_index)
            flat_index = y_index * BINS_L + x_index

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

        console.print(f"  ✓ Valid catalogue rows accumulated: {valid_total:,}")
        console.print("  → Temperature and colour statistics")
        console.print(f"    teff_gspphot available: {teff_total:,}")
        console.print(f"    bp_rp available: {bp_rp_total:,}")
        console.print(
            "    no temperature or colour data: "
            f"{missing_temperature_total:,}"
        )

    console.print(
        f"  ✓ Nonzero flux cells: {np.count_nonzero(hist_flux):,} "
        f"/ {hist_flux.size:,}"
    )
    _save_debug_npz(
        NPZ_DIR / f"histogram_raw_{RESOLUTION_TAG}.npz",
        hist_flux=hist_flux,
        hist_temp_flux=hist_temp_flux,
        hist_count=hist_count,
    )
    return hist_flux, hist_temp_flux, hist_count


# ─────────────────────────────────────────────────────────────
# PSF, tone mapping, and Hammer projection
# ─────────────────────────────────────────────────────────────

def build_rgb_cache(hist_flux, hist_temp_flux, hist_count):
    console.print("\n[AURORA] Tone-mapping stellar field")
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
        PROJECTED_RGB_CACHE,
        mode="w+",
        dtype=np.float16,
        shape=(BINS_B, BINS_L, 3),
    )
    rgb_max = 0.0

    for y0 in console.progress(
        range(0, BINS_B, TONE_MAP_TILE_ROWS),
        desc="Tone mapping",
        unit="tile",
    ):
        MEMORY.throttle()
        y1 = min(y0 + TONE_MAP_TILE_ROWS, BINS_B)
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

        tile = np.empty((y1 - y0, BINS_L, 3), dtype=np.float32)
        tile[:, :, 0] = r * brightness * detail
        tile[:, :, 1] = g * brightness * detail
        tile[:, :, 2] = b * brightness * detail

        luminance = tile.mean(axis=2, keepdims=True)
        tile = np.clip(luminance + 3.5 * (tile - luminance), 0.0, 1.0)
        rgb_max = max(rgb_max, float(tile.max()))
        rgb[y0:y1] = tile

    if rgb_max > 0.0:
        for y0 in console.progress(
            range(0, BINS_B, TONE_MAP_TILE_ROWS),
            desc="Final display stretch",
            unit="tile",
        ):
            MEMORY.throttle()
            y1 = min(y0 + TONE_MAP_TILE_ROWS, BINS_B)
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
    console.print(f"  ✓ RGB cache saved: {PROJECTED_RGB_CACHE}")

    if SAVE_DEBUG_FRAMES:
        preview_step = max(1, BINS_L // 4096)
        preview = np.clip(
            np.asarray(
                rgb[::preview_step, ::preview_step],
                dtype=np.float32,
            )
            * 255.0,
            0,
            255,
        )
        preview = preview.astype(np.uint8)
        debug_rgb_path = NPZ_DIR / "debug_rgb.png"
        cv2.imwrite(
            str(debug_rgb_path),
            cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
        console.print(f"  ✓ Debug RGB preview saved: {debug_rgb_path}")
    return rgb


def project_hammer_to_png(rgb, output_path=None):
    """Rasterise an equirectangular RGB field into a Hammer ellipse by tiles."""
    console.print("\n[AURORA] Rasterizing Hammer projection")
    console.print("─" * 45)
    output_path = MAP_FILE if output_path is None else Path(output_path)
    temp_map = NPZ_DIR / f".{output_path.stem}_projected_u8.dat"
    projected = np.memmap(
        temp_map,
        mode="w+",
        dtype=np.uint8,
        shape=(BINS_B, BINS_L, 3),
    )

    x_hammer = (
        ((np.arange(BINS_L, dtype=np.float32) + 0.5) / BINS_L) * 2.0 - 1.0
    ) * (2.0 * np.sqrt(2.0))

    for y0 in console.progress(
        range(0, BINS_B, PROJECTION_TILE_ROWS),
        desc="Hammer projection",
        unit="tile",
    ):
        MEMORY.throttle()
        y1 = min(y0 + PROJECTION_TILE_ROWS, BINS_B)
        y_hammer = (
            1.0
            - ((np.arange(y0, y1, dtype=np.float32) + 0.5) / BINS_B) * 2.0
        )[:, None] * np.sqrt(2.0)
        x = x_hammer[None, :]
        z2 = 1.0 - np.square(x) / 16.0 - np.square(y_hammer) / 4.0
        # The Hammer domain is x²/8 + y²/2 <= 1, which is equivalent
        # to z² >= 1/2 for the inverse formula below.
        valid = z2 >= 0.5
        z = np.sqrt(np.maximum(z2, 0.0))
        longitude = 2.0 * np.arctan2(
            z * x,
            2.0 * (2.0 * z * z - 1.0),
        )
        latitude = np.arcsin(np.clip(z * y_hammer, -1.0, 1.0))

        source_x = np.floor(
            (longitude + np.pi) * (BINS_L / (2.0 * np.pi))
        ).astype(np.int64)
        source_y = np.floor(
            (latitude + np.pi / 2.0) * (BINS_B / np.pi)
        ).astype(np.int64)
        np.clip(source_x, 0, BINS_L - 1, out=source_x)
        np.clip(source_y, 0, BINS_B - 1, out=source_y)

        sampled = np.asarray(rgb[source_y, source_x], dtype=np.float32)
        tile = np.clip(sampled * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        tile[~valid] = 0
        # OpenCV expects BGR at file-write time; store projected data as BGR
        # so no second full-size array is needed.
        projected[y0:y1] = tile[:, :, ::-1]

    projected.flush()
    temp_output = output_path.with_name(f".{output_path.stem}.tmp.png")
    console.print(
        f"  → Encoding {PNG_BIT_DEPTH}-bit PNG "
        f"(compression {PNG_COMPRESSION}): {output_path}"
    )
    if not cv2.imwrite(
        str(temp_output),
        projected,
        [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
    ):
        raise RuntimeError(f"OpenCV could not save {temp_output}")
    temp_output.replace(output_path)

    del projected
    try:
        temp_map.unlink()
    except FileNotFoundError:
        pass
    console.print(f"  ✓ Hammer projection map saved: {output_path}")


def save_png_from_projected_cache():
    """Rebuild the final Hammer PNG from the memory-mappable RGB cache."""
    console.print("\n[AURORA] Restoring PNG from projected RGB cache")
    console.print("─" * 45)
    rgb = np.load(PROJECTED_RGB_CACHE, mmap_mode="r")
    expected = (BINS_B, BINS_L, 3)
    if rgb.shape != expected:
        raise ValueError(f"RGB cache shape {rgb.shape} does not match {expected}")
    console.print(f"  ✓ RGB cache loaded: {rgb.shape}")
    project_hammer_to_png(rgb)


def build_and_save_sky_map():
    hist_flux, hist_temp_flux, hist_count = build_catalog_histograms()
    kernel = combined_psf_kernel()

    console.print("  → Applying combined PSF: flux")
    smoothed_flux = apply_psf(hist_flux, kernel)
    console.print("  ✓ PSF complete: flux")
    del hist_flux
    gc.collect()
    console.print("  → Applying combined PSF: temperature-weighted flux")
    smoothed_temp_flux = apply_psf(hist_temp_flux, kernel)
    console.print("  ✓ PSF complete: temperature-weighted flux")
    del hist_temp_flux
    gc.collect()
    console.print("  → Applying combined PSF: source count")
    smoothed_count = apply_psf(hist_count, kernel)
    console.print("  ✓ PSF complete: source count")
    del hist_count
    gc.collect()

    if SAVE_DEBUG_FRAMES:
        _save_debug_npz(
            NPZ_DIR / f"histogram_after_psf_{RESOLUTION_TAG}.npz",
            hist_flux=smoothed_flux,
            hist_temp_flux=smoothed_temp_flux,
            hist_count=smoothed_count,
        )

    rgb = build_rgb_cache(
        smoothed_flux,
        smoothed_temp_flux,
        smoothed_count,
    )
    del (
        smoothed_flux,
        smoothed_temp_flux,
        smoothed_count,
    )
    gc.collect()

    project_hammer_to_png(rgb)
    _write_map_layout()


# ─────────────────────────────────────────────────────────────
# Main program
# ─────────────────────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the AURORA Gaia all-sky Hammer map."
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
    start_time = time.perf_counter()
    console.section("MAP RENDER — Gaia all-sky")
    console.detail("Projection: Hammer equal-area")
    console.detail(f"Resolution: {RESOLUTION_TAG} ({BINS_L} × {BINS_B} px)")
    console.detail(f"Output directory: {MAPS_DIR}")
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"  ✓ Cache directory ready: {NPZ_DIR}")
    (
        include_constellations,
        include_coordinate_grid,
        include_poland_limits,
    ) = prompt_reference_overlays(
        constellations=args.constellations,
        coordinate_grid=args.coordinate_grid,
        poland_limits=args.poland_limits,
    )
    console.detail(
        "Selected overlays: "
        f"constellations={'yes' if include_constellations else 'no'}, "
        f"coordinates={'yes' if include_coordinate_grid else 'no'}, "
        f"Poland limits={'yes' if include_poland_limits else 'no'}"
    )

    if _png_matches_output_format(MAP_FILE) and _layout_matches_renderer():
        console.print(f"  ✓ Cached sky map already exists: {MAP_FILE}")
    elif PROJECTED_RGB_CACHE.exists():
        if MAP_FILE.exists():
            console.print(
                "  → Existing base map has obsolete metadata or failed its "
                "integrity check; re-encoding it from the clean RGB cache"
            )
        save_png_from_projected_cache()
        _write_map_layout()
    else:
        build_and_save_sky_map()

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
        final_map = map_path(
            RESOLUTION.hammer_background_map_name(background)
        )
        clean_base_digest = _file_sha256(MAP_FILE)
        if _reference_map_is_current(
            final_map,
            include_constellations=include_constellations,
            include_coordinate_grid=include_coordinate_grid,
            include_poland_limits=include_poland_limits,
        ):
            console.print(
                f"  ✓ Cached reference map already exists: {final_map}"
            )
        else:
            console.print(
                "  → Projecting selected static map overlays"
            )
            polylines = (
                load_constellation_polylines(BINS_L, BINS_B)
                if include_constellations
                else []
            )
            guide_layers = load_full_sky_guide_layers(
                BINS_L,
                BINS_B,
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
            console.print(
                f"  ✓ Reference-overlay map saved: {final_map}"
            )
        if _file_sha256(MAP_FILE) != clean_base_digest:
            raise RuntimeError(
                "Static-overlay rendering modified the clean base map"
            )

    elapsed = time.perf_counter() - start_time
    console.complete("MAP RENDER — Gaia all-sky")
    console.success(f"Output: {final_map}")
    console.detail(
        f"Runtime: {elapsed / 3600.0:.2f} h "
        f"({elapsed / 60.0:.1f} min, {elapsed:.1f} s)"
    )


if __name__ == "__main__":
    main()
