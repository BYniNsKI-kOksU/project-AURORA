"""Static coordinate and Poland-visibility guides for AURORA sky maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from astropy.coordinates import SkyCoord

from core.aurora_constellations import (
    draw_constellation_polylines,
    galactic_to_hammer_pixels,
    prompt_constellation_overlay,
)
from core.aurora_console import console


# ─────────────────────────────────────────────────────────────
# Visual configuration (BGR because the map renderers use OpenCV)
# ─────────────────────────────────────────────────────────────

SKY_GUIDES_SOURCE_FILE = Path(__file__).resolve()
REFERENCE_HEIGHT_PX = 4096.0
PNG_COMPRESSION = 6
DEFAULT_GUIDE_MODE = "ask"
DEFAULT_GUIDE_OVERLAY = False

# The constellation layer is pale blue.  The grid is deliberately neutral,
# while the two Poland limits use warm, saturated hues and different dash
# patterns so all three kinds of information remain visually separate.
GRID_COLOR_BGR = (162, 156, 150)          # neutral slate-grey
GRID_OPACITY = 0.22
GRID_LINE_WIDTH_AT_REFERENCE = 1.0
FULL_SKY_LONGITUDE_SPACING_DEG = 30.0
FULL_SKY_LATITUDE_SPACING_DEG = 15.0
FULL_SKY_SAMPLE_STEP_DEG = 0.25
REGION_GRID_TARGET_LINES = 9.0
REGION_GRID_SPACINGS_DEG = (1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0)

POLAND_REFERENCE_LATITUDE_DEG = 52.0
POLAND_THEORETICAL_DECLINATION_DEG = POLAND_REFERENCE_LATITUDE_DEG - 90.0
POLAND_PRACTICAL_MIN_ALTITUDE_DEG = 10.0
POLAND_PRACTICAL_DECLINATION_DEG = (
    POLAND_REFERENCE_LATITUDE_DEG
    - (90.0 - POLAND_PRACTICAL_MIN_ALTITUDE_DEG)
)
POLAND_CURVE_SAMPLE_STEP_DEG = 0.05
POLAND_LINE_WIDTH_AT_REFERENCE = 3.0
POLAND_THEORETICAL_COLOR_BGR = (70, 186, 255)  # gold, RGB #ffba46
POLAND_THEORETICAL_OPACITY = 0.88
POLAND_THEORETICAL_DASH_AT_REFERENCE = (28.0, 17.0)
POLAND_PRACTICAL_COLOR_BGR = (164, 93, 255)    # raspberry, RGB #ff5da4
POLAND_PRACTICAL_OPACITY = 0.90
POLAND_PRACTICAL_DASH_AT_REFERENCE = (17.0, 9.0, 4.0, 9.0)
MAX_CONTINUOUS_STROKE_AT_REFERENCE = 768.0


@dataclass
class SkyGuideLayer:
    """One raster-ready family of map-guide polylines."""

    name: str
    polylines: list[np.ndarray]
    color_bgr: tuple[int, int, int]
    opacity: float
    line_width_at_reference: float
    dash_at_reference: tuple[float, ...] | None = None


def _prompt_guide_overlay(
    mode: str,
    *,
    prompt: str,
    cli_option: str,
) -> bool:
    normalized = str(mode).strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    if normalized != "ask":
        raise ValueError("Guide mode must be 'ask', 'yes', or 'no'")
    if not sys.stdin.isatty():
        console.info(
            f"Non-interactive mode: {prompt.casefold()} disabled; "
            f"use {cli_option} yes to enable it"
        )
        return False
    return console.confirm(prompt, default=DEFAULT_GUIDE_OVERLAY)


def prompt_coordinate_grid_overlay(
    mode: str = DEFAULT_GUIDE_MODE,
) -> bool:
    """Ask independently whether the Galactic coordinate grid is wanted."""
    return _prompt_guide_overlay(
        mode,
        prompt="Add subtle Galactic coordinate grid",
        cli_option="--coordinate-grid",
    )


def prompt_poland_limits_overlay(
    mode: str = DEFAULT_GUIDE_MODE,
) -> bool:
    """Ask independently whether both Poland visibility limits are wanted."""
    return _prompt_guide_overlay(
        mode,
        prompt="Add both Poland visibility-limit lines",
        cli_option="--poland-limits",
    )


def prompt_reference_overlays(
    *,
    constellations: str,
    coordinate_grid: str,
    poland_limits: str,
) -> tuple[bool, bool, bool]:
    """Ask for all three map-reference layers in one consistent order."""
    return (
        prompt_constellation_overlay(constellations),
        prompt_coordinate_grid_overlay(coordinate_grid),
        prompt_poland_limits_overlay(poland_limits),
    )


def _split_projection_seam(
    points: np.ndarray,
    *,
    width: int,
) -> list[np.ndarray]:
    if len(points) < 2:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(points[:, 0])) > width * 0.25) + 1
    return [piece for piece in np.split(points, breaks) if len(piece) >= 2]


def _project_hammer_curve(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    width: int,
    height: int,
) -> list[np.ndarray]:
    pixel_x, pixel_y = galactic_to_hammer_pixels(
        longitude_deg,
        latitude_deg,
        width,
        height,
    )
    points = np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)
    return _split_projection_seam(points, width=width)


def _equatorial_declination_in_galactic(
    declination_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    right_ascension = np.arange(
        0.0,
        360.0 + POLAND_CURVE_SAMPLE_STEP_DEG * 0.5,
        POLAND_CURVE_SAMPLE_STEP_DEG,
        dtype=np.float64,
    )
    declination = np.full_like(right_ascension, float(declination_deg))
    galactic = SkyCoord(
        ra=right_ascension,
        dec=declination,
        unit="deg",
        frame="icrs",
    ).galactic
    return (
        np.asarray(galactic.l.degree, dtype=np.float64),
        np.asarray(galactic.b.degree, dtype=np.float64),
    )


def load_full_sky_guide_layers(
    width: int,
    height: int,
    *,
    include_coordinate_grid: bool = True,
    include_poland_limits: bool = True,
) -> list[SkyGuideLayer]:
    """Build a Galactic grid and two Poland visibility limits for Hammer."""
    if width < 1 or height < 1:
        raise ValueError("Sky-guide target dimensions must be positive")

    grid_polylines: list[np.ndarray] = []
    longitude_samples = np.arange(
        0.0,
        360.0 + FULL_SKY_SAMPLE_STEP_DEG * 0.5,
        FULL_SKY_SAMPLE_STEP_DEG,
    )
    for latitude in np.arange(
        -90.0 + FULL_SKY_LATITUDE_SPACING_DEG,
        90.0,
        FULL_SKY_LATITUDE_SPACING_DEG,
    ):
        grid_polylines.extend(
            _project_hammer_curve(
                longitude_samples,
                np.full_like(longitude_samples, latitude),
                width,
                height,
            )
        )

    latitude_samples = np.arange(
        -90.0,
        90.0 + FULL_SKY_SAMPLE_STEP_DEG * 0.5,
        FULL_SKY_SAMPLE_STEP_DEG,
    )
    for longitude in np.arange(
        0.0,
        360.0,
        FULL_SKY_LONGITUDE_SPACING_DEG,
    ):
        grid_polylines.extend(
            _project_hammer_curve(
                np.full_like(latitude_samples, longitude),
                latitude_samples,
                width,
                height,
            )
        )

    theoretical_l, theoretical_b = _equatorial_declination_in_galactic(
        POLAND_THEORETICAL_DECLINATION_DEG
    )
    practical_l, practical_b = _equatorial_declination_in_galactic(
        POLAND_PRACTICAL_DECLINATION_DEG
    )
    layers: list[SkyGuideLayer] = []
    if include_coordinate_grid:
        layers.append(
            SkyGuideLayer(
                "Galactic coordinate grid",
                grid_polylines,
                GRID_COLOR_BGR,
                GRID_OPACITY,
                GRID_LINE_WIDTH_AT_REFERENCE,
            )
        )
    if include_poland_limits:
        layers.extend(
            (
                SkyGuideLayer(
                    "Poland theoretical horizon (dec -38 deg)",
                    _project_hammer_curve(
                        theoretical_l,
                        theoretical_b,
                        width,
                        height,
                    ),
                    POLAND_THEORETICAL_COLOR_BGR,
                    POLAND_THEORETICAL_OPACITY,
                    POLAND_LINE_WIDTH_AT_REFERENCE,
                    POLAND_THEORETICAL_DASH_AT_REFERENCE,
                ),
                SkyGuideLayer(
                    "Poland practical 10-degree altitude limit (dec -28 deg)",
                    _project_hammer_curve(
                        practical_l,
                        practical_b,
                        width,
                        height,
                    ),
                    POLAND_PRACTICAL_COLOR_BGR,
                    POLAND_PRACTICAL_OPACITY,
                    POLAND_LINE_WIDTH_AT_REFERENCE,
                    POLAND_PRACTICAL_DASH_AT_REFERENCE,
                ),
            ),
        )
    return layers


def _nice_region_grid_spacing(span_deg: float) -> float:
    target = float(span_deg) / REGION_GRID_TARGET_LINES
    for spacing in REGION_GRID_SPACINGS_DEG:
        if spacing >= target:
            return spacing
    return REGION_GRID_SPACINGS_DEG[-1]


def _project_region_curve(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    width: int,
    height: int,
    *,
    l_center_deg: float,
    b_center_deg: float,
    l_width_deg: float,
    b_height_deg: float,
) -> list[np.ndarray]:
    half_l = l_width_deg * 0.5
    b_min = b_center_deg - b_height_deg * 0.5
    b_max = b_center_deg + b_height_deg * 0.5
    delta_l = (np.asarray(longitude_deg) - l_center_deg + 180.0) % 360.0 - 180.0
    latitude = np.asarray(latitude_deg)
    inside = (
        (np.abs(delta_l) <= half_l)
        & (latitude >= b_min)
        & (latitude <= b_max)
    )
    pixel_x = (half_l - delta_l) * (width / l_width_deg)
    pixel_y = (b_max - latitude) * (height / b_height_deg)
    points = np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)
    breaks = np.flatnonzero(np.diff(inside.astype(np.int8)) != 0) + 1
    projected: list[np.ndarray] = []
    for point_run, inside_run in zip(
        np.split(points, breaks),
        np.split(inside, breaks),
        strict=True,
    ):
        if not inside_run[0] or len(point_run) < 2:
            continue
        point_run[:, 0] = np.clip(point_run[:, 0], 0, width - 1)
        point_run[:, 1] = np.clip(point_run[:, 1], 0, height - 1)
        projected.extend(_split_projection_seam(point_run, width=width))
    return projected


def load_region_guide_layers(
    width: int,
    height: int,
    *,
    l_center_deg: float,
    b_center_deg: float,
    l_width_deg: float,
    b_height_deg: float,
    include_coordinate_grid: bool = True,
    include_poland_limits: bool = True,
) -> list[SkyGuideLayer]:
    """Build reference layers clipped to a rectangular Galactic field."""
    if width < 1 or height < 1:
        raise ValueError("Sky-guide target dimensions must be positive")

    half_l = l_width_deg * 0.5
    b_min = b_center_deg - b_height_deg * 0.5
    b_max = b_center_deg + b_height_deg * 0.5
    l_spacing = _nice_region_grid_spacing(l_width_deg)
    b_spacing = _nice_region_grid_spacing(b_height_deg)
    unwrapped_l_min = l_center_deg - half_l
    unwrapped_l_max = l_center_deg + half_l
    first_l = np.ceil(unwrapped_l_min / l_spacing) * l_spacing
    first_b = np.ceil(b_min / b_spacing) * b_spacing
    grid_polylines: list[np.ndarray] = []

    for longitude in np.arange(
        first_l,
        unwrapped_l_max + l_spacing * 0.5,
        l_spacing,
    ):
        x = int(round((half_l - (longitude - l_center_deg)) * width / l_width_deg))
        if 0 <= x < width:
            grid_polylines.append(
                np.array([[x, 0], [x, height - 1]], dtype=np.int32)
            )
    for latitude in np.arange(first_b, b_max + b_spacing * 0.5, b_spacing):
        y = int(round((b_max - latitude) * height / b_height_deg))
        if 0 <= y < height:
            grid_polylines.append(
                np.array([[0, y], [width - 1, y]], dtype=np.int32)
            )

    projection_arguments = dict(
        width=width,
        height=height,
        l_center_deg=l_center_deg,
        b_center_deg=b_center_deg,
        l_width_deg=l_width_deg,
        b_height_deg=b_height_deg,
    )
    theoretical_l, theoretical_b = _equatorial_declination_in_galactic(
        POLAND_THEORETICAL_DECLINATION_DEG
    )
    practical_l, practical_b = _equatorial_declination_in_galactic(
        POLAND_PRACTICAL_DECLINATION_DEG
    )
    layers: list[SkyGuideLayer] = []
    if include_coordinate_grid:
        layers.append(
            SkyGuideLayer(
                "Galactic coordinate grid",
                grid_polylines,
                GRID_COLOR_BGR,
                GRID_OPACITY,
                GRID_LINE_WIDTH_AT_REFERENCE,
            )
        )
    if include_poland_limits:
        layers.extend(
            (
                SkyGuideLayer(
                    "Poland theoretical horizon (dec -38 deg)",
                    _project_region_curve(
                        theoretical_l,
                        theoretical_b,
                        **projection_arguments,
                    ),
                    POLAND_THEORETICAL_COLOR_BGR,
                    POLAND_THEORETICAL_OPACITY,
                    POLAND_LINE_WIDTH_AT_REFERENCE,
                    POLAND_THEORETICAL_DASH_AT_REFERENCE,
                ),
                SkyGuideLayer(
                    "Poland practical 10-degree altitude limit (dec -28 deg)",
                    _project_region_curve(
                        practical_l,
                        practical_b,
                        **projection_arguments,
                    ),
                    POLAND_PRACTICAL_COLOR_BGR,
                    POLAND_PRACTICAL_OPACITY,
                    POLAND_LINE_WIDTH_AT_REFERENCE,
                    POLAND_PRACTICAL_DASH_AT_REFERENCE,
                ),
            ),
        )
    return layers


def _stroke_fragments(
    points: np.ndarray,
    *,
    dash_pattern: tuple[float, ...] | None,
    max_continuous_length: float,
) -> list[np.ndarray]:
    """Split a line into bounded solid fragments, optionally using dashes."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return []
    pattern = (
        tuple(max(1.0, float(value)) for value in dash_pattern)
        if dash_pattern
        else (max(1.0, float(max_continuous_length)),)
    )
    pattern_index = 0
    pattern_remaining = pattern[0]
    draw_on = True
    fragments: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []

    for start, stop in zip(points, points[1:]):
        vector = stop - start
        segment_length = float(np.linalg.norm(vector))
        if segment_length <= 1.0e-9:
            continue
        direction = vector / segment_length
        travelled = 0.0
        while travelled < segment_length - 1.0e-9:
            step = min(segment_length - travelled, pattern_remaining)
            part_start = start + direction * travelled
            part_stop = start + direction * (travelled + step)
            if draw_on:
                if not current:
                    current.append(part_start)
                current.append(part_stop)
            elif current:
                fragments.append(current)
                current = []
            travelled += step
            pattern_remaining -= step
            if pattern_remaining <= 1.0e-9:
                if draw_on and current:
                    fragments.append(current)
                    current = []
                pattern_index = (pattern_index + 1) % len(pattern)
                pattern_remaining = pattern[pattern_index]
                draw_on = pattern_index % 2 == 0
    if current:
        fragments.append(current)
    return [
        np.rint(np.asarray(fragment)).astype(np.int32)
        for fragment in fragments
        if len(fragment) >= 2
    ]


def _screen_blend_fragment(
    bgr: np.ndarray,
    points: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    opacity: float,
    thickness: int,
) -> None:
    import cv2

    height, width = bgr.shape[:2]
    padding = thickness + 2
    x0 = max(0, int(points[:, 0].min()) - padding)
    x1 = min(width, int(points[:, 0].max()) + padding + 1)
    y0 = max(0, int(points[:, 1].min()) - padding)
    y1 = min(height, int(points[:, 1].max()) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        return
    local_points = points - np.array([x0, y0], dtype=np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.polylines(
        mask,
        [local_points],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    coverage = mask.astype(np.float32) * (float(opacity) / 255.0)
    color = np.asarray(color_bgr, dtype=np.float32) / 255.0
    region = bgr[y0:y1, x0:x1]
    for channel in range(3):
        values = region[:, :, channel].astype(np.float32)
        values += (255.0 - values) * coverage * color[channel]
        region[:, :, channel] = np.clip(values + 0.5, 0.0, 255.0).astype(
            np.uint8
        )


def draw_sky_guide_layer(bgr: np.ndarray, layer: SkyGuideLayer) -> None:
    """Screen-blend one guide layer without allocating a full-frame mask."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Sky-guide target must be an 8-bit BGR image")
    scale = bgr.shape[0] / REFERENCE_HEIGHT_PX
    thickness = max(1, int(round(layer.line_width_at_reference * scale)))
    dash_pattern = (
        tuple(value * scale for value in layer.dash_at_reference)
        if layer.dash_at_reference
        else None
    )
    max_continuous_length = MAX_CONTINUOUS_STROKE_AT_REFERENCE * scale
    for polyline in layer.polylines:
        for fragment in _stroke_fragments(
            polyline,
            dash_pattern=dash_pattern,
            max_continuous_length=max_continuous_length,
        ):
            _screen_blend_fragment(
                bgr,
                fragment,
                color_bgr=layer.color_bgr,
                opacity=layer.opacity,
                thickness=thickness,
            )


def add_reference_overlays_to_png(
    input_path: Path,
    output_path: Path,
    constellation_polylines: list[np.ndarray],
    guide_layers: list[SkyGuideLayer],
    *,
    png_compression: int = PNG_COMPRESSION,
) -> None:
    """Bake grid, constellations and Poland limits into one map copy."""
    import cv2

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve(strict=False) == output_path.resolve(strict=False):
        raise ValueError(
            "Reference-overlay output must not overwrite the base map"
        )
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {input_path}")

    # Neutral coordinate grid goes below the informational overlays.
    grid_layers = [
        layer for layer in guide_layers
        if layer.name == "Galactic coordinate grid"
    ]
    foreground_layers = [
        layer for layer in guide_layers
        if layer.name != "Galactic coordinate grid"
    ]
    for layer in grid_layers:
        draw_sky_guide_layer(image, layer)
    draw_constellation_polylines(image, constellation_polylines)
    for layer in foreground_layers:
        draw_sky_guide_layer(image, layer)

    temporary = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        if not cv2.imwrite(
            str(temporary),
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)],
        ):
            raise RuntimeError(f"OpenCV could not save {temporary}")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
