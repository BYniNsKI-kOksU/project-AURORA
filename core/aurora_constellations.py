"""Load and rasterise constellation stick figures on Hammer sky maps."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import FK4, SkyCoord

from core.aurora_console import console
from core.aurora_paths import asset_path


CONSTELLATION_INDEX_FILE = asset_path("index.json")
CONSTELLATIONS_SOURCE_FILE = Path(__file__).resolve()
EXPECTED_CONSTELLATION_COUNT = 88
FULL_SKY_MAX_STEP_DEGREES = 0.75
REGION_MAX_STEP_DEGREES = 0.25
CONSTELLATION_LINE_COLOR_BGR = (255, 210, 150)
CONSTELLATION_LINE_OPACITY = 0.52
CONSTELLATION_LINE_REFERENCE_HEIGHT = 4096.0
CONSTELLATION_LINE_PADDING_PIXELS = 2
DEFAULT_PNG_COMPRESSION = 6
DEFAULT_CONSTELLATION_MODE = "ask"
DEFAULT_CONSTELLATION_OVERLAY = False


def prompt_constellation_overlay(
    mode: str = DEFAULT_CONSTELLATION_MODE,
) -> bool:
    """Resolve the shared constellation-overlay choice for map renderers."""
    normalized = str(mode).strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    if normalized != "ask":
        raise ValueError("Constellation mode must be 'ask', 'yes', or 'no'")
    if not sys.stdin.isatty():
        console.info(
            "Non-interactive mode: constellation overlay disabled; "
            "use --constellations yes to enable it"
        )
        return False
    return console.confirm(
        "Add constellation lines from assets/index.json",
        default=DEFAULT_CONSTELLATION_OVERLAY,
    )


def _hms_to_degrees(text: str) -> float:
    fields = [float(value) for value in text.split(":")]
    if not 1 <= len(fields) <= 3:
        raise ValueError(f"Invalid right ascension value: {text!r}")
    fields.extend([0.0] * (3 - len(fields)))
    hours, minutes, seconds = fields
    return (hours + minutes / 60.0 + seconds / 3600.0) * 15.0


def _dms_to_degrees(text: str) -> float:
    sign = -1.0 if text.strip().startswith("-") else 1.0
    fields = [
        float(value)
        for value in text.strip().lstrip("+-").split(":")
    ]
    if not 1 <= len(fields) <= 3:
        raise ValueError(f"Invalid declination value: {text!r}")
    fields.extend([0.0] * (3 - len(fields)))
    degrees, minutes, seconds = fields
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def read_constellation_index(
    path: Path = CONSTELLATION_INDEX_FILE,
) -> dict[str, Any]:
    """Read and lightly validate the constellation index JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    constellations = data.get("constellations")
    if (
        not isinstance(constellations, list)
        or len(constellations) != EXPECTED_CONSTELLATION_COUNT
    ):
        raise ValueError(
            f"{path} must contain {EXPECTED_CONSTELLATION_COUNT} constellations"
        )
    edges = data.get("edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError(f"{path} must contain non-empty constellation edges")
    return data


def read_constellation_edges(
    path: Path = CONSTELLATION_INDEX_FILE,
) -> list[tuple[float, float, float, float]]:
    """Return boundary edge endpoints from ``assets/index.json``.

    The source encodes IAU boundary segments in the B1875 equatorial frame:
    edge id, edge type, RA1, Dec1, RA2, Dec2, and adjacent constellations.
    """
    data = read_constellation_index(path)
    edges: list[tuple[float, float, float, float]] = []
    for index, raw_edge in enumerate(data["edges"], start=1):
        if not isinstance(raw_edge, str):
            raise ValueError(f"{path}: edge #{index} is not a string")
        fields = raw_edge.split()
        if len(fields) != 8:
            raise ValueError(
                f"{path}: edge #{index} has {len(fields)} fields, expected 8"
            )
        try:
            edges.append(
                (
                    _hms_to_degrees(fields[2]),
                    _dms_to_degrees(fields[3]),
                    _hms_to_degrees(fields[4]),
                    _dms_to_degrees(fields[5]),
                )
            )
        except ValueError as error:
            raise ValueError(f"{path}: invalid edge #{index}") from error
    return edges


def read_constellation_line_definitions(
    path: Path = CONSTELLATION_INDEX_FILE,
) -> list[list[int]]:
    """Return Stellarium constellation stick-figure polylines."""
    data = read_constellation_index(path)
    definitions: list[list[int]] = []
    for constellation_index, constellation in enumerate(
        data["constellations"],
        start=1,
    ):
        raw_lines = constellation.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError(
                f"{path}: constellation #{constellation_index} has no lines"
            )
        for line_index, raw_line in enumerate(raw_lines, start=1):
            if (
                not isinstance(raw_line, list)
                or len(raw_line) < 2
                or not all(isinstance(value, int) for value in raw_line)
            ):
                raise ValueError(
                    f"{path}: constellation #{constellation_index} line "
                    f"#{line_index} must contain at least two HIP integers"
                )
            definitions.append(raw_line)
    if not definitions:
        raise ValueError(f"No constellation stick figures found in {path}")
    return definitions


def read_hip_star_coordinates(
    path: Path = CONSTELLATION_INDEX_FILE,
) -> dict[int, tuple[float, float]]:
    """Return HIP coordinates embedded in ``assets/index.json``."""
    data = read_constellation_index(path)
    raw_stars = data.get("hip_stars")
    if not isinstance(raw_stars, dict) or not raw_stars:
        raise ValueError(
            f"{path} has no hip_stars section; cannot draw stick figures"
        )
    coordinates: dict[int, tuple[float, float]] = {}
    for raw_hip, raw_position in raw_stars.items():
        if not isinstance(raw_position, dict):
            raise ValueError(f"{path}: hip_stars[{raw_hip!r}] is not an object")
        try:
            hip = int(raw_hip)
            coordinates[hip] = (
                float(raw_position["ra_deg"]),
                float(raw_position["dec_deg"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: invalid coordinates for HIP {raw_hip!r}"
            ) from error
    return coordinates


def galactic_to_hammer_pixels(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Galactic coordinates to floating-point Hammer pixels."""
    centered = (
        np.asarray(longitude_deg, dtype=np.float64) + 180.0
    ) % 360.0 - 180.0
    longitude = -np.radians(centered)
    latitude = np.radians(np.asarray(latitude_deg, dtype=np.float64))
    cos_latitude = np.cos(latitude)
    half_longitude = longitude * 0.5
    denominator = np.sqrt(
        np.maximum(
            1.0 + cos_latitude * np.cos(half_longitude),
            np.finfo(np.float64).tiny,
        )
    )
    hammer_x = (
        2.0
        * np.sqrt(2.0)
        * cos_latitude
        * np.sin(half_longitude)
        / denominator
    )
    hammer_y = np.sqrt(2.0) * np.sin(latitude) / denominator
    pixel_x = (
        hammer_x / (2.0 * np.sqrt(2.0)) + 1.0
    ) * 0.5 * width
    pixel_y = (
        1.0 - hammer_y / np.sqrt(2.0)
    ) * 0.5 * height
    return (
        np.clip(pixel_x, 0.0, width - 1.0),
        np.clip(pixel_y, 0.0, height - 1.0),
    )


def _spherical_segment(
    start: np.ndarray,
    stop: np.ndarray,
    *,
    max_step_degrees: float,
) -> np.ndarray:
    dot = float(np.clip(np.dot(start, stop), -1.0, 1.0))
    angle = math.acos(dot)
    steps = max(
        1,
        int(math.ceil(math.degrees(angle) / max_step_degrees)),
    )
    fractions = np.linspace(0.0, 1.0, steps + 1)
    if angle < 1.0e-9:
        points = (
            (1.0 - fractions[:, None]) * start
            + fractions[:, None] * stop
        )
    else:
        denominator = math.sin(angle)
        points = (
            np.sin((1.0 - fractions) * angle)[:, None] * start
            + np.sin(fractions * angle)[:, None] * stop
        ) / denominator
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _split_projection_seam(
    points: np.ndarray,
    *,
    width: int,
) -> list[np.ndarray]:
    if len(points) < 2:
        return []
    breaks = np.flatnonzero(np.abs(np.diff(points[:, 0])) > width * 0.25) + 1
    pieces = np.split(points, breaks)
    return [piece for piece in pieces if len(piece) >= 2]


def _load_stick_figure_vectors(
    index_path: Path,
) -> tuple[list[list[int]], dict[int, np.ndarray]]:
    definitions = read_constellation_line_definitions(index_path)
    equatorial = read_hip_star_coordinates(index_path)
    required_ids = sorted({star_id for line in definitions for star_id in line})
    missing = [star_id for star_id in required_ids if star_id not in equatorial]
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise KeyError(f"Missing HIP coordinates for {preview}{suffix}")

    ra = np.array([equatorial[star_id][0] for star_id in required_ids])
    dec = np.array([equatorial[star_id][1] for star_id in required_ids])
    galactic = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs").galactic
    longitude = np.radians(np.asarray(galactic.l.degree))
    latitude = np.radians(np.asarray(galactic.b.degree))
    cos_latitude = np.cos(latitude)
    vectors = np.column_stack(
        (
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        )
    )
    return definitions, dict(zip(required_ids, vectors, strict=True))


def load_constellation_polylines(
    width: int,
    height: int,
    *,
    index_path: Path = CONSTELLATION_INDEX_FILE,
    max_step_degrees: float = FULL_SKY_MAX_STEP_DEGREES,
) -> list[np.ndarray]:
    """Load constellation stick figures and return projected pixel polylines."""
    definitions, vector_by_id = _load_stick_figure_vectors(index_path)

    projected: list[np.ndarray] = []
    for star_ids in definitions:
        pieces: list[np.ndarray] = []
        for first, second in zip(star_ids, star_ids[1:]):
            segment = _spherical_segment(
                vector_by_id[first],
                vector_by_id[second],
                max_step_degrees=max_step_degrees,
            )
            if pieces:
                segment = segment[1:]
            pieces.append(segment)
        if not pieces:
            continue
        vectors_on_line = np.concatenate(pieces)
        line_longitude = np.degrees(
            np.arctan2(vectors_on_line[:, 1], vectors_on_line[:, 0])
        ) % 360.0
        line_latitude = np.degrees(
            np.arcsin(np.clip(vectors_on_line[:, 2], -1.0, 1.0))
        )
        pixel_x, pixel_y = galactic_to_hammer_pixels(
            line_longitude,
            line_latitude,
            width,
            height,
        )
        points = np.rint(np.column_stack((pixel_x, pixel_y))).astype(
            np.int32
        )
        projected.extend(_split_projection_seam(points, width=width))
    return projected


def load_constellation_region_polylines(
    width: int,
    height: int,
    *,
    l_center_deg: float,
    b_center_deg: float,
    l_width_deg: float,
    b_height_deg: float,
    index_path: Path = CONSTELLATION_INDEX_FILE,
    max_step_degrees: float = REGION_MAX_STEP_DEGREES,
) -> list[np.ndarray]:
    """Project stick figures into a rectangular Galactic-coordinate field."""
    if width < 1 or height < 1:
        raise ValueError("Constellation target dimensions must be positive")
    if not 0.0 < l_width_deg <= 360.0:
        raise ValueError("l_width_deg must be in the range (0, 360]")
    if not 0.0 < b_height_deg <= 180.0:
        raise ValueError("b_height_deg must be in the range (0, 180]")

    half_l = l_width_deg * 0.5
    b_min = b_center_deg - b_height_deg * 0.5
    b_max = b_center_deg + b_height_deg * 0.5
    definitions, vector_by_id = _load_stick_figure_vectors(index_path)
    projected: list[np.ndarray] = []

    for star_ids in definitions:
        pieces: list[np.ndarray] = []
        for first, second in zip(star_ids, star_ids[1:]):
            segment = _spherical_segment(
                vector_by_id[first],
                vector_by_id[second],
                max_step_degrees=max_step_degrees,
            )
            if pieces:
                segment = segment[1:]
            pieces.append(segment)
        if not pieces:
            continue

        vectors_on_line = np.concatenate(pieces)
        longitude = np.degrees(
            np.arctan2(vectors_on_line[:, 1], vectors_on_line[:, 0])
        ) % 360.0
        latitude = np.degrees(
            np.arcsin(np.clip(vectors_on_line[:, 2], -1.0, 1.0))
        )
        delta_l = (longitude - l_center_deg + 180.0) % 360.0 - 180.0
        inside = (
            (np.abs(delta_l) <= half_l)
            & (latitude >= b_min)
            & (latitude <= b_max)
        )
        if np.count_nonzero(inside) < 2:
            continue

        pixel_x = (half_l - delta_l) * (width / l_width_deg)
        pixel_y = (b_max - latitude) * (height / b_height_deg)
        points = np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)
        run_breaks = np.flatnonzero(np.diff(inside.astype(np.int8)) != 0) + 1
        for point_run, visibility_run in zip(
            np.split(points, run_breaks),
            np.split(inside, run_breaks),
            strict=True,
        ):
            if visibility_run[0] and len(point_run) >= 2:
                point_run[:, 0] = np.clip(point_run[:, 0], 0, width - 1)
                point_run[:, 1] = np.clip(point_run[:, 1], 0, height - 1)
                projected.extend(
                    _split_projection_seam(point_run, width=width)
                )
    return projected


def load_constellation_boundary_polylines(
    width: int,
    height: int,
    *,
    index_path: Path = CONSTELLATION_INDEX_FILE,
    max_step_degrees: float = FULL_SKY_MAX_STEP_DEGREES,
) -> list[np.ndarray]:
    """Load constellation boundary edges and return projected pixel polylines."""
    edges = read_constellation_edges(index_path)
    if not edges:
        raise ValueError(f"No constellation boundary edges found in {index_path}")
    endpoints = np.array(
        [(ra1, dec1) for ra1, dec1, _, _ in edges]
        + [(ra2, dec2) for _, _, ra2, dec2 in edges],
        dtype=np.float64,
    )
    galactic = SkyCoord(
        ra=endpoints[:, 0],
        dec=endpoints[:, 1],
        unit="deg",
        frame=FK4(equinox="B1875"),
    ).galactic
    longitude = np.radians(np.asarray(galactic.l.degree))
    latitude = np.radians(np.asarray(galactic.b.degree))
    cos_latitude = np.cos(latitude)
    vectors = np.column_stack(
        (
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        )
    )
    projected: list[np.ndarray] = []
    edge_count = len(edges)
    for edge_index in range(edge_count):
        vectors_on_line = _spherical_segment(
            vectors[edge_index],
            vectors[edge_index + edge_count],
            max_step_degrees=max_step_degrees,
        )
        line_longitude = np.degrees(
            np.arctan2(vectors_on_line[:, 1], vectors_on_line[:, 0])
        ) % 360.0
        line_latitude = np.degrees(
            np.arcsin(np.clip(vectors_on_line[:, 2], -1.0, 1.0))
        )
        pixel_x, pixel_y = galactic_to_hammer_pixels(
            line_longitude,
            line_latitude,
            width,
            height,
        )
        points = np.rint(np.column_stack((pixel_x, pixel_y))).astype(
            np.int32
        )
        projected.extend(_split_projection_seam(points, width=width))
    return projected


def draw_constellation_polylines(
    bgr: np.ndarray,
    polylines: list[np.ndarray],
    *,
    color_bgr: tuple[int, int, int] = CONSTELLATION_LINE_COLOR_BGR,
    opacity: float = CONSTELLATION_LINE_OPACITY,
    line_width: int | None = None,
) -> None:
    """Screen-blend anti-aliased constellation lines into a BGR raster."""
    import cv2

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Constellation target must be an 8-bit BGR image")
    height, width = bgr.shape[:2]
    thickness = (
        max(1, int(round(height / CONSTELLATION_LINE_REFERENCE_HEIGHT)))
        if line_width is None
        else max(1, int(line_width))
    )
    opacity = float(np.clip(opacity, 0.0, 1.0))
    color = np.asarray(color_bgr, dtype=np.float32) / 255.0

    for points in polylines:
        points = np.asarray(points, dtype=np.int32)
        if len(points) < 2:
            continue
        padding = thickness + CONSTELLATION_LINE_PADDING_PIXELS
        x0 = max(0, int(points[:, 0].min()) - padding)
        x1 = min(width, int(points[:, 0].max()) + padding + 1)
        y0 = max(0, int(points[:, 1].min()) - padding)
        y1 = min(height, int(points[:, 1].max()) + padding + 1)
        if x1 <= x0 or y1 <= y0:
            continue
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
        coverage = (
            np.asarray(mask, dtype=np.float32) * (opacity / 255.0)
        )
        region = bgr[y0:y1, x0:x1]
        for channel in range(3):
            values = np.asarray(region[:, :, channel], dtype=np.float32)
            values += (
                (255.0 - values)
                * coverage
                * color[channel]
            )
            region[:, :, channel] = np.clip(
                values + 0.5,
                0.0,
                255.0,
            ).astype(np.uint8)


def add_constellations_to_png(
    input_path: Path,
    output_path: Path,
    polylines: list[np.ndarray],
    *,
    png_compression: int = DEFAULT_PNG_COMPRESSION,
) -> None:
    """Read a PNG, add constellation lines, and atomically save a copy."""
    import cv2

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve(strict=False) == output_path.resolve(strict=False):
        raise ValueError(
            "Constellation output must not overwrite the clean base map"
        )
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {input_path}")
    draw_constellation_polylines(image, polylines)
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
