"""AURORA-compatible full-sky, region, image, and catalogue backgrounds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from core.aurora_paths import map_path, region_map_path
from core.aurora_region_selection import SkyRegionSelection, discover_sky_regions
from core.aurora_render_core import (
    draw_gaussians_u8,
    galactic_to_hammer_pixel,
    galactic_to_region_pixel,
    load_rectangular_sky_region,
    temperature_to_rgb,
)
from core.aurora_resolution import get_resolution, get_sky_map_background


BACKGROUND_MODES = ("all_sky", "region", "custom", "catalog")


def internal_galactic_radians(longitude_deg, latitude_deg):
    longitude = np.radians(np.asarray(longitude_deg, dtype=np.float32))
    longitude = -np.where(longitude > np.pi, longitude - 2.0 * np.pi, longitude)
    return longitude.astype(np.float32), np.radians(latitude_deg).astype(np.float32)


@dataclass(frozen=True)
class BackgroundConfig:
    mode: str = "all_sky"
    image_path: Path | None = None
    layout_path: Path | None = None
    catalog_path: Path | None = None
    aurora_resolution: str = "16k"
    constellations: bool = False
    sky_map_background: str = "plain"
    region_name: str | None = None
    region_index: int = 0

    def __post_init__(self) -> None:
        if self.mode not in BACKGROUND_MODES:
            raise ValueError(f"background mode must be one of {', '.join(BACKGROUND_MODES)}")
        for name in ("image_path", "layout_path", "catalog_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if self.region_index < 0:
            raise ValueError("region_index cannot be negative")
        object.__setattr__(
            self,
            "sky_map_background",
            get_sky_map_background(self.sky_map_background),
        )


@dataclass(frozen=True)
class BackgroundFrame:
    pixels: np.ndarray
    target_x: float
    target_y: float


def project_galactic_position(
    config: BackgroundConfig,
    width: int,
    height: int,
    longitude_deg: float,
    latitude_deg: float,
) -> tuple[float, float, bool]:
    """Project one event onto a resolved full-sky or regional background."""
    config = resolve_background(config)
    longitude, latitude = internal_galactic_radians(
        np.asarray([longitude_deg]), np.asarray([latitude_deg])
    )
    if config.mode == "region":
        region = load_rectangular_sky_region(config.layout_path)
        px, py, visible = galactic_to_region_pixel(
            longitude, latitude, width, height, region
        )
        return float(px[0]), float(py[0]), bool(visible[0])
    px, py = galactic_to_hammer_pixel(longitude, latitude, width, height)
    return float(px[0]), float(py[0]), True


def image_dimensions(path: Path) -> tuple[int, int]:
    """Read source dimensions without decoding or resampling the image."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"background does not exist: {path}")
    with path.open("rb") as handle:
        header = handle.read(24)
    if (
        len(header) == 24
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR"
    ):
        return (
            int.from_bytes(header[16:20], "big"),
            int.from_bytes(header[20:24], "big"),
        )
    if shutil.which("ffprobe") is None:
        raise RuntimeError("FFprobe is required to inspect a non-PNG background")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"background has no image stream: {path}")
    return int(streams[0]["width"]), int(streams[0]["height"])


def _region_defaults(config: BackgroundConfig) -> tuple[Path, Path]:
    profile = get_resolution(config.aurora_resolution)
    background = config.sky_map_background
    if background == "plain" and config.constellations:
        background = "constellations"
    map_name_method = getattr(profile, "region_background_map_name", None)
    if callable(map_name_method):
        map_name = map_name_method(background)
    else:
        suffix = "" if background == "plain" else f"_{background}"
        map_name = f"aurora_sky_region_rect_pic1_{profile.tag}{suffix}.png"
    return region_map_path(map_name), region_map_path(profile.region_layout_name)


def resolve_region(config: BackgroundConfig) -> SkyRegionSelection:
    """Resolve one validated PNG/NPZ region pair from settings in the code."""
    if config.mode != "region":
        raise ValueError("resolve_region requires background mode 'region'")
    if (config.image_path is None) != (config.layout_path is None):
        raise ValueError(
            "region background requires both image_path and layout_path, or neither"
        )
    default_map, default_layout = _region_defaults(config)
    search_map = config.image_path or default_map
    search_layout = config.layout_path or default_layout
    choices = discover_sky_regions(search_map, search_layout)
    available_names = [choice.map_path.stem for choice in choices]
    if config.image_path is not None:
        wanted_map = config.image_path.expanduser().resolve(strict=False)
        wanted_layout = config.layout_path.expanduser().resolve(strict=False)
        choices = [
            choice
            for choice in choices
            if choice.map_path.resolve(strict=False) == wanted_map
            and choice.layout_path.resolve(strict=False) == wanted_layout
        ]
    elif not config.region_name:
        marker = "_constellations"
        matching_style = [
            choice
            for choice in choices
            if (marker in choice.map_path.stem.casefold()) == config.constellations
        ]
        choices = matching_style or choices
    if config.region_name:
        wanted = config.region_name.casefold().strip()
        exact = [
            choice for choice in choices
            if wanted in {choice.map_path.name.casefold(), choice.map_path.stem.casefold()}
        ]
        choices = exact or [
            choice for choice in choices if wanted in choice.map_path.stem.casefold()
        ]
    if not choices:
        selection = f" named {config.region_name!r}" if config.region_name else ""
        available = ", ".join(available_names) or "none"
        raise FileNotFoundError(
            f"no matching AURORA sky-region PNG/NPZ pair{selection}; "
            f"available regions: {available}; expected files below {region_map_path('')}"
        )
    if config.region_index >= len(choices):
        raise IndexError(
            f"region_index={config.region_index} but only {len(choices)} region(s) match"
        )
    return choices[config.region_index]


def resolve_background(config: BackgroundConfig) -> BackgroundConfig:
    """Fill canonical AURORA paths while preserving the requested mode."""
    profile = get_resolution(config.aurora_resolution)
    if config.mode == "region":
        selection = resolve_region(config)
        return replace(
            config,
            image_path=selection.map_path,
            layout_path=selection.layout_path,
        )
    if config.mode == "all_sky" and config.image_path is None:
        background = config.sky_map_background
        if background == "plain" and config.constellations:
            background = "constellations"
        name = profile.hammer_background_map_name(background)
        return replace(config, image_path=map_path(name))
    return config


def background_native_dimensions(config: BackgroundConfig) -> tuple[int, int]:
    """Return the native video canvas size for the selected background."""
    resolved = resolve_background(config)
    if resolved.mode == "catalog":
        profile = get_resolution(resolved.aurora_resolution)
        return profile.width, profile.height
    if resolved.image_path is None:
        raise ValueError(f"{resolved.mode} background requires an image")
    dimensions = image_dimensions(resolved.image_path)
    if resolved.mode == "region":
        region = load_rectangular_sky_region(resolved.layout_path)
        if dimensions != (region.width, region.height):
            raise ValueError(
                f"region image dimensions {dimensions} do not match layout "
                f"{region.width} x {region.height}"
            )
    return dimensions


def _load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"background does not exist: {path}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required to load the video background")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
    ]
    if image_dimensions(path) != (width, height):
        command.extend(["-vf", f"scale={width}:{height}:flags=lanczos"])
    command.extend([
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ])
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    expected_size = width * height * 3
    if len(result.stdout) != expected_size:
        raise RuntimeError(
            f"FFmpeg returned {len(result.stdout)} background bytes; "
            f"expected {expected_size}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3).copy()


def _read_catalog(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"star catalog does not exist: {path}")
    if path.suffix.casefold() in {".fits", ".fit", ".fts"}:
        from astropy.table import Table

        table = Table.read(path)
        names = set(table.colnames)
        get = lambda name: np.asarray(table[name])
    else:
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = set(table.dtype.names or ())
        get = lambda name: np.asarray(table[name])
    aliases = {
        "l": ("l", "galactic_longitude_deg", "longitude_deg"),
        "b": ("b", "galactic_latitude_deg", "latitude_deg"),
    }
    result: dict[str, np.ndarray] = {}
    for key, candidates in aliases.items():
        selected = next((name for name in candidates if name in names), None)
        if selected is None:
            raise ValueError(f"catalog requires a {key!r} Galactic coordinate column")
        result[key] = get(selected).astype(np.float32)
    result["magnitude"] = (
        get("magnitude").astype(np.float32) if "magnitude" in names else np.full(result["l"].shape, 12.0, np.float32)
    )
    result["temperature_k"] = (
        get("temperature_k").astype(np.float32) if "temperature_k" in names else np.full(result["l"].shape, 5800.0, np.float32)
    )
    return result


def load_background(
    config: BackgroundConfig,
    width: int,
    height: int,
    *,
    target_longitude_deg: float,
    target_latitude_deg: float,
) -> BackgroundFrame:
    """Load/render a background and project the event onto it."""
    config = resolve_background(config)
    target_x, target_y, target_visible = project_galactic_position(
        config,
        width,
        height,
        target_longitude_deg,
        target_latitude_deg,
    )
    if config.mode == "region":
        image_path = config.image_path
        layout_path = config.layout_path
        region = load_rectangular_sky_region(layout_path)
        pixels = _load_rgb(image_path, width, height)
        if not target_visible:
            raise ValueError("the progenitor is outside the selected sky region")
    else:
        if config.mode == "all_sky":
            image_path = config.image_path
            pixels = _load_rgb(image_path, width, height)
        elif config.mode == "custom":
            if config.image_path is None:
                raise ValueError("custom background requires image_path")
            pixels = _load_rgb(config.image_path, width, height)
        else:
            if config.catalog_path is None:
                raise ValueError("catalog background requires catalog_path")
            catalog = _read_catalog(config.catalog_path)
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            lon, lat = internal_galactic_radians(catalog["l"], catalog["b"])
            star_x, star_y = galactic_to_hammer_pixel(lon, lat, width, height)
            scale = height / 8192.0
            sigma = np.clip((14.5 - catalog["magnitude"]) * 0.35, 0.7, 3.5) * max(scale, 0.2)
            alpha = np.clip(np.power(10.0, -0.16 * (catalog["magnitude"] - 8.0)), 0.08, 0.9)
            colors = temperature_to_rgb(catalog["temperature_k"])
            draw_gaussians_u8(pixels, star_x, star_y, sigma, alpha, colors)
    return BackgroundFrame(np.ascontiguousarray(pixels), target_x, target_y)
