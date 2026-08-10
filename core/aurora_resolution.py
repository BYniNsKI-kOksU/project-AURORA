"""Shared output-resolution profiles for AURORA renderers.

The sky products use a 2:1 Hammer canvas.  Keeping the dimensions and all
resolution-dependent names here prevents a 16K cache from being accidentally
reused for an 8K, 32K or 64K render.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.aurora_console import console


@dataclass(frozen=True)
class ResolutionProfile:
    """One supported AURORA sky/output resolution."""

    k: int
    width: int
    height: int

    @property
    def tag(self) -> str:
        return f"{self.k}k"

    @property
    def bins_l(self) -> int:
        return self.width

    @property
    def bins_b(self) -> int:
        return self.height

    @property
    def histogram_bytes(self) -> int:
        """Three float32 histogram planes (flux + weighted temperature + count)."""
        return self.width * self.height * 3 * 4

    @property
    def hammer_map_name(self) -> str:
        return f"aurora_sky_map_hammer_{self.tag}.png"

    @property
    def hammer_constellation_map_name(self) -> str:
        return f"aurora_sky_map_hammer_{self.tag}_constellations.png"

    def hammer_background_map_name(self, background: object) -> str:
        suffix = sky_map_background_suffix(background)
        return f"aurora_sky_map_hammer_{self.tag}{suffix}.png"

    @property
    def hammer_layout_name(self) -> str:
        return f"aurora_sky_map_hammer_{self.tag}_layout.npz"

    @property
    def hammer_rgb_cache_name(self) -> str:
        return f"aurora_rgb_projected_{self.tag}.npy"

    @property
    def region_map_name(self) -> str:
        return f"aurora_sky_region_rect_pic1_{self.tag}.png"

    @property
    def region_constellation_map_name(self) -> str:
        return f"aurora_sky_region_rect_pic1_{self.tag}_constellations.png"

    def region_background_map_name(self, background: object) -> str:
        suffix = sky_map_background_suffix(background)
        return f"aurora_sky_region_rect_pic1_{self.tag}{suffix}.png"

    @property
    def region_layout_name(self) -> str:
        return f"aurora_sky_region_{self.tag}_layout.npz"

    @property
    def region_rgb_cache_name(self) -> str:
        return f"aurora_sky_region_rgb_{self.tag}.npy"


RESOLUTION_PROFILES = {
    8: ResolutionProfile(8, 8_192, 4_096),
    16: ResolutionProfile(16, 16_384, 8_192),
    32: ResolutionProfile(32, 32_768, 16_384),
    64: ResolutionProfile(64, 65_536, 32_768),
}
DEFAULT_RESOLUTION_K = 16
SKY_MAP_MODE_ALIASES = {
    "1": "full",
    "full": "full",
    "all": "full",
    "whole": "full",
    "cale": "full",
    "całe": "full",
    "calosc": "full",
    "całość": "full",
    "2": "region",
    "region": "region",
    "regional": "region",
    "fragment": "region",
}
SKY_MAP_BACKGROUND_ALIASES = {
    "1": "plain",
    "plain": "plain",
    "normal": "plain",
    "zwykla": "plain",
    "zwykła": "plain",
    "2": "constellations",
    "constellations": "constellations",
    "constellation": "constellations",
    "konstelacje": "constellations",
    "gwiazdozbiory": "constellations",
    "3": "coordinates",
    "coordinates": "coordinates",
    "coordinate_grid": "coordinates",
    "grid": "coordinates",
    "siatka": "coordinates",
    "wspolrzedne": "coordinates",
    "współrzędne": "coordinates",
    "4": "poland_limits",
    "poland": "poland_limits",
    "poland_limits": "poland_limits",
    "granice": "poland_limits",
    "granice_polski": "poland_limits",
    "5": "constellations_coordinates",
    "constellations_coordinates": "constellations_coordinates",
    "6": "constellations_poland_limits",
    "constellations_poland_limits": "constellations_poland_limits",
    "7": "coordinates_poland_limits",
    "coordinates_poland_limits": "coordinates_poland_limits",
    "8": "constellations_coordinates_poland_limits",
    "constellations_coordinates_poland_limits": (
        "constellations_coordinates_poland_limits"
    ),
    "all_overlays": "constellations_coordinates_poland_limits",
    "all_lines": "constellations_coordinates_poland_limits",
}
SKY_MAP_BACKGROUND_FLAGS = {
    "plain": (False, False, False),
    "constellations": (True, False, False),
    "coordinates": (False, True, False),
    "poland_limits": (False, False, True),
    "constellations_coordinates": (True, True, False),
    "constellations_poland_limits": (True, False, True),
    "coordinates_poland_limits": (False, True, True),
    "constellations_coordinates_poland_limits": (True, True, True),
}
SKY_MAP_BACKGROUND_NUMBERS = {
    background: str(index)
    for index, background in enumerate(SKY_MAP_BACKGROUND_FLAGS, start=1)
}
SKY_MAP_BACKGROUND_MENU = (
    ("plain", "plain"),
    ("constellations", "constellations"),
    ("coordinates", "coordinate grid"),
    ("poland_limits", "Poland limits"),
    ("constellations_coordinates", "constellations + coordinates"),
    ("constellations_poland_limits", "constellations + Poland limits"),
    ("coordinates_poland_limits", "coordinates + Poland limits"),
    (
        "constellations_coordinates_poland_limits",
        "constellations + coordinates + Poland limits",
    ),
)


def get_resolution(value: object, *, default: int | None = None) -> ResolutionProfile:
    """Return a profile for ``8``, ``16``, ``32`` or ``64`` (optionally ``k``)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is None:
            raise ValueError("resolution is required")
        value = default
    text = str(value).strip().lower().removesuffix("k")
    try:
        key = int(text)
    except ValueError as exc:
        raise ValueError("choose 8, 16, 32 or 64") from exc
    try:
        return RESOLUTION_PROFILES[key]
    except KeyError as exc:
        raise ValueError("choose 8, 16, 32 or 64") from exc


def prompt_resolution(*, default: int = DEFAULT_RESOLUTION_K) -> ResolutionProfile:
    """Ask for a resolution when attached to a terminal.

    ``AURORA_RESOLUTION_K`` is useful for batch jobs.  Non-interactive
    processes use the safe 16K default so imports and scheduled jobs never
    block waiting for stdin.
    """
    def _announce(profile: ResolutionProfile) -> ResolutionProfile:
        if profile.k >= 32:
            console.warning(
                "This profile requires about "
                f"{profile.histogram_bytes / 2**30:.1f} GiB "
                "for three histograms; treat 64K as a viewing-only profile."
            )
        return profile

    # On macOS and Windows multiprocessing uses ``spawn``.  Spawned workers
    # import the renderer module again; they must never open a second prompt.
    is_main_process = mp.current_process().name == "MainProcess"
    env_value = os.environ.get("AURORA_RESOLUTION_K")
    if not is_main_process:
        return get_resolution(env_value or default)
    if env_value:
        return _announce(get_resolution(env_value))
    if not sys.stdin.isatty():
        return _announce(get_resolution(default))

    default_profile = get_resolution(default)
    while True:
        try:
            response = console.prompt(
                "AURORA resolution [8/16/32/64] "
                f"(Enter = {default_profile.k}K)"
            ).strip()
        except EOFError:
            return _announce(default_profile)
        if not response:
            return _announce(default_profile)
        try:
            return _announce(get_resolution(response))
        except ValueError:
            console.warning("Enter only 8, 16, 32, or 64")


def get_sky_map_mode(value: object, *, default: str = "full") -> str:
    """Normalize an all-sky/region selection."""
    text = str(default if value is None else value).strip().casefold()
    if not text:
        text = str(default).strip().casefold()
    try:
        return SKY_MAP_MODE_ALIASES[text]
    except KeyError as exc:
        raise ValueError("choose full/all-sky or region") from exc


def get_sky_map_background(
    value: object,
    *,
    default: str = "plain",
) -> str:
    """Normalize one of the supported static background-map variants."""
    text = str(default if value is None else value).strip().casefold()
    if not text:
        text = str(default).strip().casefold()
    try:
        return SKY_MAP_BACKGROUND_ALIASES[text]
    except KeyError as exc:
        raise ValueError("choose a background-map variant from 1 to 8") from exc


def sky_map_background_from_flags(
    *,
    constellations: bool,
    coordinates: bool,
    poland_limits: bool,
) -> str:
    """Return the canonical background name for three independent toggles."""
    selected = (
        bool(constellations),
        bool(coordinates),
        bool(poland_limits),
    )
    for background, flags in SKY_MAP_BACKGROUND_FLAGS.items():
        if flags == selected:
            return background
    raise RuntimeError(f"Unsupported sky-map background flags: {selected}")


def sky_map_background_flags(background: object) -> tuple[bool, bool, bool]:
    """Return constellation/grid/Poland-limit flags for a background."""
    return SKY_MAP_BACKGROUND_FLAGS[get_sky_map_background(background)]


def sky_map_background_suffix(background: object) -> str:
    """Return the stable filename suffix for a background variant."""
    normalized = get_sky_map_background(background)
    return "" if normalized == "plain" else f"_{normalized}"


def strip_sky_map_background_suffix(stem: str) -> str:
    """Remove any supported overlay suffix from a map filename stem."""
    normalized = str(stem)
    suffixes = sorted(
        (
            sky_map_background_suffix(background)
            for background in SKY_MAP_BACKGROUND_FLAGS
            if background != "plain"
        ),
        key=len,
        reverse=True,
    )
    for suffix in suffixes:
        if normalized.casefold().endswith(suffix.casefold()):
            return normalized[: -len(suffix)]
    return normalized


def available_sky_map_backgrounds(
    resolution: ResolutionProfile | object,
    maps_dir: Path | str,
) -> tuple[str, ...]:
    """Return background variants whose exact PNG files already exist."""
    profile = (
        resolution
        if isinstance(resolution, ResolutionProfile)
        else get_resolution(resolution)
    )
    directory = Path(maps_dir)
    return tuple(
        background
        for background, _label in SKY_MAP_BACKGROUND_MENU
        if (
            directory / profile.hammer_background_map_name(background)
        ).is_file()
    )


def _normalize_available_sky_map_backgrounds(
    values: Iterable[object],
) -> tuple[str, ...]:
    requested = {get_sky_map_background(value) for value in values}
    return tuple(
        background
        for background, _label in SKY_MAP_BACKGROUND_MENU
        if background in requested
    )


def prompt_sky_map_background(
    *,
    default: str = "plain",
    available_backgrounds: Iterable[object] | None = None,
) -> str:
    """Ask which full-sky map variant should be used as video background."""
    default_background = get_sky_map_background(default)
    available = (
        tuple(background for background, _label in SKY_MAP_BACKGROUND_MENU)
        if available_backgrounds is None
        else _normalize_available_sky_map_backgrounds(available_backgrounds)
    )
    if not available:
        raise FileNotFoundError("No full-sky background maps are available")

    if default_background not in available:
        default_background = available[0]

    def _require_available(background: str) -> str:
        if background not in available:
            available_numbers = ", ".join(
                SKY_MAP_BACKGROUND_NUMBERS[item] for item in available
            )
            raise ValueError(
                f"background map {background!r} is not available; "
                f"choose one of: {available_numbers}"
            )
        return background

    is_main_process = mp.current_process().name == "MainProcess"
    env_value = os.environ.get("AURORA_SKY_MAP_BACKGROUND")

    if not is_main_process:
        return _require_available(
            get_sky_map_background(env_value, default=default_background)
        )
    if env_value:
        background = _require_available(
            get_sky_map_background(
                env_value,
                default=default_background,
            )
        )
        os.environ["AURORA_SKY_MAP_BACKGROUND"] = background
        return background
    if not sys.stdin.isatty():
        os.environ["AURORA_SKY_MAP_BACKGROUND"] = default_background
        return default_background

    default_number = SKY_MAP_BACKGROUND_NUMBERS[default_background]
    while True:
        console.print("Available background maps:")
        for background, label in SKY_MAP_BACKGROUND_MENU:
            if background not in available:
                continue
            console.print(
                f"  {SKY_MAP_BACKGROUND_NUMBERS[background]} = {label}"
            )
        available_numbers = ",".join(
            SKY_MAP_BACKGROUND_NUMBERS[background]
            for background in available
        )
        try:
            response = console.prompt(
                f"Background map [{available_numbers}] "
                f"(Enter = {default_number})"
            ).strip()
        except EOFError:
            response = ""
        try:
            background = _require_available(
                get_sky_map_background(
                    response,
                    default=default_background,
                )
            )
        except ValueError:
            console.warning(
                "Enter one of the displayed background-map numbers"
            )
            continue
        os.environ["AURORA_SKY_MAP_BACKGROUND"] = background
        return background


def prompt_sky_map_mode(*, default: str = "full") -> str:
    """Ask whether an animation should use the full sky or a region.

    ``AURORA_SKY_MAP_MODE`` remains a non-interactive override. The selected
    value is written back to the environment so spawned workers inherit it and
    never display a second prompt.
    """
    default_mode = get_sky_map_mode(default)
    is_main_process = mp.current_process().name == "MainProcess"
    env_value = os.environ.get("AURORA_SKY_MAP_MODE")

    if not is_main_process:
        return get_sky_map_mode(env_value, default=default_mode)
    if env_value:
        mode = get_sky_map_mode(env_value, default=default_mode)
        os.environ["AURORA_SKY_MAP_MODE"] = mode
        return mode
    if not sys.stdin.isatty():
        os.environ["AURORA_SKY_MAP_MODE"] = default_mode
        return default_mode

    default_number = "1" if default_mode == "full" else "2"
    while True:
        try:
            response = console.prompt(
                "Sky-map coverage [1 = full sky, 2 = region] "
                f"(Enter = {default_number})"
            ).strip()
        except EOFError:
            response = ""
        try:
            mode = get_sky_map_mode(response, default=default_mode)
        except ValueError:
            console.warning("Enter 1 for the full sky or 2 for a region")
            continue
        os.environ["AURORA_SKY_MAP_MODE"] = mode
        return mode
