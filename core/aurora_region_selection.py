"""Discover and interactively select matching AURORA region map/layout pairs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np

from core.aurora_console import console
from core.aurora_paths import REGION_MAPS_DIR
from core.aurora_render_core import (
    RectangularSkyRegion,
    load_rectangular_sky_region,
)
from core.aurora_resolution import strip_sky_map_background_suffix


@dataclass(frozen=True)
class SkyRegionSelection:
    """One validated rectangular layout and its matching PNG background."""

    map_path: Path
    layout_path: Path
    region: RectangularSkyRegion


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        return None
    return (
        int.from_bytes(header[16:20], "big"),
        int.from_bytes(header[20:24], "big"),
    )


def _layout_map_filename(path: Path) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as layout:
            if "map_filename" not in layout.files:
                return None
            value = str(
                np.asarray(layout["map_filename"]).reshape(-1)[0]
            ).strip()
            return value or None
    except (OSError, TypeError, ValueError):
        return None


def _usable_project_file(path: Path) -> bool:
    ignored = {".git", ".history", "__pycache__"}
    return path.is_file() and not ignored.intersection(path.parts)


def _candidate_layouts(default_layout: Path) -> list[Path]:
    paths = [Path(default_layout)]
    if REGION_MAPS_DIR.is_dir():
        paths.extend(REGION_MAPS_DIR.glob("*layout*.npz"))
    return sorted(
        {
            path.resolve(strict=False)
            for path in paths
            if _usable_project_file(path)
        },
        key=lambda path: str(path).casefold(),
    )


def _candidate_maps(
    default_map: Path,
    layout_paths: list[Path] | tuple[Path, ...] = (),
) -> list[Path]:
    paths = [Path(default_map)]
    if REGION_MAPS_DIR.is_dir():
        paths.extend(REGION_MAPS_DIR.glob("*.png"))
    for layout_path in layout_paths:
        map_filename = _layout_map_filename(layout_path)
        if not map_filename:
            continue
        metadata_path = Path(map_filename).expanduser()
        if metadata_path.is_absolute():
            try:
                metadata_path.resolve(strict=False).relative_to(
                    REGION_MAPS_DIR.resolve(strict=False)
                )
            except ValueError:
                continue
            paths.append(metadata_path)
            continue
        paths.extend(
            (
                REGION_MAPS_DIR / metadata_path,
                layout_path.parent / metadata_path,
            )
        )
    return sorted(
        {
            path.resolve(strict=False)
            for path in paths
            if _usable_project_file(path)
        },
        key=lambda path: str(path).casefold(),
    )


def _name_key(path: Path) -> str:
    key = path.stem.casefold()
    if key.endswith("_layout"):
        key = key[: -len("_layout")]
    return strip_sky_map_background_suffix(key)


def _pair_score(
    map_path: Path,
    layout_path: Path,
) -> tuple[int, int] | None:
    metadata_name = _layout_map_filename(layout_path)
    if metadata_name and Path(metadata_name).name == map_path.name:
        return (0, 0)
    if metadata_name and _name_key(Path(metadata_name)) == _name_key(map_path):
        return (0, 1)
    map_key = _name_key(map_path)
    layout_key = _name_key(layout_path)
    if map_key == layout_key:
        return (1, 0)
    return None


def discover_sky_regions(
    default_map: Path,
    default_layout: Path,
) -> list[SkyRegionSelection]:
    """Return validated region choices, pairing files by metadata and names."""
    default_map = Path(default_map)
    default_layout = Path(default_layout)
    layout_paths = _candidate_layouts(default_layout)
    layouts: list[tuple[Path, RectangularSkyRegion]] = []
    for layout_path in layout_paths:
        try:
            layouts.append(
                (layout_path, load_rectangular_sky_region(layout_path))
            )
        except ValueError:
            continue

    choices: list[SkyRegionSelection] = []
    map_paths = _candidate_maps(default_map, layout_paths)
    for map_path in map_paths:
        dimensions = _png_dimensions(map_path)
        compatible = [
            (layout_path, region)
            for layout_path, region in layouts
            if dimensions == (region.width, region.height)
        ]
        if not compatible:
            continue
        scored = [
            (
                _pair_score(
                    map_path,
                    layout_path,
                ),
                layout_path,
                region,
            )
            for layout_path, region in compatible
        ]
        scored = [item for item in scored if item[0] is not None]
        if scored:
            scored.sort(key=lambda item: item[0])
            _, layout_path, region = scored[0]
        else:
            continue
        choices.append(
            SkyRegionSelection(
                map_path=map_path,
                layout_path=layout_path,
                region=region,
            )
        )
    choices.sort(
        key=lambda choice: (
            choice.map_path.name.casefold(),
            choice.layout_path.name.casefold(),
        )
    )
    return choices


def _discovery_error(default_map: Path, default_layout: Path) -> str:
    layout_paths = _candidate_layouts(default_layout)
    map_paths = _candidate_maps(default_map, layout_paths)
    valid_layouts = []
    for path in layout_paths:
        try:
            region = load_rectangular_sky_region(path)
        except ValueError:
            continue
        valid_layouts.append((path, region))
    map_preview = ", ".join(path.name for path in map_paths[:8]) or "none"
    layout_preview = (
        ", ".join(
            f"{path.name} [l={region.l_center_deg:g}°, "
            f"b={region.b_center_deg:g}°, "
            f"{region.l_width_deg:g}°×{region.b_height_deg:g}°]"
            for path, region in valid_layouts[:8]
        )
        or "none"
    )
    return (
        "No matching sky-region PNG/layout pairs were found. "
        f"Region PNG files found: {map_preview}. "
        f"Valid rectangular layouts found: {layout_preview}. "
        f"Expected defaults: {default_map} and {default_layout}. "
        f"Only {REGION_MAPS_DIR} is scanned. The PNG and NPZ must describe "
        "the same named region and dimensions. Run "
        "main/aurora_sky_region_render.py if no valid layout exists."
    )


def _validate_explicit_pair(
    map_path: Path,
    layout_path: Path,
) -> SkyRegionSelection:
    map_path = Path(map_path).expanduser().resolve(strict=False)
    layout_path = Path(layout_path).expanduser().resolve(strict=False)
    region_root = REGION_MAPS_DIR.resolve(strict=False)
    for candidate in (map_path, layout_path):
        try:
            candidate.relative_to(region_root)
        except ValueError as exc:
            raise ValueError(
                f"Region files must be stored under {REGION_MAPS_DIR}: "
                f"{candidate}"
            ) from exc
    if not map_path.is_file():
        raise FileNotFoundError(f"Missing sky-region map: {map_path}")
    if not layout_path.is_file():
        raise FileNotFoundError(f"Missing sky-region layout: {layout_path}")
    region = load_rectangular_sky_region(layout_path)
    metadata_name = _layout_map_filename(layout_path)
    if not metadata_name or _name_key(Path(metadata_name)) != _name_key(map_path):
        raise ValueError(
            f"Region layout does not identify the selected map: {layout_path}"
        )
    dimensions = _png_dimensions(map_path)
    if dimensions != (region.width, region.height):
        raise ValueError(
            f"Region map dimensions {dimensions} do not match layout "
            f"{region.width} × {region.height}: {map_path}"
        )
    return SkyRegionSelection(map_path, layout_path, region)


def select_sky_region(
    default_map: Path,
    default_layout: Path,
) -> SkyRegionSelection:
    """Select one explicit region pair from the canonical region directory."""
    env_map = os.environ.get("AURORA_REGION_MAP")
    env_layout = os.environ.get("AURORA_REGION_LAYOUT")
    if env_map or env_layout:
        if not env_map or not env_layout:
            raise RuntimeError(
                "Set both AURORA_REGION_MAP and AURORA_REGION_LAYOUT; "
                "a region PNG cannot be selected without its matching layout"
            )
        selection = _validate_explicit_pair(
            Path(env_map),
            Path(env_layout),
        )
    else:
        choices = discover_sky_regions(default_map, default_layout)
        if not choices:
            raise FileNotFoundError(
                _discovery_error(default_map, default_layout)
            )
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Sky-region selection requires an interactive terminal. "
                "For a batch render, set both AURORA_REGION_MAP and "
                "AURORA_REGION_LAYOUT explicitly."
            )
        console.section("Available sky regions")
        for index, choice in enumerate(choices, start=1):
            region = choice.region
            console.print(f"  {index:>2}. {choice.map_path.name}")
            console.detail(
                f"l={region.l_center_deg:g}°, b={region.b_center_deg:g}°, "
                f"field={region.l_width_deg:g}° × "
                f"{region.b_height_deg:g}°"
            )
            console.detail(f"Layout: {choice.layout_path.name}")
        while True:
            response = console.prompt(
                f"Select sky region [1-{len(choices)}] (Enter = 1)"
            ).strip()
            if not response:
                selection = choices[0]
                break
            try:
                selected_index = int(response)
            except ValueError:
                selected_index = 0
            if 1 <= selected_index <= len(choices):
                selection = choices[selected_index - 1]
                break
            console.warning(f"Enter a number from 1 to {len(choices)}")

    os.environ["AURORA_REGION_MAP"] = str(selection.map_path)
    os.environ["AURORA_REGION_LAYOUT"] = str(selection.layout_path)
    region = selection.region
    console.success(f"Selected sky region: {selection.map_path.name}")
    console.detail(
        f"Galactic field: l={region.l_center_deg:g}°, "
        f"b={region.b_center_deg:g}°, "
        f"{region.l_width_deg:g}° × {region.b_height_deg:g}°"
    )
    console.detail(f"Layout: {selection.layout_path}")
    return selection
