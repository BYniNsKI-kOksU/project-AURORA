"""
Render one or more all-sky intensity maps in the Hammer projection.

Unlike ``aurora_sky_render.py``, this program is not tied to Gaia columns or
visible-light stellar colours.  A JSON file describes independent radio,
microwave, infrared, optical, ultraviolet, X-ray, gamma-ray, or custom maps.
The input adapter understands:

* FITS images with celestial WCS;
* FITS HEALPix tables in RING or NESTED ordering;
* point catalogues stored as FITS binary tables, CSV, or TSV;
* equirectangular NumPy arrays and common raster image formats.

Every selected layer is written to a separate PNG.  When several layers are
selected, the program can additionally build a labelled, side-by-side collage;
the maps are never blended with one another.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import traceback
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_constellations import (
    CONSTELLATION_INDEX_FILE,
    DEFAULT_CONSTELLATION_MODE,
    draw_constellation_polylines,
    load_constellation_polylines,
    prompt_constellation_overlay,
)
from core.aurora_paths import MAPS_DIR, asset_path
from core.aurora_render_core import find_fits_table_hdu
from core.aurora_resolution import DEFAULT_RESOLUTION_K, prompt_resolution


DEFAULT_CONFIG = asset_path("aurora_multiband_maps.json")
DEFAULT_WIDTH = 16384
DEFAULT_TILE_ROWS = 64
DEFAULT_SAMPLE_SIZE = 2_000_000
FINITE_SAMPLE_CHUNK_SIZE = 1_000_000
DEFAULT_FITS_CATALOG_CHUNK_ROWS = 1_000_000
DEFAULT_DELIMITED_CATALOG_CHUNK_ROWS = 250_000
DEFAULT_CATALOG_VALUE_MODE = "intensity"
DEFAULT_CATALOG_SMOOTHING_SIGMA = 1.2
DEFAULT_SCALE_FACTOR = 1.0
DEFAULT_OFFSET = 0.0
DEFAULT_STRETCH = "asinh"
DEFAULT_STRETCH_STRENGTH = 8.0
DEFAULT_GAMMA = 1.0
DEFAULT_PALETTE = "inferno"
DEFAULT_PNG_COMPRESSION = 6
DEFAULT_COLLAGE_TILE_WIDTH = 1600
DEFAULT_COLLAGE_GAP = 12
DEFAULT_COLLAGE_TITLE_HEIGHT = 72
DEFAULT_COLLAGE_MODE = "ask"
MEMORY = MemoryController.from_environment()

CATALOG_LONGITUDE_COLUMNS = (
    "l",
    "glon",
    "gal_lon",
    "galactic_longitude",
    "ra",
    "ra_icrs",
    "raj2000",
    "ra_j2000",
)
CATALOG_LATITUDE_COLUMNS = (
    "b",
    "glat",
    "gal_lat",
    "galactic_latitude",
    "dec",
    "dec_icrs",
    "dej2000",
    "dec_j2000",
)
CATALOG_VALUE_COLUMNS = (
    "intensity",
    "flux",
    "flux_density",
    "value",
    "signal",
    "temperature",
    "counts",
    "count",
    "magnitude",
    "mag",
)
HEALPIX_PIXEL_COLUMNS = ("pixel", "pix", "ipix", "index")
RASTER_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fz"}


def _is_fits_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return path.suffix.lower() in FITS_EXTENSIONS or lower_name.endswith(
        (".fits.gz", ".fit.gz", ".fts.gz")
    )


def _case_insensitive_name(names: list[str] | tuple[str, ...], requested: str):
    lookup = {str(name).lower(): str(name) for name in names}
    return lookup.get(str(requested).lower())


def _first_matching_name(
    names: list[str] | tuple[str, ...],
    candidates: tuple[str, ...],
):
    lookup = {str(name).lower(): str(name) for name in names}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    slug = slug.strip("._-")
    return slug or "map"


def _normalize_frame(frame: str | None) -> str:
    value = str(frame or "galactic").strip().lower()
    aliases = {
        "g": "galactic",
        "gal": "galactic",
        "galactic": "galactic",
        "c": "icrs",
        "celestial": "icrs",
        "equatorial": "icrs",
        "j2000": "icrs",
        "icrs": "icrs",
        "fk5": "fk5",
        "e": "geocentrictrueecliptic",
        "ecliptic": "geocentrictrueecliptic",
        "geocentrictrueecliptic": "geocentrictrueecliptic",
    }
    return aliases.get(value, value)


def _frame_from_header(header: fits.Header, fallback: str = "galactic") -> str:
    coordinate_code = str(
        header.get("COORDSYS", header.get("COORDTYPE", ""))
    ).strip()
    if coordinate_code:
        return _normalize_frame(coordinate_code)
    ctype = " ".join(
        str(header.get(key, "")).upper() for key in ("CTYPE1", "CTYPE2")
    )
    if "GLON" in ctype or "GLAT" in ctype:
        return "galactic"
    if "RA" in ctype or "DEC" in ctype:
        return "icrs"
    if "ELON" in ctype or "ELAT" in ctype:
        return "geocentrictrueecliptic"
    return _normalize_frame(fallback)


def _transform_from_galactic(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    target_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_frame = _normalize_frame(target_frame)
    if target_frame == "galactic":
        return longitude_deg, latitude_deg
    coordinates = SkyCoord(
        l=np.asarray(longitude_deg) * 1.0,
        b=np.asarray(latitude_deg) * 1.0,
        unit="deg",
        frame="galactic",
    ).transform_to(target_frame)
    spherical = coordinates.spherical
    return (
        np.asarray(spherical.lon.degree),
        np.asarray(spherical.lat.degree),
    )


def _transform_to_galactic(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    source_frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    source_frame = _normalize_frame(source_frame)
    if source_frame == "galactic":
        return longitude_deg, latitude_deg
    coordinates = SkyCoord(
        np.asarray(longitude_deg) * 1.0,
        np.asarray(latitude_deg) * 1.0,
        unit="deg",
        frame=source_frame,
    ).galactic
    return np.asarray(coordinates.l.degree), np.asarray(coordinates.b.degree)


def hammer_world_tile(
    y0: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Galactic longitude/latitude and validity for a Hammer tile."""
    x_hammer = (
        ((np.arange(width, dtype=np.float64) + 0.5) / width) * 2.0 - 1.0
    ) * (2.0 * np.sqrt(2.0))
    y_hammer = (
        1.0
        - ((np.arange(y0, y1, dtype=np.float64) + 0.5) / height) * 2.0
    )[:, None] * np.sqrt(2.0)
    x = x_hammer[None, :]
    z2 = 1.0 - np.square(x) / 16.0 - np.square(y_hammer) / 4.0
    valid = z2 >= 0.5
    z = np.sqrt(np.maximum(z2, 0.0))
    display_longitude = 2.0 * np.arctan2(
        z * x,
        2.0 * (2.0 * z * z - 1.0),
    )
    latitude = np.arcsin(np.clip(z * y_hammer, -1.0, 1.0))

    # Match aurora_sky_render.py: Galactic longitude grows to the left,
    # while l=0 remains at the centre of the ellipse.
    galactic_longitude = np.mod(-np.degrees(display_longitude), 360.0)
    galactic_latitude = np.broadcast_to(
        np.degrees(latitude),
        galactic_longitude.shape,
    )
    return galactic_longitude, galactic_latitude, valid


def hammer_valid_mask(width: int, height: int) -> np.ndarray:
    mask = np.empty((height, width), dtype=bool)
    for y0 in range(0, height, DEFAULT_TILE_ROWS):
        MEMORY.throttle()
        y1 = min(y0 + DEFAULT_TILE_ROWS, height)
        _, _, mask[y0:y1] = hammer_world_tile(y0, y1, width, height)
    return mask


def galactic_to_hammer_indices(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Galactic degrees directly to integer Hammer pixels."""
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
    pixel_x = np.floor(
        (hammer_x / (2.0 * np.sqrt(2.0)) + 1.0) * 0.5 * width
    ).astype(np.int64)
    pixel_y = np.floor(
        (1.0 - hammer_y / np.sqrt(2.0)) * 0.5 * height
    ).astype(np.int64)
    valid = (
        np.isfinite(longitude_deg)
        & np.isfinite(latitude_deg)
        & (latitude_deg >= -90.0)
        & (latitude_deg <= 90.0)
    )
    # The antipode (l=180°) and both poles lie exactly on the continuous
    # Hammer boundary. Assign them to the nearest edge pixel instead of
    # discarding otherwise valid catalogue rows.
    np.clip(pixel_x, 0, width - 1, out=pixel_x)
    np.clip(pixel_y, 0, height - 1, out=pixel_y)
    return pixel_x, pixel_y, valid


def healpix_ang2pix_ring(
    nside: int,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> np.ndarray:
    """Vectorised HEALPix angular-to-pixel conversion for RING ordering."""
    if nside < 1:
        raise ValueError("HEALPix NSIDE must be positive")
    longitude = np.mod(np.radians(longitude_deg), 2.0 * np.pi)
    z = np.sin(np.radians(latitude_deg))
    absolute_z = np.abs(z)
    tt = longitude / (0.5 * np.pi)
    result = np.empty(np.broadcast(z, tt).shape, dtype=np.int64)
    equatorial = absolute_z <= (2.0 / 3.0)
    polar = ~equatorial
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside

    if np.any(equatorial):
        temp1 = nside * (0.5 + tt[equatorial])
        temp2 = nside * z[equatorial] * 0.75
        ascending = np.floor(temp1 - temp2).astype(np.int64)
        descending = np.floor(temp1 + temp2).astype(np.int64)
        ring = nside + 1 + ascending - descending
        shift = 1 - (ring & 1)
        position = np.floor_divide(
            ascending + descending - nside + shift + 1,
            2,
        ) + 1
        position = (position - 1) % (4 * nside) + 1
        result[equatorial] = (
            ncap + (ring - 1) * 4 * nside + position - 1
        )

    if np.any(polar):
        local_tt = tt[polar]
        local_z = z[polar]
        local_absolute_z = absolute_z[polar]
        fractional = local_tt - np.floor(local_tt)
        radius = nside * np.sqrt(3.0 * (1.0 - local_absolute_z))
        ascending = np.floor(fractional * radius).astype(np.int64)
        descending = np.floor((1.0 - fractional) * radius).astype(np.int64)
        ring = ascending + descending + 1
        position = np.floor(local_tt * ring).astype(np.int64) + 1
        position = (position - 1) % (4 * ring) + 1
        north_index = 2 * ring * (ring - 1) + position - 1
        south_index = npix - 2 * ring * (ring + 1) + position - 1
        result[polar] = np.where(local_z >= 0.0, north_index, south_index)
    return result


def _spread_nested_bits(values: np.ndarray, bit_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    result = np.zeros(values.shape, dtype=np.int64)
    for bit in range(bit_count):
        result |= ((values >> bit) & 1) << (2 * bit)
    return result


def healpix_ang2pix_nested(
    nside: int,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> np.ndarray:
    """Vectorised HEALPix angular-to-pixel conversion for NESTED ordering."""
    if nside < 1 or (nside & (nside - 1)):
        raise ValueError("NESTED HEALPix NSIDE must be a power of two")
    longitude = np.mod(np.radians(longitude_deg), 2.0 * np.pi)
    z = np.sin(np.radians(latitude_deg))
    absolute_z = np.abs(z)
    tt = longitude / (0.5 * np.pi)
    face = np.empty(z.shape, dtype=np.int64)
    x_index = np.empty(z.shape, dtype=np.int64)
    y_index = np.empty(z.shape, dtype=np.int64)
    equatorial = absolute_z <= (2.0 / 3.0)
    polar = ~equatorial

    if np.any(equatorial):
        temp1 = nside * (0.5 + tt[equatorial])
        temp2 = nside * z[equatorial] * 0.75
        ascending = np.floor(temp1 - temp2).astype(np.int64)
        descending = np.floor(temp1 + temp2).astype(np.int64)
        ascending_face = ascending // nside
        descending_face = descending // nside
        local_face = np.where(
            ascending_face == descending_face,
            ascending_face | 4,
            np.where(
                ascending_face < descending_face,
                ascending_face,
                descending_face + 8,
            ),
        )
        face[equatorial] = local_face
        x_index[equatorial] = descending % nside
        y_index[equatorial] = nside - (ascending % nside) - 1

    if np.any(polar):
        local_tt = tt[polar]
        local_z = z[polar]
        local_absolute_z = absolute_z[polar]
        quadrant = np.minimum(3, np.floor(local_tt).astype(np.int64))
        fractional = local_tt - quadrant
        radius = nside * np.sqrt(3.0 * (1.0 - local_absolute_z))
        ascending = np.minimum(
            nside - 1,
            np.floor(fractional * radius).astype(np.int64),
        )
        descending = np.minimum(
            nside - 1,
            np.floor((1.0 - fractional) * radius).astype(np.int64),
        )
        north = local_z >= 0.0
        face[polar] = np.where(north, quadrant, quadrant + 8)
        x_index[polar] = np.where(
            north,
            nside - descending - 1,
            ascending,
        )
        y_index[polar] = np.where(
            north,
            nside - ascending - 1,
            descending,
        )

    bits = int(math.log2(nside))
    within_face = _spread_nested_bits(x_index, bits) | (
        _spread_nested_bits(y_index, bits) << 1
    )
    return face * nside * nside + within_face


class SourceSampler(AbstractContextManager):
    """Base class for an all-sky scalar source."""

    def sample(
        self,
        galactic_longitude_deg: np.ndarray,
        galactic_latitude_deg: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback_object):
        self.close()
        return False

    def close(self) -> None:
        """Release resources owned by the source."""


class EquirectangularSampler(SourceSampler):
    def __init__(
        self,
        data: np.ndarray,
        *,
        frame: str,
        longitude_range: tuple[float, float],
        latitude_order: str,
        owner: Any = None,
    ):
        if np.asarray(data).ndim != 2:
            raise ValueError(
                f"Equirectangular input must be 2-D, got {data.shape}"
            )
        self.data = data
        self.frame = _normalize_frame(frame)
        self.longitude_min = float(longitude_range[0])
        self.longitude_max = float(longitude_range[1])
        self.latitude_order = latitude_order.lower()
        self.owner = owner

    def sample(self, galactic_longitude_deg, galactic_latitude_deg):
        longitude, latitude = _transform_from_galactic(
            galactic_longitude_deg,
            galactic_latitude_deg,
            self.frame,
        )
        height, width = self.data.shape
        longitude_span = self.longitude_max - self.longitude_min
        if longitude_span <= 0.0:
            raise ValueError("longitude_range must have increasing endpoints")
        x = (
            np.mod(longitude - self.longitude_min, longitude_span)
            / longitude_span
            * width
            - 0.5
        )
        if self.latitude_order in {"south_to_north", "ascending", "south"}:
            y = (latitude + 90.0) / 180.0 * height - 0.5
        else:
            y = (90.0 - latitude) / 180.0 * height - 0.5
        return _bilinear_equirectangular(self.data, x, y)

    def close(self):
        close = getattr(self.owner, "close", None)
        if close is not None:
            close()


class WCSSampler(SourceSampler):
    def __init__(
        self,
        data: np.ndarray,
        wcs: WCS,
        *,
        frame: str,
        owner: Any = None,
    ):
        if np.asarray(data).ndim != 2:
            raise ValueError(f"WCS image must be 2-D, got {data.shape}")
        self.data = data
        self.wcs = wcs
        self.frame = _normalize_frame(frame)
        self.owner = owner

    def sample(self, galactic_longitude_deg, galactic_latitude_deg):
        from scipy.ndimage import map_coordinates

        longitude, latitude = _transform_from_galactic(
            galactic_longitude_deg,
            galactic_latitude_deg,
            self.frame,
        )
        pixel_x, pixel_y = self.wcs.world_to_pixel_values(longitude, latitude)
        sampled = map_coordinates(
            self.data,
            [pixel_y, pixel_x],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        return np.asarray(sampled, dtype=np.float32)

    def close(self):
        close = getattr(self.owner, "close", None)
        if close is not None:
            close()


class HealpixSampler(SourceSampler):
    def __init__(
        self,
        values: np.ndarray,
        *,
        nside: int,
        ordering: str,
        frame: str,
        owner: Any = None,
    ):
        expected = 12 * int(nside) * int(nside)
        if np.asarray(values).size != expected:
            raise ValueError(
                f"HEALPix array has {np.asarray(values).size:,} pixels; "
                f"NSIDE={nside} requires {expected:,}"
            )
        self.values = np.asarray(values).reshape(-1)
        self.nside = int(nside)
        self.ordering = ordering.strip().upper()
        self.frame = _normalize_frame(frame)
        self.owner = owner

    def sample(self, galactic_longitude_deg, galactic_latitude_deg):
        longitude, latitude = _transform_from_galactic(
            galactic_longitude_deg,
            galactic_latitude_deg,
            self.frame,
        )
        if self.ordering in {"NESTED", "NEST"}:
            pixel = healpix_ang2pix_nested(
                self.nside,
                longitude,
                latitude,
            )
        elif self.ordering == "RING":
            pixel = healpix_ang2pix_ring(
                self.nside,
                longitude,
                latitude,
            )
        else:
            raise ValueError(
                f"Unsupported HEALPix ordering: {self.ordering!r}"
            )
        result = np.asarray(self.values[pixel], dtype=np.float32)
        result[result <= np.float32(-1.0e29)] = np.nan
        return result

    def close(self):
        close = getattr(self.owner, "close", None)
        if close is not None:
            close()


def _bilinear_equirectangular(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    height, width = data.shape
    valid = np.isfinite(x) & np.isfinite(y) & (y >= -0.5) & (y < height - 0.5)
    safe_x = np.where(np.isfinite(x), x, 0.0)
    safe_y = np.where(np.isfinite(y), y, 0.0)
    x0_unwrapped = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x_fraction = safe_x - x0_unwrapped
    y_fraction = safe_y - y0
    x0 = x0_unwrapped % width
    x1 = (x0 + 1) % width
    y0_clipped = np.clip(y0, 0, height - 1)
    y1_clipped = np.clip(y0 + 1, 0, height - 1)
    source = np.asarray(data)
    top = (
        np.asarray(source[y0_clipped, x0], dtype=np.float32)
        * (1.0 - x_fraction)
        + np.asarray(source[y0_clipped, x1], dtype=np.float32) * x_fraction
    )
    bottom = (
        np.asarray(source[y1_clipped, x0], dtype=np.float32)
        * (1.0 - x_fraction)
        + np.asarray(source[y1_clipped, x1], dtype=np.float32) * x_fraction
    )
    result = top * (1.0 - y_fraction) + bottom * y_fraction
    result = np.asarray(result, dtype=np.float32)
    result[~valid] = np.nan
    return result


def _select_plane(data: np.ndarray, plane: Any) -> np.ndarray:
    array = np.asarray(data)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim < 2:
        raise ValueError(f"Image data must be at least 2-D, got {array.shape}")
    indices = plane if isinstance(plane, list) else [plane]
    indices = [0 if value is None else int(value) for value in indices]
    while array.ndim > 2:
        index = indices.pop(0) if indices else 0
        array = array[index]
    return np.asarray(array)


def _select_image_hdu(hdul: fits.HDUList, requested: Any):
    if requested is not None:
        hdu = hdul[requested]
        if hdu.data is None:
            raise ValueError(f"FITS HDU {requested!r} contains no image")
        return hdu
    for hdu in hdul:
        if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and (
            hdu.data is not None
        ):
            return hdu
    raise RuntimeError("FITS file contains no image HDU")


def _select_table_hdu(hdul: fits.HDUList, requested: Any):
    if requested is None:
        return find_fits_table_hdu(hdul)
    hdu = hdul[requested]
    if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
        raise ValueError(f"FITS HDU {requested!r} is not a table")
    return hdu


def _open_fits(path: Path) -> fits.HDUList:
    return fits.open(path, memmap=True, lazy_load_hdus=True)


def _fits_image_sampler(path: Path, spec: dict[str, Any]) -> SourceSampler:
    hdul = _open_fits(path)
    try:
        hdu = _select_image_hdu(hdul, spec.get("hdu"))
        try:
            data = _select_plane(hdu.data, spec.get("plane"))
        except ValueError as error:
            if "memory-mapped image" not in str(error):
                raise
            hdul.close()
            hdul = fits.open(path, memmap=False, lazy_load_hdus=True)
            hdu = _select_image_hdu(hdul, spec.get("hdu"))
            data = _select_plane(hdu.data, spec.get("plane"))
        celestial_wcs = WCS(hdu.header).celestial
        frame = spec.get("coordinates") or _frame_from_header(hdu.header)
        if celestial_wcs.has_celestial and not spec.get(
            "force_equirectangular",
            False,
        ):
            return WCSSampler(
                data,
                celestial_wcs,
                frame=frame,
                owner=hdul,
            )
        return EquirectangularSampler(
            data,
            frame=frame,
            longitude_range=tuple(spec.get("longitude_range", [-180.0, 180.0])),
            latitude_order=spec.get("latitude_order", "north_to_south"),
            owner=hdul,
        )
    except Exception:
        hdul.close()
        raise


def _numeric_healpix_field(hdu, requested: str | None) -> str:
    names = list(hdu.data.names or [])
    if requested:
        selected = _case_insensitive_name(names, requested)
        if selected is None:
            raise KeyError(
                f"HEALPix field {requested!r} not found; available: "
                f"{', '.join(names)}"
            )
        return selected
    excluded = {name.lower() for name in HEALPIX_PIXEL_COLUMNS}
    for name in names:
        if name.lower() in excluded:
            continue
        if np.issubdtype(np.asarray(hdu.data[name]).dtype, np.number):
            return name
    raise KeyError("HEALPix table contains no numeric signal field")


def _healpix_vector(
    raw_values: np.ndarray,
    plane: Any,
    expected_size: int | None,
) -> np.ndarray:
    values = np.asanyarray(raw_values)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    values = np.asarray(values)
    values = np.squeeze(values)
    if values.ndim == 1:
        return values
    if expected_size is not None and values.size == expected_size:
        # Some HEALPix writers store consecutive blocks (for example 1024E
        # per row) instead of one scalar per row.  This is still one map, not
        # a multi-channel array.
        return values.reshape(-1)
    selected_plane = int(plane or 0)
    if expected_size is not None and values.shape[0] == expected_size:
        return values[:, selected_plane]
    if expected_size is not None and values.shape[-1] == expected_size:
        return values[selected_plane]
    return values[:, selected_plane].reshape(-1)


def _healpix_sampler(path: Path, spec: dict[str, Any]) -> SourceSampler:
    hdul = _open_fits(path)
    try:
        hdu = _select_table_hdu(hdul, spec.get("hdu"))
        header = hdu.header
        nside_header = spec.get("nside", header.get("NSIDE"))
        expected_size = (
            12 * int(nside_header) ** 2 if nside_header is not None else None
        )
        field = _numeric_healpix_field(hdu, spec.get("field"))
        values = _healpix_vector(
            hdu.data[field],
            spec.get("plane"),
            expected_size,
        )
        index_scheme = str(header.get("INDXSCHM", "IMPLICIT")).upper()
        if index_scheme == "EXPLICIT":
            pixel_field = (
                _case_insensitive_name(
                    list(hdu.data.names or []),
                    str(spec.get("pixel_column", "")),
                )
                if spec.get("pixel_column")
                else _first_matching_name(
                    list(hdu.data.names or []),
                    HEALPIX_PIXEL_COLUMNS,
                )
            )
            if pixel_field is None:
                raise KeyError(
                    "Explicit HEALPix table has no PIXEL/IPIX column"
                )
            pixels = np.asarray(hdu.data[pixel_field], dtype=np.int64).reshape(-1)
            if nside_header is None:
                raise ValueError("Explicit HEALPix table requires NSIDE")
            full_values = np.full(expected_size, np.nan, dtype=np.float32)
            valid = (pixels >= 0) & (pixels < expected_size)
            full_values[pixels[valid]] = np.asarray(
                values,
                dtype=np.float32,
            )[valid]
            values = full_values
        if nside_header is None:
            inferred = int(round(math.sqrt(values.size / 12.0)))
            if 12 * inferred * inferred != values.size:
                raise ValueError(
                    f"Cannot infer HEALPix NSIDE from {values.size:,} values"
                )
            nside_header = inferred
        nside = int(nside_header)
        bad_data = header.get("BAD_DATA")
        if bad_data is not None:
            values = np.asarray(values, dtype=np.float32)
            values[np.isclose(values, float(bad_data))] = np.nan
        return HealpixSampler(
            values,
            nside=nside,
            ordering=str(
                spec.get("ordering", header.get("ORDERING", "RING"))
            ),
            frame=spec.get("coordinates") or _frame_from_header(header),
            owner=hdul,
        )
    except Exception:
        hdul.close()
        raise


def _numpy_sampler(path: Path, spec: dict[str, Any]) -> SourceSampler:
    loaded = np.load(path, mmap_mode="r", allow_pickle=False)
    owner = loaded if isinstance(loaded, np.lib.npyio.NpzFile) else None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        field = spec.get("field")
        if field is None:
            if not loaded.files:
                loaded.close()
                raise ValueError(f"NPZ file is empty: {path}")
            field = loaded.files[0]
        if field not in loaded.files:
            loaded.close()
            raise KeyError(
                f"NPZ field {field!r} not found; available: "
                f"{', '.join(loaded.files)}"
            )
        data = loaded[field]
    else:
        data = loaded
    return EquirectangularSampler(
        _select_plane(data, spec.get("plane")),
        frame=spec.get("coordinates", "galactic"),
        longitude_range=tuple(spec.get("longitude_range", [-180.0, 180.0])),
        latitude_order=spec.get("latitude_order", "north_to_south"),
        owner=owner,
    )


def _raster_sampler(path: Path, spec: dict[str, Any]) -> SourceSampler:
    from PIL import Image

    image = Image.open(path)
    array = np.asarray(image)
    channel = spec.get("channel", "luminance")
    if array.ndim == 3:
        if isinstance(channel, int):
            array = array[:, :, channel]
        elif str(channel).lower() in {"r", "red"}:
            array = array[:, :, 0]
        elif str(channel).lower() in {"g", "green"}:
            array = array[:, :, 1]
        elif str(channel).lower() in {"b", "blue"}:
            array = array[:, :, 2]
        else:
            rgb = np.asarray(array[:, :, :3], dtype=np.float32)
            array = (
                0.2126 * rgb[:, :, 0]
                + 0.7152 * rgb[:, :, 1]
                + 0.0722 * rgb[:, :, 2]
            )
    return EquirectangularSampler(
        np.asarray(array),
        frame=spec.get("coordinates", "galactic"),
        longitude_range=tuple(spec.get("longitude_range", [-180.0, 180.0])),
        latitude_order=spec.get("latitude_order", "north_to_south"),
        owner=image,
    )


def detect_input_kind(path: Path, spec: dict[str, Any]) -> str:
    requested = str(spec.get("format", "auto")).strip().lower()
    aliases = {
        "fits": "fits_image",
        "wcs": "fits_image",
        "fits_wcs": "fits_image",
        "healpix_fits": "healpix",
        "table": "catalog",
        "fits_catalog": "catalog",
        "csv_catalog": "catalog",
        "npy": "equirectangular",
        "npz": "equirectangular",
        "image": "equirectangular_image",
        "raster": "equirectangular_image",
    }
    requested = aliases.get(requested, requested)
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".ecsv"}:
        return "catalog"
    if suffix in {".npy", ".npz"}:
        return "equirectangular"
    if suffix in RASTER_EXTENSIONS:
        return "equirectangular_image"
    if not _is_fits_path(path):
        raise ValueError(
            f"Cannot detect input type from extension {suffix!r}; "
            "set the 'format' field in the JSON configuration"
        )
    with _open_fits(path) as hdul:
        for hdu in hdul:
            if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                continue
            header = hdu.header
            if (
                str(header.get("PIXTYPE", "")).upper() == "HEALPIX"
                or "NSIDE" in header
                or "ORDERING" in header
            ):
                return "healpix"
        for hdu in hdul:
            if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU)) and (
                hdu.data is not None
            ):
                return "fits_image"
        try:
            find_fits_table_hdu(hdul)
        except RuntimeError:
            pass
        else:
            return "catalog"
    raise ValueError(f"No supported data HDU found in {path}")


def open_source_sampler(
    path: Path,
    spec: dict[str, Any],
    kind: str,
) -> SourceSampler:
    if kind == "fits_image":
        return _fits_image_sampler(path, spec)
    if kind == "healpix":
        return _healpix_sampler(path, spec)
    if kind == "equirectangular":
        return _numpy_sampler(path, spec)
    if kind == "equirectangular_image":
        return _raster_sampler(path, spec)
    raise ValueError(f"Input type {kind!r} is not a raster map")


def _allocate_scalar(
    width: int,
    height: int,
    output_dir: Path,
    slug: str,
) -> tuple[np.ndarray, Path | None]:
    byte_count = width * height * np.dtype(np.float32).itemsize
    if byte_count <= 256 * 1024**2:
        return np.empty((height, width), dtype=np.float32), None
    temporary_path = output_dir / f".{slug}_projected_f32.dat"
    array = np.memmap(
        temporary_path,
        mode="w+",
        dtype=np.float32,
        shape=(height, width),
    )
    return array, temporary_path


def project_sampler(
    sampler: SourceSampler,
    *,
    width: int,
    height: int,
    tile_rows: int,
    output_dir: Path,
    slug: str,
) -> tuple[np.ndarray, Path | None]:
    projected, temporary_path = _allocate_scalar(
        width,
        height,
        output_dir,
        slug,
    )
    ranges = range(0, height, tile_rows)
    for y0 in console.progress(
        ranges,
        total=(height + tile_rows - 1) // tile_rows,
        desc=f"Projecting {slug}",
        unit="tile",
    ):
        MEMORY.throttle()
        y1 = min(y0 + tile_rows, height)
        longitude, latitude, valid = hammer_world_tile(
            y0,
            y1,
            width,
            height,
        )
        tile = sampler.sample(longitude, latitude)
        tile = np.asarray(tile, dtype=np.float32)
        tile[~valid] = np.nan
        projected[y0:y1] = tile
    flush = getattr(projected, "flush", None)
    if flush is not None:
        flush()
    return projected, temporary_path


def _catalog_columns(
    names: list[str],
    spec: dict[str, Any],
) -> tuple[str, str, str | None, str]:
    longitude = (
        _case_insensitive_name(names, spec["longitude_column"])
        if spec.get("longitude_column")
        else _first_matching_name(names, CATALOG_LONGITUDE_COLUMNS)
    )
    latitude = (
        _case_insensitive_name(names, spec["latitude_column"])
        if spec.get("latitude_column")
        else _first_matching_name(names, CATALOG_LATITUDE_COLUMNS)
    )
    if longitude is None or latitude is None:
        raise KeyError(
            "Could not detect catalogue coordinate columns. Set "
            "'longitude_column' and 'latitude_column'. Available columns: "
            + ", ".join(names)
        )
    frame = spec.get("coordinates")
    if frame is None:
        frame = (
            "icrs"
            if longitude.lower().startswith("ra")
            or latitude.lower().startswith("dec")
            else "galactic"
        )
    mode = str(spec.get("value_mode", "intensity")).lower()
    if mode == "count":
        value = None
    elif spec.get("value_column"):
        value = _case_insensitive_name(names, spec["value_column"])
        if value is None:
            raise KeyError(
                f"Value column {spec['value_column']!r} not found; "
                f"available: {', '.join(names)}"
            )
    else:
        value = _first_matching_name(names, CATALOG_VALUE_COLUMNS)
        if value is None:
            raise KeyError(
                "Could not detect a catalogue value column. Set "
                "'value_column', or use 'value_mode': 'count'."
            )
    return longitude, latitude, value, _normalize_frame(frame)


def _catalog_weights(values: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == "count":
        if values is None:
            raise ValueError("Count mode requires a row-count placeholder")
        return np.ones(np.asarray(values).shape, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if mode in {"magnitude", "mag"}:
        return np.power(
            np.float32(10.0),
            np.float32(-0.4) * values,
        ).astype(np.float32, copy=False)
    if mode in {"log10", "dex"}:
        return np.power(np.float32(10.0), values).astype(
            np.float32,
            copy=False,
        )
    return values


def _accumulate_catalog_chunk(
    result: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    values: np.ndarray | None,
    *,
    frame: str,
    value_mode: str,
) -> int:
    longitude = np.asarray(longitude, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)
    if value_mode == "count":
        weights = _catalog_weights(np.empty(longitude.shape), value_mode)
    else:
        weights = _catalog_weights(values, value_mode)
    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(weights)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )
    if not np.any(valid):
        return 0
    longitude, latitude = _transform_to_galactic(
        longitude[valid],
        latitude[valid],
        frame,
    )
    weights = weights[valid]
    pixel_x, pixel_y, projected_valid = galactic_to_hammer_indices(
        longitude,
        latitude,
        result.shape[1],
        result.shape[0],
    )
    np.add.at(
        result,
        (pixel_y[projected_valid], pixel_x[projected_valid]),
        weights[projected_valid],
    )
    return int(np.count_nonzero(projected_valid))


def _catalog_from_fits(
    path: Path,
    spec: dict[str, Any],
    result: np.ndarray,
) -> int:
    total = 0
    chunk_rows = int(spec.get("chunk_rows", DEFAULT_FITS_CATALOG_CHUNK_ROWS))
    with _open_fits(path) as hdul:
        hdu = _select_table_hdu(hdul, spec.get("hdu"))
        data = hdu.data
        names = list(data.names or [])
        longitude_name, latitude_name, value_name, frame = _catalog_columns(
            names,
            spec,
        )
        value_mode = str(
            spec.get("value_mode", DEFAULT_CATALOG_VALUE_MODE)
        ).lower()
        ranges = range(0, len(data), chunk_rows)
        for start in console.progress(
            ranges,
            total=(len(data) + chunk_rows - 1) // chunk_rows,
            desc="Binning catalogue",
            unit="chunk",
        ):
            MEMORY.throttle()
            stop = min(start + chunk_rows, len(data))
            values = (
                None
                if value_name is None
                else np.asarray(data[value_name][start:stop])
            )
            total += _accumulate_catalog_chunk(
                result,
                data[longitude_name][start:stop],
                data[latitude_name][start:stop],
                values,
                frame=frame,
                value_mode=value_mode,
            )
    return total


def _catalog_from_delimited(
    path: Path,
    spec: dict[str, Any],
    result: np.ndarray,
) -> int:
    delimiter = spec.get("delimiter")
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    chunk_rows = int(
        spec.get("chunk_rows", DEFAULT_DELIMITED_CATALOG_CHUNK_ROWS)
    )
    total = 0
    with path.open("r", encoding=spec.get("encoding", "utf-8-sig"), newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        names = list(reader.fieldnames or [])
        longitude_name, latitude_name, value_name, frame = _catalog_columns(
            names,
            spec,
        )
        value_mode = str(
            spec.get("value_mode", DEFAULT_CATALOG_VALUE_MODE)
        ).lower()
        longitude_values: list[float] = []
        latitude_values: list[float] = []
        signal_values: list[float] = []

        def flush_chunk() -> int:
            if not longitude_values:
                return 0
            signal = (
                None
                if value_name is None
                else np.asarray(signal_values, dtype=np.float64)
            )
            count = _accumulate_catalog_chunk(
                result,
                np.asarray(longitude_values, dtype=np.float64),
                np.asarray(latitude_values, dtype=np.float64),
                signal,
                frame=frame,
                value_mode=value_mode,
            )
            longitude_values.clear()
            latitude_values.clear()
            signal_values.clear()
            return count

        for row in reader:
            try:
                longitude_values.append(float(row[longitude_name]))
                latitude_values.append(float(row[latitude_name]))
                if value_name is not None:
                    signal_values.append(float(row[value_name]))
            except (KeyError, TypeError, ValueError):
                longitude_values.append(np.nan)
                latitude_values.append(np.nan)
                if value_name is not None:
                    signal_values.append(np.nan)
            if len(longitude_values) >= chunk_rows:
                total += flush_chunk()
        total += flush_chunk()
    return total


def _catalog_from_ecsv(
    path: Path,
    spec: dict[str, Any],
    result: np.ndarray,
) -> int:
    from astropy.table import Table

    table = Table.read(path, format="ascii.ecsv")
    names = list(table.colnames)
    longitude_name, latitude_name, value_name, frame = _catalog_columns(
        names,
        spec,
    )
    value_mode = str(
        spec.get("value_mode", DEFAULT_CATALOG_VALUE_MODE)
    ).lower()
    chunk_rows = int(spec.get("chunk_rows", DEFAULT_FITS_CATALOG_CHUNK_ROWS))
    total = 0
    for start in range(0, len(table), chunk_rows):
        MEMORY.throttle()
        stop = min(start + chunk_rows, len(table))
        values = (
            None
            if value_name is None
            else np.asarray(table[value_name][start:stop])
        )
        total += _accumulate_catalog_chunk(
            result,
            table[longitude_name][start:stop],
            table[latitude_name][start:stop],
            values,
            frame=frame,
            value_mode=value_mode,
        )
    return total


def project_catalog(
    path: Path,
    spec: dict[str, Any],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    result = np.zeros((height, width), dtype=np.float32)
    if _is_fits_path(path):
        total = _catalog_from_fits(path, spec, result)
    elif path.suffix.lower() == ".ecsv":
        total = _catalog_from_ecsv(path, spec, result)
    else:
        total = _catalog_from_delimited(path, spec, result)
    console.success(f"Catalogue rows projected: {total:,}")
    smoothing = float(
        spec.get("catalog_smoothing_sigma", DEFAULT_CATALOG_SMOOTHING_SIGMA)
    )
    if smoothing > 0.0:
        from scipy.ndimage import gaussian_filter

        result = gaussian_filter(
            result,
            sigma=smoothing,
            mode="constant",
        ).astype(np.float32, copy=False)
    result[~hammer_valid_mask(width, height)] = np.nan
    return result


def _finite_sample(
    array: np.ndarray,
    *,
    sample_size: int,
    ignore_zeros: bool,
) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    step = max(1, flat.size // max(sample_size, 1))
    sample = np.asarray(flat[::step], dtype=np.float32)
    valid = np.isfinite(sample)
    if ignore_zeros:
        valid &= sample != 0.0
    sample = sample[valid]
    if sample.size >= min(1000, sample_size) or step == 1:
        return sample[:sample_size]

    pieces = []
    found = 0
    for start in range(0, flat.size, FINITE_SAMPLE_CHUNK_SIZE):
        MEMORY.throttle()
        chunk = np.asarray(
            flat[start : start + FINITE_SAMPLE_CHUNK_SIZE],
            dtype=np.float32,
        )
        valid = np.isfinite(chunk)
        if ignore_zeros:
            valid &= chunk != 0.0
        chunk = chunk[valid]
        if not chunk.size:
            continue
        remaining = sample_size - found
        if chunk.size > remaining:
            chunk = chunk[:: max(1, chunk.size // remaining)][:remaining]
        pieces.append(chunk)
        found += chunk.size
        if found >= sample_size:
            break
    if not pieces:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(pieces)[:sample_size]


def _parse_percentiles(
    spec: dict[str, Any],
    options: dict[str, Any],
) -> tuple[float, float]:
    percentiles = spec.get(
        "percentiles",
        options.get("percentiles", [0.5, 99.5]),
    )
    if len(percentiles) != 2:
        raise ValueError("'percentiles' must contain exactly two values")
    lower, upper = map(float, percentiles)
    if not (0.0 <= lower < upper <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    return lower, upper


def _make_palette_lut(palette: str, color: str | None = None) -> np.ndarray:
    import cv2

    if color:
        text = str(color).lstrip("#")
        if len(text) != 6:
            raise ValueError("'color' must use the #RRGGBB format")
        red, green, blue = (
            int(text[0:2], 16),
            int(text[2:4], 16),
            int(text[4:6], 16),
        )
        fraction = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
        rgb = fraction * np.array([[red, green, blue]], dtype=np.float32)
        return np.clip(rgb[:, ::-1] + 0.5, 0, 255).astype(np.uint8)

    name = str(palette or "inferno").strip().lower()
    reversed_palette = name.endswith("_r")
    if reversed_palette:
        name = name[:-2]
    if name in {"cmb", "blue_white_red", "bwr"}:
        fraction = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        rgb = np.empty((256, 3), dtype=np.float32)

        lower_half = fraction <= 0.5
        upper_half = ~lower_half

        # blue -> white
        t = fraction[lower_half] / 0.5
        rgb[lower_half, 0] = 255.0 * t
        rgb[lower_half, 1] = 255.0 * t
        rgb[lower_half, 2] = 255.0

        # white -> red
        t = (fraction[upper_half] - 0.5) / 0.5
        rgb[upper_half, 0] = 255.0
        rgb[upper_half, 1] = 255.0 * (1.0 - t)
        rgb[upper_half, 2] = 255.0 * (1.0 - t)

        # RGB -> BGR (OpenCV)
        lut = np.clip(rgb[:, ::-1] + 0.5, 0, 255).astype(np.uint8)

        return lut[::-1] if reversed_palette else lut
    if name in {"gray", "grey", "grayscale", "greyscale"}:
        levels = np.arange(256, dtype=np.uint8)
        lut = np.stack([levels, levels, levels], axis=1)
    else:
        constants = {
            "autumn": cv2.COLORMAP_AUTUMN,
            "bone": cv2.COLORMAP_BONE,
            "cool": cv2.COLORMAP_COOL,
            "hot": cv2.COLORMAP_HOT,
            "hsv": cv2.COLORMAP_HSV,
            "inferno": cv2.COLORMAP_INFERNO,
            "jet": cv2.COLORMAP_JET,
            "magma": cv2.COLORMAP_MAGMA,
            "ocean": cv2.COLORMAP_OCEAN,
            "parula": cv2.COLORMAP_PARULA,
            "pink": cv2.COLORMAP_PINK,
            "plasma": cv2.COLORMAP_PLASMA,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "spring": cv2.COLORMAP_SPRING,
            "summer": cv2.COLORMAP_SUMMER,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "winter": cv2.COLORMAP_WINTER,
        }
        if name not in constants:
            raise ValueError(
                f"Unknown palette {palette!r}; available: "
                + ", ".join(sorted(constants))
                + ", gray"
            )
        levels = np.arange(256, dtype=np.uint8).reshape(256, 1)
        lut = cv2.applyColorMap(levels, constants[name]).reshape(256, 3)
    return lut[::-1] if reversed_palette else lut


def _stretch_normalized(
    normalized: np.ndarray,
    stretch: str,
    strength: float,
    gamma: float,
) -> np.ndarray:
    stretch = stretch.lower()
    if stretch == "linear":
        result = normalized
    elif stretch == "asinh":
        strength = max(float(strength), 1.0e-6)
        result = np.arcsinh(strength * normalized) / np.arcsinh(strength)
    elif stretch == "log":
        strength = max(float(strength), 1.0e-6)
        result = np.log1p(strength * normalized) / np.log1p(strength)
    elif stretch == "sqrt":
        result = np.sqrt(normalized)
    elif stretch in {"power", "gamma"}:
        result = np.power(normalized, max(float(gamma), 1.0e-6))
    else:
        raise ValueError(
            f"Unknown stretch {stretch!r}; use linear, asinh, log, sqrt, "
            "or power"
        )
    return np.asarray(result, dtype=np.float32)


def save_tone_mapped_png(
    projected: np.ndarray,
    output_path: Path,
    spec: dict[str, Any],
    options: dict[str, Any],
    *,
    kind: str,
    tile_rows: int,
    constellation_polylines: list[np.ndarray] | None = None,
) -> dict[str, float | str]:
    import cv2

    scale_factor = float(spec.get("scale_factor", DEFAULT_SCALE_FACTOR))
    offset = float(spec.get("offset", DEFAULT_OFFSET))
    ignore_zeros = bool(
        spec.get("ignore_zeros", kind == "catalog")
    )
    sample = _finite_sample(
        projected,
        sample_size=int(options.get("percentile_sample_size", DEFAULT_SAMPLE_SIZE)),
        ignore_zeros=ignore_zeros,
    )
    if not sample.size:
        raise RuntimeError("No finite signal values found after projection")
    sample = sample * np.float32(scale_factor) + np.float32(offset)
    lower_percentile, upper_percentile = _parse_percentiles(spec, options)
    lower, upper = np.percentile(
        sample,
        [lower_percentile, upper_percentile],
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        finite_min = float(np.min(sample))
        finite_max = float(np.max(sample))
        if finite_max <= finite_min:
            upper = finite_min + 1.0
            lower = finite_min
        else:
            lower, upper = finite_min, finite_max
    stretch = str(
        spec.get("stretch", options.get("stretch", DEFAULT_STRETCH))
    )
    strength = float(
        spec.get(
            "stretch_strength",
            options.get("stretch_strength", DEFAULT_STRETCH_STRENGTH),
        )
    )
    gamma = float(spec.get("gamma", options.get("gamma", DEFAULT_GAMMA)))
    lut = _make_palette_lut(
        str(spec.get("palette", options.get("palette", DEFAULT_PALETTE))),
        spec.get("color"),
    )
    height, width = projected.shape
    temporary_raster = output_path.with_name(f".{output_path.stem}_bgr.dat")
    bgr = np.memmap(
        temporary_raster,
        mode="w+",
        dtype=np.uint8,
        shape=(height, width, 3),
    )
    temporary_png = output_path.with_name(f".{output_path.stem}.tmp.png")
    try:
        ranges = range(0, height, tile_rows)
        for y0 in console.progress(
            ranges,
            total=(height + tile_rows - 1) // tile_rows,
            desc=f"Tone mapping {output_path.stem}",
            unit="tile",
        ):
            MEMORY.throttle()
            y1 = min(y0 + tile_rows, height)
            values = (
                np.asarray(projected[y0:y1], dtype=np.float32)
                * np.float32(scale_factor)
                + np.float32(offset)
            )
            finite = np.isfinite(values)
            normalized = np.clip(
                (values - np.float32(lower))
                / np.float32(max(upper - lower, np.finfo(np.float32).eps)),
                0.0,
                1.0,
            )
            normalized[~finite] = 0.0
            normalized = _stretch_normalized(
                normalized,
                stretch,
                strength,
                gamma,
            )
            indices = np.clip(
                normalized * 255.0 + 0.5,
                0,
                255,
            ).astype(np.uint8)
            tile = lut[indices]
            tile[~finite] = 0
            bgr[y0:y1] = tile
        if constellation_polylines:
            draw_constellation_polylines(
                bgr,
                constellation_polylines,
            )
        bgr.flush()
        if not cv2.imwrite(
            str(temporary_png),
            bgr,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                int(options.get("png_compression", DEFAULT_PNG_COMPRESSION)),
            ],
        ):
            raise RuntimeError(f"OpenCV could not save {temporary_png}")
        temporary_png.replace(output_path)
    finally:
        del bgr
        try:
            temporary_raster.unlink()
        except FileNotFoundError:
            pass
        if temporary_png.exists():
            temporary_png.unlink()
    return {
        "lower": float(lower),
        "upper": float(upper),
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "stretch": stretch,
    }


def _resolve_input_path(spec: dict[str, Any], config_dir: Path) -> Path:
    if not spec.get("path"):
        raise ValueError("Map configuration has no 'path'")
    path = Path(spec["path"]).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def render_map(
    spec: dict[str, Any],
    options: dict[str, Any],
    *,
    config_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    constellation_polylines: list[np.ndarray] | None = None,
) -> tuple[Path, dict[str, Any]]:
    name = str(spec.get("name") or spec.get("id") or "Map")
    slug = _safe_slug(str(spec.get("id") or name))
    path = _resolve_input_path(spec, config_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    kind = detect_input_kind(path, spec)
    console.section(name)
    console.info(f"Input: {path}")
    console.info(f"Detected format: {kind}")
    temporary_scalar: Path | None = None
    projected: np.ndarray | None = None
    try:
        if kind == "catalog":
            projected = project_catalog(
                path,
                spec,
                width=width,
                height=height,
            )
        else:
            with open_source_sampler(path, spec, kind) as sampler:
                projected, temporary_scalar = project_sampler(
                    sampler,
                    width=width,
                    height=height,
                    tile_rows=int(options.get("tile_rows", DEFAULT_TILE_ROWS)),
                    output_dir=output_dir,
                    slug=slug,
                )
        output_path = output_dir / f"{slug}_hammer.png"
        tone_metadata = save_tone_mapped_png(
            projected,
            output_path,
            spec,
            options,
            kind=kind,
            tile_rows=int(options.get("tile_rows", DEFAULT_TILE_ROWS)),
            constellation_polylines=constellation_polylines,
        )
        console.success(f"Map saved: {output_path}")
        return output_path, {
            "id": slug,
            "name": name,
            "wavelength": str(spec.get("wavelength", "")),
            "input": str(path),
            "format": kind,
            "output": str(output_path),
            "width": width,
            "height": height,
            "constellation_lines": bool(constellation_polylines),
            **tone_metadata,
        }
    finally:
        if projected is not None:
            flush = getattr(projected, "flush", None)
            if flush is not None:
                flush()
            del projected
        if temporary_scalar is not None:
            try:
                temporary_scalar.unlink()
            except FileNotFoundError:
                pass


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_side_by_side_collage(
    rendered: list[tuple[Path, dict[str, Any]]],
    output_path: Path,
    *,
    tile_width: int,
    gap: int = 12,
    title_height: int = 72,
) -> Path:
    from PIL import Image, ImageDraw, ImageOps

    if len(rendered) < 2:
        raise ValueError("A collage requires at least two rendered maps")
    opened = [Image.open(path).convert("RGB") for path, _ in rendered]
    try:
        widths = [
            image.width if tile_width <= 0 else min(tile_width, image.width)
            for image in opened
        ]
        heights = [
            max(1, round(image.height * width / image.width))
            for image, width in zip(opened, widths)
        ]
        content_height = max(heights)
        canvas_width = sum(widths) + gap * (len(widths) - 1)
        canvas = Image.new(
            "RGB",
            (canvas_width, title_height + content_height),
            (0, 0, 0),
        )
        draw = ImageDraw.Draw(canvas)
        font = _load_font(max(14, title_height // 3))
        x = 0
        for image, width, height, (_, metadata) in zip(
            opened,
            widths,
            heights,
            rendered,
        ):
            resized = ImageOps.contain(
                image,
                (width, content_height),
                method=Image.Resampling.LANCZOS,
            )
            y = title_height + (content_height - resized.height) // 2
            canvas.paste(resized, (x, y))
            title = str(metadata["name"])
            wavelength = str(metadata.get("wavelength", "")).strip()
            if wavelength and wavelength.lower() not in title.lower():
                title = f"{title} — {wavelength}"
            draw.text(
                (x + width / 2, title_height / 2),
                title,
                font=font,
                fill=(245, 245, 245),
                anchor="mm",
            )
            x += width + gap
        temporary = output_path.with_name(f".{output_path.stem}.tmp.png")
        canvas.save(temporary, format="PNG", compress_level=6)
        temporary.replace(output_path)
    finally:
        for image in opened:
            image.close()
    console.success(f"Side-by-side collage saved: {output_path}")
    return output_path


def load_configuration(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}\n"
            f"Start from the template at {DEFAULT_CONFIG}"
        ) from None
    if not isinstance(configuration, dict):
        raise ValueError("Configuration root must be a JSON object")
    maps = configuration.get("maps")
    if not isinstance(maps, list) or not maps:
        raise ValueError("Configuration must contain a non-empty 'maps' list")
    enabled = [entry for entry in maps if entry.get("enabled", True)]
    if not enabled:
        raise ValueError("Every map in the configuration is disabled")
    identifiers: set[str] = set()
    for index, entry in enumerate(enabled, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Map entry #{index} is not a JSON object")
        identifier = _safe_slug(
            str(entry.get("id") or entry.get("name") or f"map_{index}")
        )
        if identifier in identifiers:
            raise ValueError(f"Duplicate map id: {identifier}")
        identifiers.add(identifier)
        entry.setdefault("id", identifier)
    output_options = configuration.get("output", {})
    if not isinstance(output_options, dict):
        raise ValueError("'output' must be a JSON object")
    return output_options, enabled


def list_maps(
    maps: list[dict[str, Any]],
    config_dir: Path,
) -> None:
    console.section("Available multiwavelength maps")
    for index, spec in enumerate(maps, start=1):
        name = str(spec.get("name") or spec.get("id"))
        wavelength = str(spec.get("wavelength", "")).strip()
        try:
            path = _resolve_input_path(spec, config_dir)
            state = "ready" if path.is_file() else "missing file"
        except ValueError:
            state = "missing path"
        details = f" — {wavelength}" if wavelength else ""
        console.print(f"  {index:>2}. {name}{details} [{state}]")


def parse_selection(
    text: str,
    maps: list[dict[str, Any]],
) -> list[int]:
    value = text.strip().lower()
    if value in {"a", "all", "wszystkie", "*"}:
        return list(range(len(maps)))
    if value in {"q", "quit", "exit", "koniec"}:
        return []
    tokens = [token for token in re.split(r"[,;\s]+", value) if token]
    if not tokens:
        raise ValueError("No maps were selected")
    identifiers = {
        str(spec.get("id", "")).lower(): index
        for index, spec in enumerate(maps)
    }
    selected: list[int] = []
    for token in tokens:
        if token in identifiers:
            index = identifiers[token]
            if index not in selected:
                selected.append(index)
            continue
        if "-" in token:
            first_text, last_text = token.split("-", 1)
            if not first_text.isdigit() or not last_text.isdigit():
                raise ValueError(f"Invalid range: {token!r}")
            first = int(first_text)
            last = int(last_text)
            if first > last:
                first, last = last, first
            values = range(first, last + 1)
        elif token.isdigit():
            values = [int(token)]
        else:
            raise ValueError(f"Unknown selection: {token!r}")
        for one_based in values:
            if one_based < 1 or one_based > len(maps):
                raise ValueError(
                    f"Number {one_based} is outside the range 1-{len(maps)}"
                )
            index = one_based - 1
            if index not in selected:
                selected.append(index)
    return selected


def ask_yes_no(message: str, *, default: bool = False) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    response = console.prompt(f"{message.rstrip('?')}? {hint}").strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}


def _resolved_output_directory(
    argument: str | None,
    options: dict[str, Any],
    config_dir: Path,
) -> Path:
    # Final multiwavelength products live under the canonical map tree.
    return MAPS_DIR / "multiband"


def _write_run_metadata(
    output_dir: Path,
    rendered: list[tuple[Path, dict[str, Any]]],
    *,
    collage: Path | None,
) -> None:
    metadata = {
        "projection": "hammer",
        "maps": [entry for _, entry in rendered],
        "collage": str(collage) if collage is not None else None,
    }
    path = output_dir / "aurora_multiband_render.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_output_dimensions(
    args: argparse.Namespace,
    options: dict[str, Any],
) -> tuple[int, int]:
    if args.width is not None or args.height is not None:
        width = int(args.width or options.get("width", DEFAULT_WIDTH))
        height = int(args.height or options.get("height", width // 2))
        return width, height
    default_resolution = options.get(
        "resolution_k",
        options.get("resolution", DEFAULT_RESOLUTION_K),
    )
    profile = prompt_resolution(default=default_resolution)
    return profile.width, profile.height


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render selected radio/IR/optical/UV/X-ray/gamma or custom "
            "all-sky data as separate Hammer-projection PNG files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"JSON map registry (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--select",
        help="Map numbers/ids, e.g. '1,3-5', or 'all'; omit for a prompt",
    )
    parser.add_argument(
        "--collage",
        choices=("ask", "yes", "no"),
        default=DEFAULT_COLLAGE_MODE,
        help="Build a side-by-side collage when multiple maps are selected",
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
    parser.add_argument("--list", action="store_true", help="List maps and exit")
    parser.add_argument("--width", type=int, help="Output PNG width")
    parser.add_argument("--height", type=int, help="Output PNG height")
    parser.add_argument(
        "--output-dir",
        help=(
            "Deprecated compatibility option; final maps are always written "
            "to maps/multiband/"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tracebacks when an individual map fails",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    start_time = time.perf_counter()
    args = build_argument_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        options, maps = load_configuration(config_path)
    except Exception as error:
        console.error(error)
        return 2
    config_dir = config_path.parent
    list_maps(maps, config_dir)
    if args.list:
        return 0

    selection_text = args.select
    if selection_text is None:
        if not sys.stdin.isatty():
            console.error(
                "Interactive selection needs a terminal; use --select "
                "'1,3' or --select all"
            )
            return 2
        selection_text = console.prompt(
            "Select maps (for example 1 3 5, 2-4, all; q = quit)"
        )
    try:
        selected_indices = parse_selection(selection_text, maps)
    except ValueError as error:
        console.error(error)
        return 2
    if not selected_indices:
        console.info("No maps selected; exiting")
        return 0

    width, height = _resolve_output_dimensions(args, options)
    if width < 64 or height < 32:
        console.error("Output dimensions must be at least 64 x 32 pixels")
        return 2
    output_dir = _resolved_output_directory(
        args.output_dir,
        options,
        config_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        console.warning(
            "--output-dir is ignored; final maps are always saved to "
            f"{MAPS_DIR / 'multiband'}"
        )
    console.section("MAP RENDER — Multiwavelength all-sky")
    console.detail("Projection: Hammer equal-area")
    console.detail(f"Dimensions: {width} × {height} px")
    console.detail(f"Selected maps: {len(selected_indices)}")
    console.detail(f"Output directory: {output_dir}")

    include_constellations = prompt_constellation_overlay(
        args.constellations
    )
    console.detail(
        "Constellation overlay: "
        f"{'enabled' if include_constellations else 'disabled'}"
    )

    constellation_polylines: list[np.ndarray] | None = None
    if include_constellations:
        try:
            constellation_polylines = load_constellation_polylines(
                width,
                height,
            )
        except Exception as error:
            console.error(f"Could not load constellation lines: {error}")
            return 2
        console.info(
            f"Constellation index: {CONSTELLATION_INDEX_FILE}"
        )

    rendered: list[tuple[Path, dict[str, Any]]] = []
    failures: list[tuple[str, str]] = []
    for index in selected_indices:
        MEMORY.throttle()
        spec = maps[index]
        name = str(spec.get("name") or spec.get("id"))
        try:
            rendered.append(
                render_map(
                    spec,
                    options,
                    config_dir=config_dir,
                    output_dir=output_dir,
                    width=width,
                    height=height,
                    constellation_polylines=constellation_polylines,
                )
            )
        except Exception as error:
            failures.append((name, str(error)))
            console.error(f"{name}: {error}")
            if args.debug:
                traceback.print_exc()

    collage_path: Path | None = None
    if len(rendered) > 1:
        if args.collage == "yes":
            make_collage = True
        elif args.collage == "no":
            make_collage = False
        elif sys.stdin.isatty():
            make_collage = ask_yes_no(
                "Also create a side-by-side map collage"
            )
        else:
            make_collage = False
        if make_collage:
            collage_path = build_side_by_side_collage(
                rendered,
                output_dir / "aurora_multiband_collage.png",
                tile_width=int(
                    options.get(
                        "collage_tile_width", DEFAULT_COLLAGE_TILE_WIDTH
                    )
                ),
                gap=int(options.get("collage_gap", DEFAULT_COLLAGE_GAP)),
                title_height=int(
                    options.get(
                        "collage_title_height", DEFAULT_COLLAGE_TITLE_HEIGHT
                    )
                ),
            )
    if rendered:
        _write_run_metadata(
            output_dir,
            rendered,
            collage=collage_path,
        )

    elapsed = time.perf_counter() - start_time
    console.complete("MAP RENDER — Multiwavelength all-sky")
    console.success(f"Output: {output_dir}")
    console.detail(
        f"Runtime: {elapsed / 3600.0:.2f} h "
        f"({elapsed / 60.0:.1f} min, {elapsed:.1f} s)"
    )
    console.success(f"Separate maps created: {len(rendered)}")
    if collage_path is not None:
        console.success(f"Collage: {collage_path}")
    if failures:
        console.warning(f"Maps that failed: {len(failures)}")
        for name, message in failures:
            console.detail(f"{name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
