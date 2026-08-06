"""Shared numerical and rasterisation helpers for AURORA renderers.

This module contains calculations that are independent of catalogue paths,
cache layout and video encoding.  Render scripts keep their own configuration
and I/O, while using one implementation of stellar colour, Hammer coordinates,
PSF kernels, light curves and Gaussian drawing.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


SOLAR_TEMPERATURE_K = 5772.0
REFERENCE_MAP_HEIGHT = 8192.0
BP_RP_MIN = -0.5
BP_RP_MAX = 4.0
BLACKBODY_TEMPERATURE_MIN_K = 1_000.0
BLACKBODY_TEMPERATURE_MAX_K = 40_000.0
VARIABLE_COLOUR_FALLBACK_TEMPERATURE_K = 5_800.0
VARIABLE_COLOUR_TEMPERATURE_ANCHORS_K = np.array(
    [2_500.0, 3_500.0, 5_000.0, 5_800.0, 7_000.0, 9_000.0, 10_000.0],
    dtype=np.float32,
)
VARIABLE_COLOUR_RGB_ANCHORS = np.array(
    [
        [1.0, 0.25, 0.08],
        [1.0, 0.55, 0.18],
        [1.0, 0.85, 0.45],
        [1.0, 0.95, 0.85],
        [0.85, 0.90, 1.0],
        [0.65, 0.80, 1.0],
        [0.45, 0.65, 1.0],
    ],
    dtype=np.float32,
)

PSF_SMALL_MOFFAT = (31, 2.0, 2.8)
PSF_LARGE_MOFFAT = (81, 6.0, 2.8)
PSF_SMALL_WEIGHT = 0.8
PSF_LARGE_WEIGHT = 0.2
PSF_GAUSSIAN_SIGMA = 0.8

VARIABLE_MODEL_CODES = {
    "rr_lyrae": 0,
    "cepheid": 1,
    "zz_ceti": 2,
    "lbv": 3,
    "cataclysmic": 4,
    "eclipsing": 5,
    "ellipsoidal": 6,
    "multimode": 7,
    "long_period": 8,
    "rcb_decline": 9,
    "microlensing_event": 10,
    "supernova_event": 11,
    "yso_irregular": 12,
    "rotational": 13,
    "gentle_pulsation": 14,
    "solar_like": 15,
    "agn_noise": 16,
    "planet_transit": 17,
    "conservative": 18,
    "stochastic_hot": 19,
}
VARIABLE_FLUX_MODES = {
    "range": 0,
    "outburst": 1,
    "dimming": 2,
    "signed": 3,
    "supernova": 4,
}

STOCHASTIC_HOT_SLOW_WEIGHT = np.float32(0.72)
STOCHASTIC_HOT_FAST_WEIGHT = np.float32(0.28)
STOCHASTIC_HOT_MIN_CHANGES = np.float32(12.0)
STOCHASTIC_HOT_ADDITIONAL_CHANGES = np.float32(20.0)
STOCHASTIC_HOT_FAST_OFFSET_SCALE = np.float32(1.731)
STOCHASTIC_HOT_FAST_CHANGE_SCALE = np.float32(2.37)

CATACLYSMIC_RISE_END = 0.045
CATACLYSMIC_PLATEAU_END = 0.13
CATACLYSMIC_DECAY_END = 0.43
CATACLYSMIC_DECAY_POWER = 1.35

DEFAULT_VARIABLE_BASE_SIGMA_16K = 3.20
DEFAULT_VARIABLE_MIN_SIGMA_16K = 2.20
DEFAULT_VARIABLE_MAX_SIGMA_16K = 12.50
VARIABLE_RELATIVE_FLUX_MIN = 0.01
VARIABLE_RELATIVE_FLUX_MAX = 100.0
VARIABLE_BRIGHTNESS_RESPONSE_MIN = 0.25
VARIABLE_BRIGHTNESS_RESPONSE_MAX = 6.0
VARIABLE_SIZE_LOG_RESPONSE_MIN = -0.65
VARIABLE_SIZE_LOG_RESPONSE_MAX = 0.75
VARIABLE_BASE_ALPHA = 0.52
VARIABLE_MIN_ALPHA = 0.012
VARIABLE_MAX_ALPHA = 0.95
VARIABLE_AMPLITUDE_MAG_MIN = 0.0
VARIABLE_AMPLITUDE_MAG_MAX = 8.0

GAUSSIAN_MIN_DRAW_SIGMA = 1.0
GAUSSIAN_CUTOFF_SIGMA = 3.5
GAUSSIAN_CORE_GAIN = 2.0


@dataclass(frozen=True)
class RectangularSkyRegion:
    """Geometry written by ``aurora_sky_region_render.py``."""

    width: int
    height: int
    l_center_deg: float
    b_center_deg: float
    l_width_deg: float
    b_height_deg: float

    def __post_init__(self):
        values = np.asarray(
            [
                self.l_center_deg,
                self.b_center_deg,
                self.l_width_deg,
                self.b_height_deg,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("sky-region layout contains non-finite values")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sky-region layout dimensions must be positive")
        if not 0.0 < self.l_width_deg <= 360.0:
            raise ValueError("sky-region longitude width must be in (0, 360]")
        if not 0.0 < self.b_height_deg <= 180.0:
            raise ValueError("sky-region latitude height must be in (0, 180]")
        b_min = self.b_center_deg - self.b_height_deg * 0.5
        b_max = self.b_center_deg + self.b_height_deg * 0.5
        if b_min < -90.0 or b_max > 90.0:
            raise ValueError("sky-region latitude limits extend beyond the sky")

    def signature_array(self):
        """Return stable numeric geometry for cache signatures."""
        return np.asarray(
            [
                self.width,
                self.height,
                self.l_center_deg,
                self.b_center_deg,
                self.l_width_deg,
                self.b_height_deg,
            ],
            dtype=np.float64,
        )


def load_rectangular_sky_region(path):
    """Load and validate a rectangular sky-region layout NPZ."""
    path = Path(path)
    required = {
        "projection",
        "dimensions",
        "l_center_deg",
        "b_center_deg",
        "l_width_deg",
        "b_height_deg",
    }
    try:
        with np.load(path, allow_pickle=False) as layout:
            missing = required - set(layout.files)
            if missing:
                raise KeyError(
                    "missing layout fields: " + ", ".join(sorted(missing))
                )
            projection = str(np.asarray(layout["projection"]).reshape(-1)[0])
            if projection != "rectangular":
                raise ValueError(
                    f"unsupported sky-region projection: {projection}"
                )
            dimensions = np.asarray(layout["dimensions"], dtype=np.int64)
            if dimensions.shape != (2,):
                raise ValueError("sky-region dimensions must contain width, height")
            return RectangularSkyRegion(
                width=int(dimensions[0]),
                height=int(dimensions[1]),
                l_center_deg=float(layout["l_center_deg"]),
                b_center_deg=float(layout["b_center_deg"]),
                l_width_deg=float(layout["l_width_deg"]),
                b_height_deg=float(layout["b_height_deg"]),
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sky-region layout {path}: {error}") from error


def galactic_region_mask(longitude_deg, latitude_deg, region):
    """Select catalogue coordinates inside a rectangular Galactic field."""
    longitude_deg = np.asarray(longitude_deg, dtype=np.float64)
    latitude_deg = np.asarray(latitude_deg, dtype=np.float64)
    if longitude_deg.shape != latitude_deg.shape:
        raise ValueError("longitude and latitude must have matching shapes")
    delta_l = np.remainder(
        longitude_deg - region.l_center_deg + 180.0,
        360.0,
    ) - 180.0
    half_l = region.l_width_deg * 0.5
    half_b = region.b_height_deg * 0.5
    b_min = region.b_center_deg - half_b
    b_max = region.b_center_deg + half_b
    return (
        np.isfinite(longitude_deg)
        & np.isfinite(latitude_deg)
        & (np.abs(delta_l) <= half_l)
        & (latitude_deg >= b_min)
        & (latitude_deg <= b_max)
    )


def find_fits_table_hdu(hdul):
    """Return the first table HDU from an opened FITS file."""
    from astropy.io import fits

    for hdu in hdul:
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return hdu
    raise RuntimeError("FITS file contains no table HDU")


def bp_rp_to_temperature(bp_rp):
    """Estimate effective temperature in Kelvin from Gaia BP-RP colour."""
    colour_index = np.clip(
        np.asarray(bp_rp, dtype=np.float32),
        BP_RP_MIN,
        BP_RP_MAX,
    )
    return (
        4600.0
        * (
            1.0 / (0.92 * colour_index + 1.7)
            + 1.0 / (0.92 * colour_index + 0.62)
        )
    ).astype(np.float32, copy=False)


def temperature_from_columns(
    teff,
    bp_rp,
    fallback=SOLAR_TEMPERATURE_K,
):
    """Use measured temperatures, then BP-RP estimates, then a fallback."""
    measured = np.asarray(teff, dtype=np.float32)
    estimated = bp_rp_to_temperature(bp_rp)
    temperature = np.where(np.isfinite(measured), measured, estimated)
    return np.where(
        np.isfinite(temperature),
        temperature,
        np.float32(fallback),
    ).astype(np.float32, copy=False)


def temperature_to_rgb_channels(temperature):
    """Convert Kelvin temperatures to normalized blackbody-like RGB channels."""
    scaled = (
        np.clip(
            np.asarray(temperature, dtype=np.float32),
            BLACKBODY_TEMPERATURE_MIN_K,
            BLACKBODY_TEMPERATURE_MAX_K,
        )
        / 100.0
    )
    red = np.ones_like(scaled)
    green = np.ones_like(scaled)
    blue = np.ones_like(scaled)

    cool = scaled <= 66.0
    hot = ~cool
    green[cool] = (
        0.3900815787690196 * np.log(scaled[cool])
        - 0.6318414437886275
    )
    red[hot] = 1.292936186062745 * np.power(
        scaled[hot] - 60.0,
        -0.1332047592,
    )
    green[hot] = 1.129890860895294 * np.power(
        scaled[hot] - 60.0,
        -0.0755148492,
    )

    blue[scaled >= 66.0] = 1.0
    blue[scaled <= 19.0] = 0.0
    middle = (scaled > 19.0) & (scaled < 66.0)
    blue[middle] = (
        0.5432067891101961 * np.log(scaled[middle] - 10.0)
        - 1.19625408914
    )
    return (
        np.clip(red, 0.0, 1.0),
        np.clip(green, 0.0, 1.0),
        np.clip(blue, 0.0, 1.0),
    )


def temperature_to_rgb(temperature):
    """Return blackbody-like colours with RGB stored in the final axis."""
    channels = temperature_to_rgb_channels(temperature)
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def star_colors_from_columns(
    teff,
    bp_rp,
    fallback=SOLAR_TEMPERATURE_K,
):
    """Calculate normalized RGB colours directly from Gaia columns."""
    return temperature_to_rgb(
        temperature_from_columns(teff, bp_rp, fallback=fallback)
    )


def variable_star_colors_from_columns(teff, bp_rp):
    """Reproduce the display palette used by the reference variable video."""
    measured = np.asarray(teff, dtype=np.float32)
    colour_index = np.clip(
        np.asarray(bp_rp, dtype=np.float32),
        BP_RP_MIN,
        5.0,
    )
    estimated = (
        4600.0
        * (
            1.0 / (0.92 * colour_index + 1.7)
            + 1.0 / (0.92 * colour_index + 0.62)
        )
    ).astype(np.float32, copy=False)
    temperature = np.where(np.isfinite(measured), measured, estimated)
    temperature = np.where(
        np.isfinite(temperature),
        temperature,
        np.float32(VARIABLE_COLOUR_FALLBACK_TEMPERATURE_K),
    )

    return variable_temperature_to_rgb(temperature)


def variable_temperature_to_rgb(temperature):
    """Map Kelvin values to a vivid palette suited to bright variable stars."""
    temperature = np.asarray(temperature, dtype=np.float32)
    temperature = np.clip(
        temperature,
        VARIABLE_COLOUR_TEMPERATURE_ANCHORS_K[0],
        VARIABLE_COLOUR_TEMPERATURE_ANCHORS_K[-1],
    )
    result = np.empty((temperature.size, 3), dtype=np.float32)
    for channel in range(3):
        result[:, channel] = np.interp(
            temperature,
            VARIABLE_COLOUR_TEMPERATURE_ANCHORS_K,
            VARIABLE_COLOUR_RGB_ANCHORS[:, channel],
        )
    return result


def galactic_to_hammer_pixel(longitude, latitude, image_width, image_height):
    """Map centered, pre-negated Galactic radians to Hammer-map pixels."""
    longitude = np.asarray(longitude, dtype=np.float32)
    latitude = np.asarray(latitude, dtype=np.float32)
    cos_latitude = np.cos(latitude)
    half_longitude = longitude * 0.5
    denominator = np.sqrt(
        1.0 + cos_latitude * np.cos(half_longitude)
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
    ) * 0.5 * image_width
    pixel_y = (
        1.0 - hammer_y / np.sqrt(2.0)
    ) * 0.5 * image_height
    return (
        pixel_x.astype(np.float32, copy=False),
        pixel_y.astype(np.float32, copy=False),
    )


def galactic_to_region_pixel(
    longitude,
    latitude,
    image_width,
    image_height,
    region,
):
    """Project AURORA's internal Galactic radians onto a region raster.

    Animation catalogues store longitude as centered, sign-reversed radians
    for the Hammer renderer.  This function recovers catalogue longitude and
    applies the same decreasing-longitude rectangular mapping as
    ``aurora_sky_region_render.py``.
    """
    longitude = np.asarray(longitude, dtype=np.float32)
    latitude = np.asarray(latitude, dtype=np.float32)
    if longitude.shape != latitude.shape:
        raise ValueError("longitude and latitude must have matching shapes")

    longitude_deg = np.remainder(-np.degrees(longitude), 360.0)
    latitude_deg = np.degrees(latitude)
    visible = galactic_region_mask(longitude_deg, latitude_deg, region)
    delta_l = np.remainder(
        longitude_deg - region.l_center_deg + 180.0,
        360.0,
    ) - 180.0
    half_l = region.l_width_deg * 0.5
    b_max = region.b_center_deg + region.b_height_deg * 0.5
    pixel_x = (
        (half_l - delta_l) * (float(image_width) / region.l_width_deg)
    )
    pixel_y = (
        (b_max - latitude_deg) * (float(image_height) / region.b_height_deg)
    )
    pixel_x = np.clip(pixel_x, 0.0, max(float(image_width - 1), 0.0))
    pixel_y = np.clip(pixel_y, 0.0, max(float(image_height - 1), 0.0))
    return (
        pixel_x.astype(np.float32, copy=False),
        pixel_y.astype(np.float32, copy=False),
        visible,
    )


def moffat_kernel(size=31, alpha=2.0, beta=2.8):
    """Return a normalized two-dimensional Moffat PSF kernel."""
    if size < 1 or size % 2 == 0:
        raise ValueError("Moffat kernel size must be a positive odd number")
    axis = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    kernel = np.power(
        1.0 + (xx * xx + yy * yy) / alpha**2,
        -beta,
    )
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def combined_psf_kernel():
    """Build the common Gaussian plus two-scale Moffat stellar PSF."""
    from scipy.ndimage import gaussian_filter

    small = moffat_kernel(*PSF_SMALL_MOFFAT)
    large = moffat_kernel(*PSF_LARGE_MOFFAT)
    combined = PSF_LARGE_WEIGHT * large
    offset = (large.shape[0] - small.shape[0]) // 2
    combined[
        offset : offset + small.shape[0],
        offset : offset + small.shape[1],
    ] += PSF_SMALL_WEIGHT * small
    combined = gaussian_filter(
        combined,
        sigma=PSF_GAUSSIAN_SIGMA,
        mode="constant",
    )
    combined /= combined.sum()
    return combined.astype(np.float32, copy=False)


def apply_psf(channel, kernel):
    """Convolve a stellar histogram with a PSF using overlap-add."""
    from scipy.signal import oaconvolve

    return oaconvolve(channel, kernel, mode="same").astype(
        np.float32,
        copy=False,
    )


def accumulate_sorted_histograms(
    histograms,
    flat_index,
    flux,
    temperature,
):
    """Accumulate flux, temperature flux and counts using one shared sort."""
    if not flat_index.size:
        return
    order = np.argsort(flat_index)
    sorted_index = flat_index[order]
    starts = np.r_[
        0,
        np.flatnonzero(sorted_index[1:] != sorted_index[:-1]) + 1,
    ]
    unique_index = sorted_index[starts]

    hist_flux, hist_temperature_flux, hist_count = histograms
    hist_flux.reshape(-1)[unique_index] += np.add.reduceat(
        flux[order],
        starts,
    )
    weighted_temperature = flux * temperature
    hist_temperature_flux.reshape(-1)[unique_index] += np.add.reduceat(
        weighted_temperature[order],
        starts,
    )
    hist_count.reshape(-1)[unique_index] += np.diff(
        np.r_[starts, len(sorted_index)]
    )


def sample_finite_column_windows(
    column,
    row_count,
    sample_size,
    window_count=64,
):
    """Sample finite values through contiguous windows of a huge FITS column."""
    row_count = int(row_count)
    sample_size = int(sample_size)
    window_count = int(window_count)
    if row_count <= 0 or sample_size <= 0:
        return np.empty(0, dtype=np.float32)

    rows_per_window = max(1, int(np.ceil(sample_size / max(window_count, 1))))
    if row_count <= sample_size:
        starts = np.array([0], dtype=np.int64)
        rows_per_window = row_count
    else:
        rows_per_window = min(rows_per_window, row_count)
        max_start = row_count - rows_per_window
        starts = np.linspace(
            0,
            max_start,
            num=max(1, window_count),
            dtype=np.int64,
        )

    samples = []
    sample_total = 0
    for start in starts:
        stop = min(int(start) + rows_per_window, row_count)
        values = np.asarray(column[int(start):stop], dtype=np.float32)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        samples.append(values)
        sample_total += len(values)
        if sample_total >= sample_size:
            break

    if not samples:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(samples)[:sample_size]


def _wrapped_distance(cycle, centre):
    """Return signed cyclic distance in the interval [-0.5, 0.5)."""
    return np.remainder(cycle - centre + 0.5, 1.0) - 0.5


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _asymmetric_cycle(cycle, rise_fraction, decay_power=1.0):
    """Unit curve with a quick rise and a slower, tunable decline."""
    x = np.remainder(cycle, 1.0)
    rising = x < rise_fraction
    result = np.empty_like(x, dtype=np.float32)
    result[rising] = _smoothstep(x[rising] / rise_fraction)
    decline = (x[~rising] - rise_fraction) / (1.0 - rise_fraction)
    result[~rising] = np.power(
        np.clip(1.0 - decline, 0.0, 1.0),
        decay_power,
    )
    return result


def _curve_rr(cycle, context):
    signal = _asymmetric_cycle(cycle, 0.16, 0.72)
    return signal, 2.0 * signal - 1.0


def _curve_cepheid(cycle, context):
    signal = _asymmetric_cycle(cycle, 0.32, 0.90)
    signal = _smoothstep(signal)
    return signal, 2.0 * signal - 1.0


def _curve_zz_ceti(cycle, context):
    p1, p2 = context["mode_phase_1"], context["mode_phase_2"]
    wave = (
        np.sin(2.0 * np.pi * cycle)
        + 0.32 * np.sin(2.0 * np.pi * 1.071 * cycle + p1)
        + 0.18 * np.sin(2.0 * np.pi * 0.927 * cycle + p2)
    ) / 1.5
    signal = np.clip(0.5 + 0.5 * wave, 0.0, 1.0)
    return signal, np.zeros_like(signal)


def _curve_lbv(cycle, context):
    p1, p2 = context["mode_phase_1"], context["mode_phase_2"]
    irregularity = context["irregularity"]
    slow = 0.5 + 0.5 * np.sin(2.0 * np.pi * cycle - 0.5 * np.pi)
    micro = (
        0.10 * np.sin(2.0 * np.pi * 7.3 * cycle + p1)
        + 0.06 * np.sin(2.0 * np.pi * 11.7 * cycle + p2)
    )
    envelope = 1.0 + irregularity * 0.25 * np.sin(
        2.0 * np.pi * 0.37 * cycle + p2
    )
    signal = np.clip(0.5 + (slow - 0.5) * envelope + micro, 0.0, 1.0)
    return signal, 2.0 * signal - 1.0


def _random_lattice_values(cells, seeds):
    """Return stable independent values for integer time cells and stars."""
    values = np.asarray(cells, dtype=np.int64).astype(np.uint64)
    values ^= np.asarray(seeds, dtype=np.uint64) + np.uint64(
        0x9E3779B97F4A7C15
    )
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (
        (values >> np.uint64(40)).astype(np.float32)
        / np.float32(2**24)
    )


def _smooth_random_values(position, seeds):
    """Interpolate non-repeating random values without a cyclic period."""
    position = np.asarray(position, dtype=np.float32)
    cells = np.floor(position).astype(np.int64)
    fraction = _smoothstep(position - cells)
    left = _random_lattice_values(cells, seeds)
    right = _random_lattice_values(cells + 1, seeds)
    return left + (right - left) * fraction


def _curve_stochastic_hot(cycle, context):
    """Smooth random brightenings with no fixed or repeating period."""
    del cycle
    progress = np.float32(context["timeline_progress"])
    event_shape = np.asarray(context["event_shape"], dtype=np.float32)
    p1 = np.asarray(context["mode_phase_1"], dtype=np.float32)
    p2 = np.asarray(context["mode_phase_2"], dtype=np.float32)
    seeds = np.asarray(
        np.floor((p1 / np.float32(2.0 * np.pi)) * np.float32(2**24)),
        dtype=np.uint64,
    )
    offset = (p2 / np.float32(2.0 * np.pi)) * np.float32(4096.0)
    changes = (
        STOCHASTIC_HOT_MIN_CHANGES
        + STOCHASTIC_HOT_ADDITIONAL_CHANGES * event_shape
    )
    slow = _smooth_random_values(offset + progress * changes, seeds)
    fast = _smooth_random_values(
        offset * STOCHASTIC_HOT_FAST_OFFSET_SCALE
        + progress * changes * STOCHASTIC_HOT_FAST_CHANGE_SCALE,
        seeds ^ np.uint64(0xD1B54A32D192ED03),
    )
    random_level = (
        STOCHASTIC_HOT_SLOW_WEIGHT * slow
        + STOCHASTIC_HOT_FAST_WEIGHT * fast
    )
    # No dead zone: every hot star fluctuates throughout the animation, while
    # independent lattice values keep the brightenings non-periodic.
    brightening = _smoothstep(random_level).astype(np.float32)
    return brightening, (np.float32(2.0) * brightening - 1.0)


def _curve_cataclysmic(cycle, context):
    x = np.remainder(cycle, 1.0)
    signal = np.zeros_like(x, dtype=np.float32)
    rising = x < CATACLYSMIC_RISE_END
    plateau = (
        (x >= CATACLYSMIC_RISE_END) & (x < CATACLYSMIC_PLATEAU_END)
    )
    decay = (
        (x >= CATACLYSMIC_PLATEAU_END) & (x < CATACLYSMIC_DECAY_END)
    )
    signal[rising] = _smoothstep(x[rising] / CATACLYSMIC_RISE_END)
    signal[plateau] = 1.0
    signal[decay] = np.power(
        1.0
        - (x[decay] - CATACLYSMIC_PLATEAU_END)
        / (CATACLYSMIC_DECAY_END - CATACLYSMIC_PLATEAU_END),
        CATACLYSMIC_DECAY_POWER,
    )
    return signal, signal


def _curve_eclipsing(cycle, context):
    primary = np.exp(-0.5 * np.square(_wrapped_distance(cycle, 0.0) / 0.045))
    secondary = 0.48 * np.exp(
        -0.5 * np.square(_wrapped_distance(cycle, 0.5) / 0.065)
    )
    dip = np.clip(primary + secondary, 0.0, 1.0)
    return dip, np.zeros_like(dip)


def _curve_ellipsoidal(cycle, context):
    signal = 0.5 + 0.5 * np.cos(4.0 * np.pi * cycle)
    return signal, np.zeros_like(signal)


def _curve_multimode(cycle, context):
    p1, p2 = context["mode_phase_1"], context["mode_phase_2"]
    wave = (
        np.sin(2.0 * np.pi * cycle)
        + 0.28 * np.sin(2.0 * np.pi * 1.043 * cycle + p1)
        + 0.16 * np.sin(2.0 * np.pi * 0.968 * cycle + p2)
    ) / 1.44
    signal = np.clip(0.5 + 0.5 * wave, 0.0, 1.0)
    return signal, 2.0 * signal - 1.0


def _curve_long_period(cycle, context):
    p1, p2 = context["mode_phase_1"], context["mode_phase_2"]
    irregularity = context["irregularity"]
    fundamental = np.sin(2.0 * np.pi * cycle)
    harmonics = (
        0.22 * np.sin(4.0 * np.pi * cycle + p1)
        + irregularity * 0.18 * np.sin(2.0 * np.pi * 0.47 * cycle + p2)
    )
    signal = np.clip(0.5 + 0.5 * (fundamental + harmonics) / 1.4, 0.0, 1.0)
    return signal, 2.0 * signal - 1.0


def _curve_rcb(cycle, context):
    progress = context["timeline_progress"]
    centre = context["event_center"]
    width = context["event_width"]
    relative = (progress - centre) / width
    decline = _smoothstep(np.clip(relative / 0.10, 0.0, 1.0))
    recovery = np.exp(-np.clip(relative - 0.10, 0.0, None) / 0.80)
    signal = np.where(relative < 0.0, 0.0, decline * recovery)
    return signal.astype(np.float32), signal.astype(np.float32)


def _curve_microlensing(cycle, context):
    progress = context["timeline_progress"]
    centre = context["event_center"]
    width = context["event_width"]
    impact = 0.12 + 0.58 * context["event_shape"]
    time_offset = (progress - centre) / width
    amplification = paczynski_amplification(
        time_offset,
        np.zeros_like(time_offset),
        np.ones_like(time_offset),
        impact,
    )
    peak = paczynski_amplification(
        np.zeros_like(impact),
        np.zeros_like(impact),
        np.ones_like(impact),
        impact,
    )
    signal = np.divide(
        amplification - 1.0,
        peak - 1.0,
        out=np.zeros_like(amplification, dtype=np.float32),
        where=peak > 1.0,
    )
    signal = np.clip(signal, 0.0, 1.0)
    return signal, np.zeros_like(signal)


def _curve_supernova(cycle, context):
    progress = context["timeline_progress"]
    centre = context["event_center"]
    width = context["event_width"]
    relative = (progress - centre) / width
    signal = np.where(
        relative < 0.0,
        np.exp(np.clip(relative / 0.12, -30.0, 0.0)),
        np.exp(-np.clip(relative, 0.0, None) / 1.35),
    )
    colour_driver = np.where(
        relative < 0.0,
        signal,
        np.clip(1.0 - 0.75 * relative, -0.55, 1.0),
    )
    return signal.astype(np.float32), colour_driver.astype(np.float32)


def _curve_yso(cycle, context):
    p1, p2 = context["mode_phase_1"], context["mode_phase_2"]
    smooth = (
        0.28 * np.sin(2.0 * np.pi * cycle + p1)
        + 0.16 * np.sin(2.0 * np.pi * 0.43 * cycle + p2)
    )
    burst_centre = np.remainder(p1 / (2.0 * np.pi), 1.0)
    dip_centre = np.remainder(p2 / (2.0 * np.pi), 1.0)
    burst = 0.75 * np.exp(
        -0.5 * np.square(_wrapped_distance(cycle, burst_centre) / 0.035)
    )
    dip = np.exp(
        -0.5 * np.square(_wrapped_distance(cycle, dip_centre) / 0.075)
    )
    signal = np.clip(smooth + burst - 0.85 * dip, -1.0, 1.0)
    return signal, dip


def _curve_rotational(cycle, context):
    p1 = context["mode_phase_1"]
    wave = (
        np.sin(2.0 * np.pi * cycle)
        + 0.24 * np.sin(4.0 * np.pi * cycle + p1)
    ) / 1.24
    signal = np.clip(0.5 + 0.5 * wave, 0.0, 1.0)
    return signal, np.zeros_like(signal)


def _curve_gentle(cycle, context):
    p1 = context["mode_phase_1"]
    wave = (
        np.sin(2.0 * np.pi * cycle)
        + 0.12 * np.sin(4.0 * np.pi * cycle + p1)
    ) / 1.12
    signal = np.clip(0.5 + 0.5 * wave, 0.0, 1.0)
    return signal, 2.0 * signal - 1.0


def _curve_solar(cycle, context):
    signal = 0.5 + 0.5 * np.sin(2.0 * np.pi * cycle)
    return signal, np.zeros_like(signal)


def _curve_agn(cycle, context):
    elapsed = context["elapsed"]
    period = context["period"]
    p1, p2 = context["mode_phase_1"], context["mode_phase_2"]
    time_scale = elapsed / np.maximum(period, np.float32(1.0e-4))
    signal = (
        0.62 * np.sin(2.0 * np.pi * 0.173 * time_scale + p1)
        + 0.27 * np.sin(2.0 * np.pi * 0.071 * time_scale + p2)
        + 0.11 * np.sin(2.0 * np.pi * 0.019 * time_scale + p1 + p2)
    )
    return np.clip(signal, -1.0, 1.0), np.zeros_like(signal)


def _curve_planet_transit(cycle, context):
    dip = np.exp(-0.5 * np.square(_wrapped_distance(cycle, 0.0) / 0.025))
    return dip, np.zeros_like(dip)


def _curve_conservative(cycle, context):
    signal = 0.5 + 0.5 * np.sin(2.0 * np.pi * cycle)
    return signal, np.zeros_like(signal)


_VARIABLE_CURVE_FUNCTIONS = {
    VARIABLE_MODEL_CODES["rr_lyrae"]: _curve_rr,
    VARIABLE_MODEL_CODES["cepheid"]: _curve_cepheid,
    VARIABLE_MODEL_CODES["zz_ceti"]: _curve_zz_ceti,
    VARIABLE_MODEL_CODES["lbv"]: _curve_lbv,
    VARIABLE_MODEL_CODES["cataclysmic"]: _curve_cataclysmic,
    VARIABLE_MODEL_CODES["eclipsing"]: _curve_eclipsing,
    VARIABLE_MODEL_CODES["ellipsoidal"]: _curve_ellipsoidal,
    VARIABLE_MODEL_CODES["multimode"]: _curve_multimode,
    VARIABLE_MODEL_CODES["long_period"]: _curve_long_period,
    VARIABLE_MODEL_CODES["rcb_decline"]: _curve_rcb,
    VARIABLE_MODEL_CODES["microlensing_event"]: _curve_microlensing,
    VARIABLE_MODEL_CODES["supernova_event"]: _curve_supernova,
    VARIABLE_MODEL_CODES["yso_irregular"]: _curve_yso,
    VARIABLE_MODEL_CODES["rotational"]: _curve_rotational,
    VARIABLE_MODEL_CODES["gentle_pulsation"]: _curve_gentle,
    VARIABLE_MODEL_CODES["solar_like"]: _curve_solar,
    VARIABLE_MODEL_CODES["agn_noise"]: _curve_agn,
    VARIABLE_MODEL_CODES["planet_transit"]: _curve_planet_transit,
    VARIABLE_MODEL_CODES["conservative"]: _curve_conservative,
    VARIABLE_MODEL_CODES["stochastic_hot"]: _curve_stochastic_hot,
}


def variable_light_curve(model_code, cycle, context):
    """Evaluate one class-group curve without looping over individual stars."""
    try:
        evaluator = _VARIABLE_CURVE_FUNCTIONS[int(model_code)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"unsupported variable model code: {model_code}") from error
    return evaluator(np.asarray(cycle, dtype=np.float32), context)


def _relative_flux(signal, amplitude_mag, flux_mode):
    """Convert a magnitude amplitude and normalized curve to relative flux."""
    amplitude_mag = np.clip(
        np.asarray(amplitude_mag, dtype=np.float32),
        VARIABLE_AMPLITUDE_MAG_MIN,
        VARIABLE_AMPLITUDE_MAG_MAX,
    )
    flux_ratio = np.power(10.0, 0.4 * amplitude_mag).astype(np.float32)
    signal = np.asarray(signal, dtype=np.float32)
    if flux_mode == VARIABLE_FLUX_MODES["range"]:
        return np.exp((signal - 0.5) * np.log(flux_ratio))
    if flux_mode == VARIABLE_FLUX_MODES["outburst"]:
        return 1.0 + (flux_ratio - 1.0) * np.clip(signal, 0.0, 1.0)
    if flux_mode == VARIABLE_FLUX_MODES["dimming"]:
        depth = 1.0 - 1.0 / flux_ratio
        return 1.0 - depth * np.clip(signal, 0.0, 1.0)
    if flux_mode == VARIABLE_FLUX_MODES["signed"]:
        return np.exp(0.5 * signal * np.log(flux_ratio))
    if flux_mode == VARIABLE_FLUX_MODES["supernova"]:
        return 0.035 + (flux_ratio - 0.035) * np.clip(signal, 0.0, 1.0)
    raise ValueError(f"unsupported variable flux mode: {flux_mode}")


def variable_visual_parameters(
    relative_flux,
    size_scale,
    size_response,
    intensity_scale,
    brightness_response,
    image_height,
    *,
    base_sigma_16k=DEFAULT_VARIABLE_BASE_SIGMA_16K,
    min_sigma_16k=DEFAULT_VARIABLE_MIN_SIGMA_16K,
    max_sigma_16k=DEFAULT_VARIABLE_MAX_SIGMA_16K,
):
    """Map physical flux to a bounded, class-sensitive display response.

    ``brightness_response`` is a display transfer exponent.  A value above one
    makes a small but real catalogue amplitude legible without replacing that
    amplitude: stars with different amplitudes still retain their individual
    flux ratios and ordering.
    """
    relative_flux = np.clip(
        np.asarray(relative_flux, dtype=np.float32),
        VARIABLE_RELATIVE_FLUX_MIN,
        VARIABLE_RELATIVE_FLUX_MAX,
    )
    resolution_scale = image_height / REFERENCE_MAP_HEIGHT
    log_flux = np.log(relative_flux)
    brightness_response = np.clip(
        np.asarray(brightness_response, dtype=np.float32),
        VARIABLE_BRIGHTNESS_RESPONSE_MIN,
        VARIABLE_BRIGHTNESS_RESPONSE_MAX,
    )
    display_flux_root = np.exp(
        0.5 * brightness_response * log_flux
    )
    size_factor = np.exp(
        np.clip(
            np.asarray(size_response, dtype=np.float32) * log_flux,
            VARIABLE_SIZE_LOG_RESPONSE_MIN,
            VARIABLE_SIZE_LOG_RESPONSE_MAX,
        )
    )
    sigma = (
        base_sigma_16k
        * size_scale
        * size_factor
        * resolution_scale
    )
    sigma = np.clip(
        sigma,
        min_sigma_16k * resolution_scale,
        max_sigma_16k * resolution_scale,
    )
    alpha = np.clip(
        VARIABLE_BASE_ALPHA * intensity_scale * display_flux_root,
        VARIABLE_MIN_ALPHA,
        VARIABLE_MAX_ALPHA,
    )
    return sigma.astype(np.float32), alpha.astype(np.float32)


def variable_frame_parameters(
    *,
    elapsed,
    frame=None,
    frame_count=None,
    timeline_progress=None,
    timeline_opacity=1.0,
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
    image_height,
    base_sigma_16k=DEFAULT_VARIABLE_BASE_SIGMA_16K,
    min_sigma_16k=DEFAULT_VARIABLE_MIN_SIGMA_16K,
    max_sigma_16k=DEFAULT_VARIABLE_MAX_SIGMA_16K,
    intro_fraction,
    outro_fraction,
    fade_in_time,
    fade_out_time,
):
    """Calculate sparse geometry plus sparse dynamic colours for one frame."""
    count = len(period)
    relative_flux = np.ones(count, dtype=np.float32)
    colour_driver = np.zeros(count, dtype=np.float32)
    two_pi = np.float32(2.0 * np.pi)

    # model_groups contains indices prepared once before multiprocessing.  The
    # loop is over variability models, while all stars in a model are handled
    # by vectorized NumPy expressions.
    for model_code, group_flux_mode, indices in model_groups:
        indices = np.asarray(indices, dtype=np.int32)
        cycle = np.remainder(
            phase[indices] / two_pi + elapsed / period[indices],
            1.0,
        ).astype(np.float32)
        context = {
            "elapsed": np.float32(elapsed),
            "timeline_progress": np.float32(
                float(timeline_progress) if timeline_progress is not None else 0.0
            ),
            "period": period[indices],
            "irregularity": irregularity[indices],
            "mode_phase_1": mode_phase_1[indices],
            "mode_phase_2": mode_phase_2[indices],
            "event_center": event_center[indices],
            "event_width": event_width[indices],
            "event_shape": event_shape[indices],
        }
        signal, group_colour_driver = variable_light_curve(
            model_code,
            cycle,
            context,
        )
        relative_flux[indices] = _relative_flux(
            signal,
            amplitude[indices],
            int(group_flux_mode),
        )
        colour_driver[indices] = group_colour_driver

    sigma, alpha = variable_visual_parameters(
        relative_flux,
        size_scale,
        size_response,
        intensity_scale,
        brightness_response,
        image_height,
        base_sigma_16k=base_sigma_16k,
        min_sigma_16k=min_sigma_16k,
        max_sigma_16k=max_sigma_16k,
    )

    if timeline_progress is None:
        if frame is None or frame_count is None:
            raise ValueError(
                "frame and frame_count are required without timeline_progress"
            )
        fraction = frame / max(frame_count - 1, 1)
    else:
        fraction = float(timeline_progress)
    appearance = np.clip(
        (fraction - intro_fraction - appearance_delay) / fade_in_time,
        0.0,
        1.0,
    )
    outro_start = 1.0 - outro_fraction
    fade_out = np.clip(
        (outro_start + fade_delay - fraction) / fade_out_time,
        0.0,
        1.0,
    )
    alpha = (
        alpha
        * appearance
        * fade_out
        * np.float32(timeline_opacity)
    ).astype(np.float32)
    visible = alpha > 0.0
    indices = np.flatnonzero(visible).astype(np.int32)

    dynamic = visible & (temperature_response != 0.0)
    color_indices = np.flatnonzero(dynamic).astype(np.int32)
    if color_indices.size:
        temperature_factor = np.clip(
            1.0
            + temperature_response[color_indices]
            * colour_driver[color_indices],
            0.52,
            1.45,
        )
        color_function = (
            variable_temperature_to_rgb
            if vivid_temperature_colors
            else temperature_to_rgb
        )
        dynamic_colors = color_function(
            temperature_base[color_indices] * temperature_factor
        )
    else:
        dynamic_colors = np.empty((0, 3), dtype=np.float32)
    return (
        indices,
        sigma[visible],
        alpha[visible],
        color_indices,
        dynamic_colors,
    )


def paczynski_amplification(time, peak_time, event_time, impact):
    """Return point-source, point-lens Paczynski amplification."""
    distance_squared = impact * impact + np.square(
        (time - peak_time) / event_time
    )
    distance_squared = np.maximum(
        distance_squared,
        np.finfo(np.float32).tiny,
    )
    return (distance_squared + 2.0) / np.sqrt(
        distance_squared * (distance_squared + 4.0)
    )


def microlensing_visual_parameters(
    time,
    peak_time,
    event_time,
    impact,
    mass,
    image_height,
    *,
    source_fraction=1.0,
    base_sigma_16k=2.0,
    base_alpha=0.2,
):
    """Map a blended Paczynski curve to a star-sized Gaussian overlay."""
    amplification = paczynski_amplification(
        time,
        peak_time,
        event_time,
        impact,
    )
    source_fraction = np.clip(
        np.asarray(source_fraction, dtype=np.float32),
        0.0,
        1.0,
    )
    strength = source_fraction * np.clip(
        amplification - 1.0,
        0.0,
        None,
    )
    peak_strength = np.clip(
        source_fraction
        * (
            paczynski_amplification(
                peak_time,
                peak_time,
                event_time,
                impact,
            )
            - 1.0
        ),
        0.0,
        None,
    )
    visual_progress = np.divide(
        np.log1p(strength),
        np.log1p(peak_strength),
        out=np.zeros_like(strength, dtype=np.float32),
        where=peak_strength > 0.0,
    )
    visual_progress = np.clip(visual_progress, 0.0, 1.0)

    clipped_mass = np.clip(mass, 0.3, 5.0)
    legacy_scale = image_height / 4320.0
    peak_size_pt2 = 10.0 + 10000.0 * np.log1p(peak_strength)
    peak_sigma = (
        np.sqrt(peak_size_pt2 / np.pi)
        * (200.0 / 72.0)
        / 3.0
        * clipped_mass
        * legacy_scale
    )
    peak_sigma = np.clip(peak_sigma, 2.0, image_height * 0.08)

    resolution_scale = image_height / REFERENCE_MAP_HEIGHT
    base_sigma = base_sigma_16k * resolution_scale
    sigma = base_sigma + (
        peak_sigma - base_sigma
    ) * visual_progress

    peak_alpha = np.clip(
        0.5
        + 0.5 * np.log1p(peak_strength) / np.log(10.0),
        0.5,
        1.0,
    )
    alpha = base_alpha + (
        peak_alpha - base_alpha
    ) * visual_progress
    return sigma.astype(np.float32), alpha.astype(np.float32)


def draw_gaussians_u8(
    canvas,
    pixel_x,
    pixel_y,
    sigma,
    alpha,
    colors,
):
    """Add temperature-coloured Gaussian spots to an RGB uint8 image."""
    if canvas.dtype != np.uint8 or canvas.ndim != 3 or canvas.shape[2] != 3:
        raise TypeError("canvas must be an H×W×3 uint8 array")

    image_height, image_width = canvas.shape[:2]
    for center_x, center_y, spot_sigma, spot_alpha, color in zip(
        pixel_x,
        pixel_y,
        sigma,
        alpha,
        colors,
    ):
        spot_sigma = max(float(spot_sigma), GAUSSIAN_MIN_DRAW_SIGMA)
        spot_alpha = float(spot_alpha)
        color = np.asarray(color, dtype=np.float32)
        if (
            spot_alpha <= 0.0
            or not np.isfinite(
                center_x + center_y + spot_sigma + spot_alpha
            )
            or not np.all(np.isfinite(color))
        ):
            continue

        cut = int(GAUSSIAN_CUTOFF_SIGMA * spot_sigma) + 1
        x0 = max(0, int(center_x) - cut)
        x1 = min(image_width, int(center_x) + cut + 1)
        y0 = max(0, int(center_y) - cut)
        y1 = min(image_height, int(center_y) + cut + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        radius_squared = (
            np.square(xx - center_x) + np.square(yy - center_y)
        )
        gaussian = np.exp(
            -radius_squared / (2.0 * spot_sigma * spot_sigma),
            dtype=np.float32,
        )
        gaussian = np.minimum(gaussian * GAUSSIAN_CORE_GAIN, 1.0)
        gaussian *= spot_alpha

        spot = np.clip(
            gaussian[:, :, np.newaxis] * color * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        patch = canvas[y0:y1, x0:x1]
        added = np.add(patch, spot, dtype=np.uint16)
        np.minimum(added, 255, out=added)
        patch[:] = added


def array_signature(config_values, *arrays, digest_size=12):
    """Fingerprint numerical configuration and arrays for cache validation."""
    digest = hashlib.blake2b(digest_size=digest_size)
    config = np.ascontiguousarray(config_values)
    digest.update(config.dtype.str.encode("ascii"))
    digest.update(np.asarray(config.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(config).cast("B"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(
            np.asarray(contiguous.shape, dtype=np.int64).tobytes()
        )
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()
