"""Educational multi-band photometry for the semi-analytic light curves.

The source is approximated as a blackbody and each broad filter as a top-hat
bandpass.  Magnitudes are reported on the AB system.  This is deliberately
separate from bolometric photometry: no bolometric luminosity is used directly
as a visual magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .constants import (
    BOLTZMANN_CONSTANT_J_K,
    PLANCK_CONSTANT_J_S,
    SPEED_OF_LIGHT_M_S,
    STEFAN_BOLTZMANN_W_M2_K4,
)
from .units import Distance


AB_ZERO_POINT_W_M2_HZ = 3631.0e-26


@dataclass(frozen=True)
class FilterBand:
    name: str
    wavelength_min_nm: float
    wavelength_max_nm: float
    extinction_over_av: float
    description: str

    @property
    def wavelength_min_m(self) -> float:
        return self.wavelength_min_nm * 1.0e-9

    @property
    def wavelength_max_m(self) -> float:
        return self.wavelength_max_nm * 1.0e-9


# Broad educational passbands and approximate R_V=3.1 extinction ratios.
# They intentionally do not claim to reproduce a specific instrument response.
FILTERS: Mapping[str, FilterBand] = {
    "UV": FilterBand("UV", 150.0, 300.0, 2.65, "near/far-UV educational band"),
    "U": FilterBand("U", 300.0, 400.0, 1.56, "Johnson-like U"),
    "B": FilterBand("B", 390.0, 490.0, 1.32, "Johnson-like B"),
    "V": FilterBand("V", 500.0, 600.0, 1.00, "Johnson-like V"),
    "R": FilterBand("R", 580.0, 750.0, 0.82, "Cousins-like R"),
    "I": FilterBand("I", 700.0, 900.0, 0.60, "Cousins-like I"),
    "IR": FilterBand("IR", 900.0, 2500.0, 0.28, "broad near-infrared educational band"),
}


@dataclass(frozen=True)
class BandPhotometry:
    filter_name: str
    intrinsic_flux_w_m2: np.ndarray
    observed_flux_w_m2: np.ndarray
    intrinsic_apparent_magnitude: np.ndarray
    apparent_magnitude: np.ndarray
    absolute_magnitude: np.ndarray
    extinction_mag: float


def _blackbody_band_fraction(
    temperature_k: np.ndarray,
    band: FilterBand,
    redshift: float,
    *,
    samples: int = 48,
) -> tuple[np.ndarray, float]:
    """Return bolometric fraction in the rest wavelengths seen by a filter."""
    temperature = np.maximum(np.asarray(temperature_k, dtype=np.float64), 100.0)
    rest_min = band.wavelength_min_m / (1.0 + redshift)
    rest_max = band.wavelength_max_m / (1.0 + redshift)
    wavelength = np.linspace(rest_min, rest_max, samples, dtype=np.float64)
    exponent = (
        PLANCK_CONSTANT_J_S
        * SPEED_OF_LIGHT_M_S
        / (wavelength[:, None] * BOLTZMANN_CONSTANT_J_K * temperature[None, :])
    )
    denominator = np.expm1(np.clip(exponent, 1.0e-8, 700.0))
    spectral_radiance = (
        2.0
        * PLANCK_CONSTANT_J_S
        * SPEED_OF_LIGHT_M_S**2
        / (wavelength[:, None] ** 5 * denominator)
    )
    normalized_per_meter = (
        math.pi
        * spectral_radiance
        / (STEFAN_BOLTZMANN_W_M2_K4 * temperature[None, :] ** 4)
    )
    fraction = np.trapezoid(normalized_per_meter, wavelength, axis=0)
    observed_frequency_width = SPEED_OF_LIGHT_M_S * (
        1.0 / band.wavelength_min_m - 1.0 / band.wavelength_max_m
    )
    return np.clip(fraction, 0.0, 1.0), observed_frequency_width


def band_photometry(
    luminosity_w: np.ndarray,
    temperature_k: np.ndarray,
    distance: Distance,
    extinction_av_mag: float,
    band: FilterBand,
    *,
    redshift: float = 0.0,
    extra_luminosity_w: np.ndarray | None = None,
    extra_temperature_k: float | np.ndarray | None = None,
) -> BandPhotometry:
    """Convert blackbody components to intrinsic and extincted AB photometry."""
    luminosity = np.asarray(luminosity_w, dtype=np.float64)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if luminosity.shape != temperature.shape:
        raise ValueError("luminosity and temperature arrays must have the same shape")
    fraction, frequency_width = _blackbody_band_fraction(temperature, band, redshift)
    band_luminosity = luminosity * fraction
    if extra_luminosity_w is not None:
        extra_luminosity = np.asarray(extra_luminosity_w, dtype=np.float64)
        if extra_luminosity.shape != luminosity.shape or extra_temperature_k is None:
            raise ValueError("extra component must match the main light-curve shape and temperature")
        extra_temperature = np.broadcast_to(
            np.asarray(extra_temperature_k, dtype=np.float64), luminosity.shape
        )
        extra_fraction, _ = _blackbody_band_fraction(extra_temperature, band, redshift)
        band_luminosity = band_luminosity + extra_luminosity * extra_fraction

    intrinsic_integrated_flux = band_luminosity / (4.0 * math.pi * distance.meters**2)
    intrinsic_fnu = intrinsic_integrated_flux / frequency_width
    intrinsic_mag = -2.5 * np.log10(
        np.maximum(intrinsic_fnu, np.finfo(np.float64).tiny) / AB_ZERO_POINT_W_M2_HZ
    )
    extinction_mag = band.extinction_over_av * float(extinction_av_mag)
    attenuation = 10.0 ** (-0.4 * extinction_mag)
    observed_integrated_flux = intrinsic_integrated_flux * attenuation
    apparent_mag = intrinsic_mag + extinction_mag

    absolute_distance_modulus = distance.distance_modulus
    absolute_mag = intrinsic_mag - absolute_distance_modulus
    return BandPhotometry(
        filter_name=band.name,
        intrinsic_flux_w_m2=intrinsic_integrated_flux,
        observed_flux_w_m2=observed_integrated_flux,
        intrinsic_apparent_magnitude=intrinsic_mag,
        apparent_magnitude=apparent_mag,
        absolute_magnitude=absolute_mag,
        extinction_mag=extinction_mag,
    )


def photometry_metadata() -> dict[str, object]:
    return {
        "system": "AB",
        "model": "blackbody SED integrated through educational top-hat passbands",
        "warning": (
            "Educational approximation: no line blanketing, spectral features, "
            "instrument throughput, Vega conversion, or full K-correction."
        ),
        "filters": {
            name: {
                "wavelength_nm": [band.wavelength_min_nm, band.wavelength_max_nm],
                "extinction_over_av": band.extinction_over_av,
                "description": band.description,
            }
            for name, band in FILTERS.items()
        },
    }
