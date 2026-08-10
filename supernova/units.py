"""Distance units and a small flat-Lambda-CDM luminosity-distance helper."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import (
    HUBBLE_KM_S_MPC,
    LIGHT_YEAR_M,
    OMEGA_LAMBDA,
    OMEGA_M,
    PARSEC_M,
    SPEED_OF_LIGHT_M_S,
)


_UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "ly": LIGHT_YEAR_M,
    "light_year": LIGHT_YEAR_M,
    "light_years": LIGHT_YEAR_M,
    "pc": PARSEC_M,
    "parsec": PARSEC_M,
    "parsecs": PARSEC_M,
    "kpc": 1.0e3 * PARSEC_M,
    "mpc": 1.0e6 * PARSEC_M,
}


@dataclass(frozen=True)
class Distance:
    """A positive luminosity distance with an optional measured redshift."""

    value: float
    unit: str = "pc"
    redshift: float | None = None

    def __post_init__(self) -> None:
        unit = str(self.unit).strip().casefold().replace("-", "_").replace(" ", "_")
        if unit not in _UNIT_TO_METERS:
            raise ValueError(f"unsupported distance unit: {self.unit!r}")
        if not math.isfinite(self.value) or self.value <= 0.0:
            raise ValueError("distance must be finite and positive")
        if self.redshift is not None and (
            not math.isfinite(self.redshift) or self.redshift < 0.0
        ):
            raise ValueError("redshift must be finite and non-negative")
        object.__setattr__(self, "unit", unit)

    @property
    def meters(self) -> float:
        return self.value * _UNIT_TO_METERS[self.unit]

    @property
    def parsecs(self) -> float:
        return self.meters / PARSEC_M

    @property
    def megaparsecs(self) -> float:
        return self.parsecs / 1.0e6

    @property
    def distance_modulus(self) -> float:
        return 5.0 * math.log10(self.parsecs / 10.0)

    @property
    def effective_redshift(self) -> float:
        if self.redshift is not None:
            return self.redshift
        # Peculiar velocities dominate at small distances.  Do not manufacture
        # a cosmological correction for Galactic and Local Group objects.
        if self.megaparsecs < 10.0:
            return 0.0
        return redshift_from_luminosity_distance(self.meters)

    def to_dict(self) -> dict[str, float | str | None]:
        return {"value": self.value, "unit": self.unit, "redshift": self.redshift}


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(96)


def luminosity_distance_m(redshift: float) -> float:
    """Return luminosity distance for a flat Lambda-CDM cosmology.

    A fixed Gauss-Legendre quadrature keeps this dependency-free and accurate
    enough for visualization over 0 <= z <= 100.
    """
    z = float(redshift)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError("redshift must be finite and non-negative")
    if z == 0.0:
        return 0.0
    sample_z = 0.5 * z * (_GL_NODES + 1.0)
    expansion = np.sqrt(OMEGA_M * np.power(1.0 + sample_z, 3.0) + OMEGA_LAMBDA)
    integral = 0.5 * z * float(np.sum(_GL_WEIGHTS / expansion))
    hubble_si = HUBBLE_KM_S_MPC * 1000.0 / (1.0e6 * PARSEC_M)
    comoving = SPEED_OF_LIGHT_M_S * integral / hubble_si
    return (1.0 + z) * comoving


def redshift_from_luminosity_distance(distance_m: float) -> float:
    """Invert luminosity distance monotonically by bisection."""
    target = float(distance_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("luminosity distance must be finite and positive")
    lower, upper = 0.0, 1.0
    while luminosity_distance_m(upper) < target and upper < 100.0:
        upper *= 2.0
    if luminosity_distance_m(upper) < target:
        raise ValueError("distance exceeds the supported z <= 100 cosmology")
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if luminosity_distance_m(middle) < target:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def flux_from_luminosity(luminosity_w: np.ndarray | float, distance: Distance):
    """Apply the inverse-square law using luminosity distance."""
    luminosity = np.asarray(luminosity_w, dtype=np.float64)
    if np.any(~np.isfinite(luminosity)) or np.any(luminosity < 0.0):
        raise ValueError("luminosity must be finite and non-negative")
    return luminosity / (4.0 * math.pi * distance.meters**2)
    

