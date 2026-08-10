"""Validated progenitor, composition, and simulation-input data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .units import Distance


SUPPORTED_SUPERNOVA_TYPES = ("II-P", "II-L", "IIn", "Ib", "Ic", "Ia")


def _finite_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    valid = math.isfinite(value) and (value >= 0.0 if allow_zero else value > 0.0)
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


@dataclass(frozen=True)
class ChemicalComposition:
    """Progenitor mass fractions; missing species are allowed."""

    mass_fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        normalized = {str(key): float(value) for key, value in self.mass_fractions.items()}
        if not normalized:
            raise ValueError("chemical composition cannot be empty")
        for species, fraction in normalized.items():
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"invalid mass fraction for {species}: {fraction}")
        total = sum(normalized.values())
        if not 0.98 <= total <= 1.02:
            raise ValueError(f"chemical mass fractions must sum to about 1 (got {total:g})")
        normalized = {key: value / total for key, value in normalized.items()}
        object.__setattr__(self, "mass_fractions", normalized)

    @property
    def hydrogen_fraction(self) -> float:
        return float(self.mass_fractions.get("H", 0.0))

    @property
    def helium_fraction(self) -> float:
        return float(self.mass_fractions.get("He", 0.0))


@dataclass(frozen=True)
class Progenitor:
    """Physical inputs immediately before the modeled explosion."""

    name: str
    initial_mass_solar: float
    final_mass_solar: float
    metallicity: float
    radius_solar: float
    star_type: str
    age_years: float
    composition: ChemicalComposition
    total_mass_lost_solar: float
    mass_loss_rate_solar_per_year: float
    distance: Distance
    supernova_type: str
    extinction_av_mag: float = 0.0
    galactic_longitude_deg: float = 0.0
    galactic_latitude_deg: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("progenitor name cannot be empty")
        for name in ("initial_mass_solar", "final_mass_solar", "radius_solar", "age_years"):
            _finite_positive(name, float(getattr(self, name)))
        for name in ("metallicity", "total_mass_lost_solar", "mass_loss_rate_solar_per_year", "extinction_av_mag"):
            _finite_positive(name, float(getattr(self, name)), allow_zero=True)
        if self.final_mass_solar > self.initial_mass_solar * 1.05:
            raise ValueError("final mass cannot materially exceed initial mass")
        if self.total_mass_lost_solar > self.initial_mass_solar * 1.05:
            raise ValueError("total mass loss cannot exceed initial mass")
        if self.supernova_type not in SUPPORTED_SUPERNOVA_TYPES:
            raise ValueError(
                f"unsupported supernova type {self.supernova_type!r}; "
                f"choose one of {', '.join(SUPPORTED_SUPERNOVA_TYPES)}"
            )
        if self.supernova_type in {"II-P", "II-L", "IIn"} and self.composition.hydrogen_fraction < 0.01:
            raise ValueError("hydrogen-rich Type II models require hydrogen in the envelope")
        if self.supernova_type in {"Ib", "Ic"} and self.composition.hydrogen_fraction > 0.1:
            raise ValueError("stripped-envelope Ib/Ic models require little hydrogen")
        if not 0.0 <= self.galactic_longitude_deg <= 360.0:
            raise ValueError("galactic longitude must be in [0, 360] degrees")
        if not -90.0 <= self.galactic_latitude_deg <= 90.0:
            raise ValueError("galactic latitude must be in [-90, 90] degrees")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Progenitor":
        values = dict(data)
        values["composition"] = ChemicalComposition(values["composition"])
        values["distance"] = Distance(**values["distance"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["composition"] = dict(self.composition.mass_fractions)
        result["distance"] = self.distance.to_dict()
        return result


@dataclass(frozen=True)
class ModelOverrides:
    """Optional calibration knobs; absent values are derived from the star."""

    energy_foe: float | None = None
    nickel_mass_solar: float | None = None
    remnant_mass_solar: float | None = None
    opacity_m2_kg: float | None = None
    gamma_opacity_m2_kg: float | None = None
    diffusion_time_days: float | None = None
    plateau_scale: float = 1.0
    csm_efficiency: float = 0.3
    temperature_floor_k: float | None = None
    luminosity_scale: float = 1.0
    shock_breakout_enabled: bool = True
    shock_breakout_luminosity_w: float | None = None
    shock_breakout_duration_hours: float = 1.0
    shock_breakout_temperature_k: float = 300_000.0
    light_echo_delay_days: float = 0.0
    light_echo_width_days: float = 30.0
    light_echo_reflection_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "energy_foe", "nickel_mass_solar", "remnant_mass_solar",
            "opacity_m2_kg", "gamma_opacity_m2_kg", "diffusion_time_days",
            "temperature_floor_k", "shock_breakout_luminosity_w",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_positive(name, float(value), allow_zero=name == "remnant_mass_solar")
        _finite_positive("plateau_scale", self.plateau_scale, allow_zero=True)
        _finite_positive("luminosity_scale", self.luminosity_scale)
        _finite_positive(
            "shock_breakout_duration_hours", self.shock_breakout_duration_hours
        )
        _finite_positive(
            "shock_breakout_temperature_k", self.shock_breakout_temperature_k
        )
        _finite_positive("light_echo_delay_days", self.light_echo_delay_days, allow_zero=True)
        _finite_positive("light_echo_width_days", self.light_echo_width_days)
        if not 0.0 <= self.csm_efficiency <= 1.0:
            raise ValueError("csm_efficiency must be between 0 and 1")
        if not 0.0 <= self.light_echo_reflection_fraction <= 0.2:
            raise ValueError("light_echo_reflection_fraction must be between 0 and 0.2")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ModelOverrides":
        return cls(**dict(data or {}))
