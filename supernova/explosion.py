"""Derive explosion-scale quantities from a validated progenitor."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import DAY_S, FOE_J, SOLAR_MASS_KG, SPEED_OF_LIGHT_M_S
from .progenitor import ModelOverrides, Progenitor


@dataclass(frozen=True)
class ExplosionParameters:
    energy_j: float
    ejecta_mass_kg: float
    remnant_mass_kg: float
    nickel56_mass_kg: float
    characteristic_velocity_m_s: float
    diffusion_time_days: float
    gamma_trapping_time_days: float
    opacity_m2_kg: float
    gamma_opacity_m2_kg: float

    @property
    def energy_foe(self) -> float:
        return self.energy_j / FOE_J

    @property
    def ejecta_mass_solar(self) -> float:
        return self.ejecta_mass_kg / SOLAR_MASS_KG

    @property
    def nickel56_mass_solar(self) -> float:
        return self.nickel56_mass_kg / SOLAR_MASS_KG


_TYPE_DEFAULTS = {
    # energy [foe], Ni-56 [Msun], opacity [m2/kg], remnant [Msun]
    "II-P": (1.0, 0.055, 0.034, 1.50),
    "II-L": (1.2, 0.070, 0.030, 1.55),
    "IIn": (1.5, 0.080, 0.034, 1.70),
    "Ib": (1.2, 0.090, 0.012, 1.50),
    "Ic": (1.5, 0.120, 0.010, 1.60),
    "Ia": (1.3, 0.600, 0.010, 0.00),
}


def derive_explosion_parameters(
    progenitor: Progenitor,
    overrides: ModelOverrides | None = None,
) -> ExplosionParameters:
    """Create a bounded semi-analytic explosion calibration.

    The velocity uses the homologous uniform-density relation
    ``E = 3 M v^2 / 10`` and the diffusion time uses the Arnett scale
    ``sqrt(2 kappa M / (beta c v))`` with beta = 13.8.
    """
    overrides = overrides or ModelOverrides()
    base_energy, base_nickel, base_opacity, base_remnant = _TYPE_DEFAULTS[
        progenitor.supernova_type
    ]

    if progenitor.supernova_type == "Ia":
        ejecta_solar = progenitor.final_mass_solar
        remnant_solar = 0.0
        energy_foe = base_energy
        nickel_solar = min(base_nickel, 0.85 * ejecta_solar)
    else:
        remnant_solar = min(base_remnant, 0.8 * progenitor.final_mass_solar)
        mass_factor = np.clip((progenitor.initial_mass_solar / 15.0) ** 0.22, 0.65, 1.8)
        metallicity_factor = np.clip(1.0 - 1.5 * (progenitor.metallicity - 0.014), 0.7, 1.25)
        energy_foe = base_energy * float(mass_factor)
        ejecta_solar = progenitor.final_mass_solar - remnant_solar
        nickel_solar = base_nickel * energy_foe**0.65 * float(metallicity_factor)

    if overrides.remnant_mass_solar is not None:
        remnant_solar = overrides.remnant_mass_solar
        ejecta_solar = progenitor.final_mass_solar - remnant_solar
    if ejecta_solar <= 0.01:
        raise ValueError("derived ejecta mass is non-positive; check final/remnant mass")
    if overrides.energy_foe is not None:
        energy_foe = overrides.energy_foe
    if overrides.nickel_mass_solar is not None:
        nickel_solar = overrides.nickel_mass_solar
    if nickel_solar >= ejecta_solar:
        raise ValueError("Ni-56 mass must be smaller than ejecta mass")

    opacity = overrides.opacity_m2_kg or base_opacity
    gamma_opacity = overrides.gamma_opacity_m2_kg or 0.0027
    energy_j = energy_foe * FOE_J
    ejecta_kg = ejecta_solar * SOLAR_MASS_KG
    velocity = math.sqrt(10.0 * energy_j / (3.0 * ejecta_kg))
    if velocity >= 0.3 * SPEED_OF_LIGHT_M_S:
        raise ValueError("derived non-relativistic ejecta velocity exceeds 0.3 c")

    beta = 13.8
    diffusion_s = math.sqrt(2.0 * opacity * ejecta_kg / (beta * SPEED_OF_LIGHT_M_S * velocity))
    diffusion_days = diffusion_s / DAY_S
    if overrides.diffusion_time_days is not None:
        diffusion_days = overrides.diffusion_time_days

    # Jeffery-like gamma-ray leakage scale for a homologous sphere.
    trapping_s = math.sqrt(9.0 * gamma_opacity * ejecta_kg**2 / (40.0 * math.pi * energy_j))
    return ExplosionParameters(
        energy_j=energy_j,
        ejecta_mass_kg=ejecta_kg,
        remnant_mass_kg=remnant_solar * SOLAR_MASS_KG,
        nickel56_mass_kg=nickel_solar * SOLAR_MASS_KG,
        characteristic_velocity_m_s=velocity,
        diffusion_time_days=float(diffusion_days),
        gamma_trapping_time_days=trapping_s / DAY_S,
        opacity_m2_kg=opacity,
        gamma_opacity_m2_kg=gamma_opacity,
    )

