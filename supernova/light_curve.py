"""Semi-analytic bolometric and educational multi-band light curves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .constants import (
    CO56_HEATING_W_KG,
    CO56_MEAN_LIFETIME_DAYS,
    DAY_S,
    M_BOL_SUN,
    NI56_HEATING_W_KG,
    NI56_MEAN_LIFETIME_DAYS,
    SOLAR_LUMINOSITY_W,
    SOLAR_MASS_KG,
    SOLAR_RADIUS_M,
    STEFAN_BOLTZMANN_W_M2_K4,
    YEAR_S,
)
from .explosion import ExplosionParameters, derive_explosion_parameters
from .photometry import BandPhotometry, FILTERS, band_photometry
from .progenitor import ModelOverrides, Progenitor
from .units import flux_from_luminosity


@dataclass(frozen=True)
class LightCurve:
    observer_time_days: np.ndarray
    rest_time_days: np.ndarray
    luminosity_w: np.ndarray
    radioactive_luminosity_w: np.ndarray
    shock_luminosity_w: np.ndarray
    interaction_luminosity_w: np.ndarray
    photospheric_radius_m: np.ndarray
    rest_temperature_k: np.ndarray
    observed_temperature_k: np.ndarray
    apparent_flux_w_m2: np.ndarray
    absolute_bolometric_magnitude: np.ndarray
    apparent_bolometric_magnitude: np.ndarray
    shock_breakout_luminosity_w: np.ndarray
    shock_breakout_temperature_k: np.ndarray
    dust_echo_luminosity_w: np.ndarray
    total_luminosity_w: np.ndarray
    total_apparent_flux_w_m2: np.ndarray
    bands: Mapping[str, BandPhotometry]
    apparent_magnitude: np.ndarray
    bolometric_magnitude: np.ndarray
    visual_flux_scale: np.ndarray
    angular_shell_radius: np.ndarray
    redshift: float

    @property
    def peak_index(self) -> int:
        return int(np.argmax(self.luminosity_w))

    @property
    def peak_time_days(self) -> float:
        return float(self.observer_time_days[self.peak_index])

    @property
    def peak_luminosity_w(self) -> float:
        return float(self.luminosity_w[self.peak_index])

    @property
    def visual_peak_index(self) -> int:
        return int(np.argmin(self.apparent_magnitude))

    @property
    def peak_apparent_magnitude(self) -> float:
        return float(self.apparent_magnitude[self.visual_peak_index])

    @property
    def peak_visual_time_days(self) -> float:
        return float(self.observer_time_days[self.visual_peak_index])


def radioactive_heating_w(
    time_days: np.ndarray | float,
    nickel56_mass_kg: float,
) -> np.ndarray:
    """Ni-56 -> Co-56 -> Fe-56 instantaneous decay-chain heating."""
    time = np.maximum(np.asarray(time_days, dtype=np.float64), 0.0)
    ni = np.exp(-time / NI56_MEAN_LIFETIME_DAYS)
    co = np.exp(-time / CO56_MEAN_LIFETIME_DAYS) - ni
    return nickel56_mass_kg * (NI56_HEATING_W_KG * ni + CO56_HEATING_W_KG * co)


def gamma_deposition_fraction(time_days: np.ndarray, trapping_time_days: float) -> np.ndarray:
    time = np.maximum(np.asarray(time_days, dtype=np.float64), 1.0e-4)
    return -np.expm1(-np.square(trapping_time_days / time))


class SupernovaLightCurveModel:
    """Compute source-frame physics and observer-frame photometry."""

    def __init__(
        self,
        progenitor: Progenitor,
        overrides: ModelOverrides | None = None,
    ) -> None:
        self.progenitor = progenitor
        self.overrides = overrides or ModelOverrides()
        self.explosion = derive_explosion_parameters(progenitor, self.overrides)

    def evaluate(self, observer_time_days: np.ndarray | list[float]) -> LightCurve:
        observer_time = np.asarray(observer_time_days, dtype=np.float64)
        if observer_time.ndim != 1 or observer_time.size < 2:
            raise ValueError("time grid must be a one-dimensional array with at least two samples")
        if np.any(~np.isfinite(observer_time)) or np.any(observer_time < 0.0):
            raise ValueError("time values must be finite and non-negative")
        if np.any(np.diff(observer_time) <= 0.0):
            raise ValueError("time grid must be strictly increasing")

        redshift = self.progenitor.distance.effective_redshift
        rest_time = observer_time / (1.0 + redshift)
        radioactive = self._radioactive_component(rest_time)
        shock, interaction = self._envelope_components(rest_time)
        luminosity = np.maximum(
            (radioactive + shock + interaction) * self.overrides.luminosity_scale,
            1.0e20,
        )
        radius = self._photospheric_radius(rest_time)
        raw_temperature = np.power(
            luminosity / (4.0 * math.pi * STEFAN_BOLTZMANN_W_M2_K4 * radius**2),
            0.25,
        )
        floor = self.overrides.temperature_floor_k or self._temperature_floor()
        evolving_floor = floor * np.exp(-rest_time / 700.0) + 1800.0 * (1.0 - np.exp(-rest_time / 700.0))
        temperature = np.maximum(raw_temperature, evolving_floor)

        breakout = self._shock_breakout_component(rest_time)
        breakout_temperature = np.full_like(
            rest_time, self.overrides.shock_breakout_temperature_k
        )
        echo = self._light_echo_component(rest_time, luminosity)
        total_luminosity = luminosity + breakout + echo

        absolute_mag = M_BOL_SUN - 2.5 * np.log10(luminosity / SOLAR_LUMINOSITY_W)
        extinction_bol = 0.85 * self.progenitor.extinction_av_mag
        apparent_flux = flux_from_luminosity(luminosity, self.progenitor.distance)
        apparent_flux *= 10.0 ** (-0.4 * extinction_bol)
        apparent_mag = absolute_mag + self.progenitor.distance.distance_modulus + extinction_bol
        total_apparent_flux = flux_from_luminosity(
            total_luminosity, self.progenitor.distance
        ) * 10.0 ** (-0.4 * extinction_bol)

        bands: dict[str, BandPhotometry] = {}
        for name, band in FILTERS.items():
            # Breakout is exposed only in the UV product.  It is never silently
            # converted into a classical visual light curve.
            extra_luminosity = breakout if name == "UV" else None
            extra_temperature = breakout_temperature if name == "UV" else None
            bands[name] = band_photometry(
                luminosity + echo,
                temperature,
                self.progenitor.distance,
                self.progenitor.extinction_av_mag,
                band,
                redshift=redshift,
                extra_luminosity_w=extra_luminosity,
                extra_temperature_k=extra_temperature,
            )
        visual = bands["V"]
        visual_flux_scale = np.power(10.0, -0.4 * visual.apparent_magnitude)
        ejecta_radius = (
            self.progenitor.radius_solar * SOLAR_RADIUS_M
            + self.explosion.characteristic_velocity_m_s * rest_time * DAY_S
        )
        angular_shell_arcsec = (
            ejecta_radius / self.progenitor.distance.meters * 206_264.806247
        )
        return LightCurve(
            observer_time_days=observer_time,
            rest_time_days=rest_time,
            luminosity_w=luminosity,
            radioactive_luminosity_w=radioactive,
            shock_luminosity_w=shock,
            interaction_luminosity_w=interaction,
            photospheric_radius_m=radius,
            rest_temperature_k=temperature,
            observed_temperature_k=temperature / (1.0 + redshift),
            apparent_flux_w_m2=apparent_flux,
            absolute_bolometric_magnitude=absolute_mag,
            apparent_bolometric_magnitude=apparent_mag,
            shock_breakout_luminosity_w=breakout,
            shock_breakout_temperature_k=breakout_temperature,
            dust_echo_luminosity_w=echo,
            total_luminosity_w=total_luminosity,
            total_apparent_flux_w_m2=total_apparent_flux,
            bands=bands,
            apparent_magnitude=visual.apparent_magnitude,
            bolometric_magnitude=apparent_mag,
            visual_flux_scale=visual_flux_scale,
            angular_shell_radius=angular_shell_arcsec,
            redshift=redshift,
        )

    def _shock_breakout_component(self, time_days: np.ndarray) -> np.ndarray:
        if not self.overrides.shock_breakout_enabled:
            return np.zeros_like(time_days)
        duration_days = self.overrides.shock_breakout_duration_hours / 24.0
        if self.overrides.shock_breakout_luminosity_w is not None:
            peak = self.overrides.shock_breakout_luminosity_w
        else:
            radius_factor = np.clip(self.progenitor.radius_solar / 500.0, 0.02, 4.0)
            peak = (
                2.0e38
                * self.explosion.energy_foe**0.9
                * radius_factor**0.65
                * self.explosion.ejecta_mass_solar**-0.35
            )
        return peak * np.exp(-np.maximum(time_days, 0.0) / duration_days)

    def _light_echo_component(
        self, time_days: np.ndarray, luminosity_w: np.ndarray
    ) -> np.ndarray:
        fraction = self.overrides.light_echo_reflection_fraction
        delay = self.overrides.light_echo_delay_days
        if fraction <= 0.0 or delay <= 0.0:
            return np.zeros_like(time_days)
        width = self.overrides.light_echo_width_days
        offsets = np.linspace(-2.5 * width, 2.5 * width, 21)
        weights = np.exp(-0.5 * np.square(offsets / width))
        weights /= np.sum(weights)
        echoed = np.zeros_like(luminosity_w)
        for offset, weight in zip(offsets, weights):
            source_time = time_days - delay - offset
            echoed += weight * np.interp(
                source_time, time_days, luminosity_w, left=0.0, right=0.0
            )
        return fraction * echoed

    def _radioactive_component(self, time_days: np.ndarray) -> np.ndarray:
        heating = radioactive_heating_w(time_days, self.explosion.nickel56_mass_kg)
        deposition = gamma_deposition_fraction(time_days, self.explosion.gamma_trapping_time_days)
        diffusion_release = -np.expm1(-np.square(time_days / self.explosion.diffusion_time_days))
        return heating * deposition * diffusion_release

    def _envelope_components(self, time_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.progenitor
        e = self.explosion.energy_foe
        mass = self.explosion.ejecta_mass_solar
        radius = p.radius_solar
        rise = -np.expm1(-np.square(time_days / 2.5))
        shock_scale = 2.0e35 * (radius / 500.0) ** 0.8 * e**0.9 * mass**-0.55
        shock_time = max(0.25, 1.8 * (radius / 500.0) ** 0.45 * mass**0.2 * e**-0.25)
        breakout_rise = -np.expm1(-time_days / 0.12)
        shock = shock_scale * breakout_rise * np.exp(-time_days / shock_time) / np.sqrt(1.0 + time_days)
        interaction = np.zeros_like(time_days)

        if p.supernova_type == "II-P":
            plateau_l = (
                1.3e35
                * (self.explosion.opacity_m2_kg / 0.034) ** (-1.0 / 3.0)
                * mass**-0.5
                * e ** (5.0 / 6.0)
                * (radius / 500.0) ** (2.0 / 3.0)
                * self.overrides.plateau_scale
            )
            plateau_days = 100.0 * (self.explosion.opacity_m2_kg / 0.034) ** (1.0 / 6.0) * mass**0.5 * e**(-1.0 / 6.0) * (radius / 500.0) ** (1.0 / 6.0)
            gate = 1.0 / (1.0 + np.exp(np.clip((time_days - plateau_days) / 6.0, -60.0, 60.0)))
            shock += plateau_l * rise * gate
        elif p.supernova_type == "II-L":
            shock += 2.0e35 * e**0.8 * rise * np.exp(-time_days / 55.0)
        elif p.supernova_type == "IIn":
            wind_speed = 120_000.0
            mass_rate = p.mass_loss_rate_solar_per_year * SOLAR_MASS_KG / YEAR_S
            csm_power = 0.5 * self.overrides.csm_efficiency * (mass_rate / wind_speed) * self.explosion.characteristic_velocity_m_s**3
            csm_power = min(csm_power, 5.0e38)
            interaction = csm_power * (-np.expm1(-time_days / 4.0)) * np.power(1.0 + time_days / 80.0, -0.55)
        elif p.supernova_type in {"Ib", "Ic"}:
            stripped_scale = 0.25e35 if p.supernova_type == "Ib" else 0.18e35
            shock += stripped_scale * e * rise * np.exp(-time_days / 14.0)
        elif p.supernova_type == "Ia":
            # No extended H/He envelope; the reference curve is radioactive.
            shock *= 0.02
        return shock, interaction

    def _photospheric_radius(self, time_days: np.ndarray) -> np.ndarray:
        initial = self.progenitor.radius_solar * SOLAR_RADIUS_M
        recession = np.power(1.0 + time_days / 50.0, -0.22)
        velocity = self.explosion.characteristic_velocity_m_s * recession
        expanded = initial + velocity * time_days * DAY_S
        # At late epochs this is an effective emitting radius, not the outermost ejecta.
        return expanded / np.power(1.0 + np.maximum(time_days - 120.0, 0.0) / 160.0, 0.55)

    def _temperature_floor(self) -> float:
        if self.progenitor.supernova_type in {"II-P", "II-L", "IIn"}:
            return 5_300.0
        if self.progenitor.supernova_type == "Ia":
            return 5_800.0
        return 4_500.0
