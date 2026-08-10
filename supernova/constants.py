"""Physical constants used by the semi-analytic supernova model."""

from __future__ import annotations


SOLAR_MASS_KG = 1.98847e30
SOLAR_RADIUS_M = 6.957e8
SOLAR_LUMINOSITY_W = 3.828e26
PARSEC_M = 3.085677581491367e16
LIGHT_YEAR_M = 9.4607304725808e15
DAY_S = 86_400.0
YEAR_S = 365.25 * DAY_S
SPEED_OF_LIGHT_M_S = 299_792_458.0
PLANCK_CONSTANT_J_S = 6.62607015e-34
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
M_BOL_SUN = 4.74
FOE_J = 1.0e44  # 10^51 erg

# Mean lifetimes (half-life / ln(2)) and deposited heating rates.
NI56_MEAN_LIFETIME_DAYS = 8.8
CO56_MEAN_LIFETIME_DAYS = 111.3
NI56_HEATING_W_KG = 3.90e6
CO56_HEATING_W_KG = 6.78e5

# Cosmology used only when the supplied luminosity distance is cosmological.
HUBBLE_KM_S_MPC = 67.66
OMEGA_M = 0.3111
OMEGA_LAMBDA = 1.0 - OMEGA_M
