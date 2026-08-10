from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from supernova.light_curve import SupernovaLightCurveModel, radioactive_heating_w
from supernova.progenitor import ChemicalComposition, ModelOverrides, Progenitor
from supernova.scenarios import load_scenario
from supernova.units import Distance


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("sn1987a")
        self.model = SupernovaLightCurveModel(
            self.scenario.progenitor, self.scenario.overrides
        )

    def test_sn1987a_calibration_is_in_physical_range(self):
        explosion = self.model.explosion
        self.assertTrue(1.0 < explosion.energy_foe < 2.0)
        self.assertTrue(14.0 < explosion.ejecta_mass_solar < 20.0)
        self.assertTrue(3000.0 < explosion.characteristic_velocity_m_s / 1000.0 < 5000.0)
        self.assertAlmostEqual(explosion.nickel56_mass_solar, 0.075, places=6)

    def test_radioactive_chain_declines_on_late_tail(self):
        heating = radioactive_heating_w(np.asarray([0.0, 100.0, 300.0]), 1.0e29)
        self.assertGreater(heating[0], heating[1])
        self.assertGreater(heating[1], heating[2])
        expected_co_ratio = np.exp(-200.0 / 111.3)
        self.assertAlmostEqual(heating[2] / heating[1], expected_co_ratio, delta=0.03)

    def test_light_curve_has_one_finite_outburst_and_tail(self):
        curve = self.model.evaluate(np.linspace(0.0, 600.0, 2401))
        peak = curve.peak_index
        self.assertGreater(peak, 0)
        self.assertLess(peak, len(curve.luminosity_w) - 1)
        self.assertTrue(np.all(np.isfinite(curve.luminosity_w)))
        self.assertGreater(curve.luminosity_w[peak], curve.luminosity_w[0])
        self.assertLess(curve.luminosity_w[-1], 0.05 * curve.peak_luminosity_w)
        above_half = np.flatnonzero(curve.luminosity_w >= 0.5 * curve.peak_luminosity_w)
        self.assertGreater(above_half.size, 10)
        self.assertLess(curve.observer_time_days[above_half[-1]], 250.0)

    def test_extinction_and_distance_change_observed_not_absolute_luminosity(self):
        near = self.scenario.progenitor
        far = replace(near, distance=Distance(102.8, "kpc"), extinction_av_mag=near.extinction_av_mag + 1.0)
        grid = np.linspace(0.0, 300.0, 601)
        curve_near = SupernovaLightCurveModel(near, self.scenario.overrides).evaluate(grid)
        curve_far = SupernovaLightCurveModel(far, self.scenario.overrides).evaluate(grid)
        np.testing.assert_allclose(curve_near.luminosity_w, curve_far.luminosity_w)
        ratio = curve_near.apparent_flux_w_m2 / curve_far.apparent_flux_w_m2
        expected = 4.0 * 10.0 ** (0.4 * 0.85)
        np.testing.assert_allclose(ratio, expected, rtol=1e-12)

    def test_cosmological_time_dilation(self):
        source = self.scenario.progenitor
        local = replace(source, distance=Distance(1.0, "Mpc", redshift=0.0))
        remote = replace(source, distance=Distance(1000.0, "Mpc", redshift=0.5))
        grid = np.linspace(0.0, 800.0, 3201)
        local_curve = SupernovaLightCurveModel(local, self.scenario.overrides).evaluate(grid)
        remote_curve = SupernovaLightCurveModel(remote, self.scenario.overrides).evaluate(grid)
        self.assertAlmostEqual(remote_curve.peak_time_days / local_curve.peak_time_days, 1.5, delta=0.02)

    def test_all_required_types_produce_positive_curves(self):
        base = self.scenario.progenitor
        for supernova_type in ("II-P", "II-L", "IIn", "Ib", "Ic", "Ia"):
            if supernova_type in {"Ib", "Ic"}:
                composition = ChemicalComposition({"H": 0.01, "He": 0.55, "metals": 0.44})
            elif supernova_type == "Ia":
                composition = ChemicalComposition({"C": 0.49, "O": 0.49, "metals": 0.02})
            else:
                composition = base.composition
            final_mass = 1.38 if supernova_type == "Ia" else base.final_mass_solar
            initial_mass = 5.0 if supernova_type == "Ia" else base.initial_mass_solar
            star = replace(
                base,
                initial_mass_solar=initial_mass,
                final_mass_solar=final_mass,
                radius_solar=0.008 if supernova_type == "Ia" else base.radius_solar,
                composition=composition,
                supernova_type=supernova_type,
            )
            curve = SupernovaLightCurveModel(star, ModelOverrides()).evaluate(np.linspace(0.0, 300.0, 301))
            self.assertTrue(np.all(curve.luminosity_w > 0.0), supernova_type)

    def test_invalid_composition_and_ejecta_are_rejected(self):
        with self.assertRaises(ValueError):
            ChemicalComposition({"H": 0.2, "He": 0.2})
        with self.assertRaises(ValueError):
            replace(self.scenario.progenitor, final_mass_solar=30.0)


if __name__ == "__main__":
    unittest.main()

