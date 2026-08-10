from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from supernova.light_curve import SupernovaLightCurveModel
from supernova.scenarios import load_scenario
from supernova.units import Distance


class PhotometryTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario("sn1987a")
        self.grid = np.linspace(0.0, 300.0, 3001)

    def test_sn1987a_visual_peak_regression(self):
        curve = SupernovaLightCurveModel(
            self.scenario.progenitor, self.scenario.overrides
        ).evaluate(self.grid)
        self.assertAlmostEqual(curve.peak_apparent_magnitude, 2.9, delta=0.35)

    def test_sn1987a_distance_is_about_50_kpc(self):
        distance_kpc = self.scenario.progenitor.distance.parsecs / 1000.0
        self.assertAlmostEqual(distance_kpc, 51.4, delta=2.0)

    def test_distance_modulus_changes_visual_magnitude(self):
        near = self.scenario.progenitor
        far = replace(near, distance=Distance(2.0 * near.distance.value, near.distance.unit))
        near_curve = SupernovaLightCurveModel(near, self.scenario.overrides).evaluate(self.grid)
        far_curve = SupernovaLightCurveModel(far, self.scenario.overrides).evaluate(self.grid)
        difference = far_curve.bands["V"].apparent_magnitude - near_curve.bands["V"].apparent_magnitude
        np.testing.assert_allclose(difference, 5.0 * np.log10(2.0), rtol=0.0, atol=1e-10)

    def test_extinction_weakens_every_band(self):
        clear = replace(self.scenario.progenitor, extinction_av_mag=0.0)
        dusty = replace(self.scenario.progenitor, extinction_av_mag=1.0)
        clear_curve = SupernovaLightCurveModel(clear, self.scenario.overrides).evaluate(self.grid)
        dusty_curve = SupernovaLightCurveModel(dusty, self.scenario.overrides).evaluate(self.grid)
        for name in clear_curve.bands:
            self.assertTrue(
                np.all(dusty_curve.bands[name].observed_flux_w_m2 < clear_curve.bands[name].observed_flux_w_m2),
                name,
            )

    def test_bolometric_and_visual_magnitudes_are_distinct(self):
        curve = SupernovaLightCurveModel(
            self.scenario.progenitor, self.scenario.overrides
        ).evaluate(self.grid)
        self.assertFalse(np.allclose(curve.bolometric_magnitude, curve.apparent_magnitude))
        self.assertFalse(np.allclose(curve.bands["UV"].apparent_magnitude, curve.bands["IR"].apparent_magnitude))

    def test_breakout_is_separate_and_uv_weighted(self):
        curve = SupernovaLightCurveModel(
            self.scenario.progenitor, self.scenario.overrides
        ).evaluate(np.linspace(0.0, 2.0, 201))
        self.assertGreater(curve.shock_breakout_luminosity_w[0], curve.luminosity_w[0])
        self.assertGreater(curve.bands["UV"].observed_flux_w_m2[0], curve.bands["V"].observed_flux_w_m2[0])
        self.assertEqual(curve.luminosity_w[0], 1.0e20)


if __name__ == "__main__":
    unittest.main()
