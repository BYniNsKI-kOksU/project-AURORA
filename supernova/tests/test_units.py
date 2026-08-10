from __future__ import annotations

import unittest

import numpy as np

from supernova.constants import LIGHT_YEAR_M, PARSEC_M
from supernova.units import Distance, flux_from_luminosity, luminosity_distance_m


class DistanceTests(unittest.TestCase):
    def test_supported_astronomical_units(self):
        self.assertAlmostEqual(Distance(1.0, "pc").meters, PARSEC_M)
        self.assertAlmostEqual(Distance(1.0, "ly").meters, LIGHT_YEAR_M)
        self.assertAlmostEqual(Distance(1.0, "Mpc").parsecs, 1.0e6)
        self.assertAlmostEqual(Distance(PARSEC_M, "m").parsecs, 1.0)

    def test_inverse_square_law(self):
        luminosity = np.asarray([1.0e36])
        near = flux_from_luminosity(luminosity, Distance(10.0, "pc"))[0]
        far = flux_from_luminosity(luminosity, Distance(20.0, "pc"))[0]
        self.assertAlmostEqual(near / far, 4.0, places=12)

    def test_distance_modulus(self):
        self.assertAlmostEqual(Distance(10.0, "pc").distance_modulus, 0.0)
        self.assertAlmostEqual(Distance(100.0, "pc").distance_modulus, 5.0)

    def test_cosmological_distance_and_explicit_redshift(self):
        distance = Distance(1000.0, "Mpc")
        self.assertGreater(distance.effective_redshift, 0.1)
        recovered = luminosity_distance_m(distance.effective_redshift)
        self.assertAlmostEqual(recovered / distance.meters, 1.0, places=8)
        self.assertEqual(Distance(1.0, "kpc", redshift=0.02).effective_redshift, 0.02)

    def test_invalid_distance(self):
        with self.assertRaises(ValueError):
            Distance(0.0, "pc")
        with self.assertRaises(ValueError):
            Distance(1.0, "furlong")


if __name__ == "__main__":
    unittest.main()

