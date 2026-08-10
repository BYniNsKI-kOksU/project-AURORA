from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from supernova import run_supernova
from supernova.scenarios import available_scenarios, load_scenario, run_monte_carlo


class ScenarioTests(unittest.TestCase):
    def test_required_scenarios_exist_and_future_cases_are_labeled(self):
        names = available_scenarios()
        self.assertIn("sn1987a", names)
        for name in ("betelgeuse", "eta_carinae"):
            scenario = load_scenario(name)
            self.assertTrue(scenario.is_hypothetical)
            self.assertIn("does not predict", scenario.disclaimer)
            self.assertTrue(scenario.uncertainty)

    def test_monte_carlo_is_reproducible_and_ordered(self):
        scenario = load_scenario("betelgeuse")
        time = np.linspace(0.0, 300.0, 41)
        first = run_monte_carlo(scenario, time, samples=12, seed=7)
        second = run_monte_carlo(scenario, time, samples=12, seed=7)
        np.testing.assert_allclose(first.luminosity_percentiles_w, second.luminosity_percentiles_w)
        self.assertTrue(np.all(first.luminosity_percentiles_w[0] <= first.luminosity_percentiles_w[1]))
        self.assertTrue(np.all(first.luminosity_percentiles_w[1] <= first.luminosity_percentiles_w[2]))

    def test_historical_and_hypothetical_catalogues_are_separate(self):
        historical = set(available_scenarios("historical"))
        hypothetical = set(available_scenarios("hypothetical"))
        self.assertFalse(historical & hypothetical)
        self.assertTrue({"sn1006", "sn1054", "sn1181", "sn1572", "sn1604", "sn1885a", "sn1987a"} <= historical)
        self.assertTrue({"betelgeuse", "eta_carinae", "antares", "r136a1"} <= hypothetical)
        self.assertTrue(all(load_scenario(name).is_historical for name in historical))
        self.assertTrue(all(load_scenario(name).is_hypothetical for name in hypothetical))

    def test_render_group_selects_every_historical_scenario(self):
        primary = load_scenario("sn1987a")
        with patch.object(run_supernova, "RENDER_SCENARIO_GROUP", "historical"):
            selected = run_supernova.render_scenarios_for_group(primary)
        self.assertEqual(
            tuple(scenario.key for scenario in selected),
            available_scenarios("historical"),
        )
        self.assertTrue(all(scenario.is_historical for scenario in selected))

    def test_render_group_selects_every_hypothetical_scenario(self):
        primary = load_scenario("sn1987a")
        with patch.object(run_supernova, "RENDER_SCENARIO_GROUP", "hypothetical"):
            selected = run_supernova.render_scenarios_for_group(primary)
        self.assertEqual(
            tuple(scenario.key for scenario in selected),
            available_scenarios("hypothetical"),
        )
        self.assertTrue(all(scenario.is_hypothetical for scenario in selected))

    def test_render_all_includes_every_scenario_once(self):
        primary = load_scenario("sn1987a")
        with patch.object(run_supernova, "RENDER_SCENARIO_GROUP", "all"):
            selected = run_supernova.render_scenarios_for_group(primary)
        keys = tuple(scenario.key for scenario in selected)
        self.assertEqual(keys, available_scenarios("all"))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn("type_ia_reference", keys)

    def test_batch_video_paths_cannot_overwrite_each_other(self):
        configured = Path("/tmp/supernova_collection.mp4")
        with patch.object(run_supernova, "VIDEO_OUTPUT_PATH", configured):
            first = run_supernova.render_video_output_path(
                "sn1987a", "supernova", batch=True
            )
            second = run_supernova.render_video_output_path(
                "betelgeuse", "supernova", batch=True
            )
        self.assertEqual(first.name, "supernova_collection_sn1987a.mp4")
        self.assertEqual(second.name, "supernova_collection_betelgeuse.mp4")
        self.assertNotEqual(first, second)

    def test_combined_group_dispatches_one_animation_and_keeps_separate_mode(self):
        primary = load_scenario("sn1987a")
        combined_result = (Path("combined.mp4"), Path("combined.png"))
        with patch.multiple(
            run_supernova,
            RENDER_SCENARIO_GROUP="historical",
            RENDER_GROUP_LAYOUT="combined",
        ), patch.object(
            run_supernova, "_run_combined_render", return_value=combined_result
        ) as combined_render:
            outputs = run_supernova._run_render_group(primary)
        self.assertEqual(outputs, (combined_result,))
        combined_render.assert_called_once()
        self.assertEqual(len(combined_render.call_args.args[0]), 7)

        separate_result = (Path("separate.mp4"), Path("separate.png"))
        with patch.multiple(
            run_supernova,
            RENDER_SCENARIO_GROUP="historical",
            RENDER_GROUP_LAYOUT="separate",
        ), patch.object(
            run_supernova, "_run_render", return_value=separate_result
        ) as separate_render:
            outputs = run_supernova._run_render_group(primary)
        self.assertEqual(len(outputs), 7)
        self.assertEqual(separate_render.call_count, 7)

    def test_hypothetical_scenario_is_never_observed(self):
        for name in available_scenarios("hypothetical"):
            scenario = load_scenario(name)
            self.assertEqual(scenario.status, "hypothetical")
            self.assertFalse(scenario.is_historical)
            self.assertIn("does not predict", scenario.disclaimer)
            self.assertEqual(set(scenario.variants), {"pessimistic", "nominal", "optimistic"})

    def test_monte_carlo_interval_contains_nominal_model(self):
        scenario = load_scenario("betelgeuse")
        envelope = run_monte_carlo(
            scenario, np.linspace(0.0, 300.0, 61), samples=80, seed=42
        )
        nominal = envelope.nominal_filter_magnitudes["V"]
        interval = envelope.filter_magnitude_percentiles["V"]
        self.assertTrue(np.all(nominal >= interval[0]))
        self.assertTrue(np.all(nominal <= interval[-1]))


if __name__ == "__main__":
    unittest.main()
