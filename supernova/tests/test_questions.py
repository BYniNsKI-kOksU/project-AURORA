from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core.aurora_console import console
from supernova import run_supernova


class QuestionWizardTests(unittest.TestCase):
    def test_background_discovery_supports_older_aurora_core(self):
        from core import aurora_resolution

        profile = aurora_resolution.get_resolution(16)
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            (maps_dir / "aurora_sky_map_hammer_16k.png").touch()
            with patch.object(
                aurora_resolution,
                "available_sky_map_backgrounds",
                new=None,
            ):
                available = run_supernova._available_sky_backgrounds(
                    aurora_resolution,
                    profile,
                    maps_dir,
                )
        self.assertEqual(available, ("plain",))

    def test_choice_accepts_number_and_case_insensitive_value(self):
        choices = (("UV", "ultraviolet"), ("V", "visual"))
        with patch.object(console, "prompt", return_value="2"):
            self.assertEqual(run_supernova.prompt_choice("Filter", choices, "V"), "V")
        with patch.object(console, "prompt", return_value="uv"):
            self.assertEqual(run_supernova.prompt_choice("Filter", choices, "V"), "UV")

    def test_numeric_question_rejects_out_of_range_value(self):
        with patch.object(console, "prompt", side_effect=("99", "10")):
            value = run_supernova.prompt_number(
                "CRF", 10, integer=True, minimum=0, maximum=51
            )
        self.assertEqual(value, 10)

    def test_simulation_wizard_asks_only_mode_scenario_and_filter(self):
        responses = (
            "3",       # simulate
            "",        # default scenario
            "b",       # B filter
        )
        with patch.multiple(
            run_supernova,
            RUN_MODE="simulate_and_render",
            SCENARIO="sn1987a",
            OBSERVATION_FILTER="V",
            SIMULATED_DAYS=500.0,
            TIME_SAMPLES=1201,
            OUTPUT_DIR=Path("supernova/output"),
        ), patch.object(console, "prompt", side_effect=responses):
            self.assertTrue(run_supernova.configure_from_questions())
            self.assertEqual(run_supernova.RUN_MODE, "simulate")
            self.assertEqual(run_supernova.SCENARIO, "sn1987a")
            self.assertEqual(run_supernova.OBSERVATION_FILTER, "B")
            self.assertEqual(run_supernova.SIMULATED_DAYS, 500.0)
            self.assertEqual(run_supernova.TIME_SAMPLES, 1201)

    def test_combined_render_wizard_keeps_project_defaults_automatic(self):
        from core import aurora_resolution

        responses = (
            "",   # render video
            "4",  # all scenarios
            "2",  # all events in one video
            "",   # V filter
        )
        with patch.multiple(
            run_supernova,
            RUN_MODE="render",
            RENDER_SCENARIO_GROUP="single",
            RENDER_GROUP_LAYOUT="separate",
            OBSERVATION_FILTER="V",
            AURORA_RESOLUTION="16k",
            BACKGROUND_MODE="all_sky",
            SKY_MAP_BACKGROUND="plain",
        ), patch.object(
            console, "prompt", side_effect=responses
        ) as prompt, patch.object(
            aurora_resolution,
            "prompt_resolution",
            return_value=aurora_resolution.get_resolution(16),
        ), patch.object(
            aurora_resolution,
            "available_sky_map_backgrounds",
            return_value=("plain",),
        ), patch.object(
            aurora_resolution,
            "prompt_sky_map_background",
            return_value="plain",
        ):
            self.assertTrue(run_supernova.configure_from_questions())
            self.assertEqual(run_supernova.RENDER_SCENARIO_GROUP, "all")
            self.assertEqual(run_supernova.RENDER_GROUP_LAYOUT, "combined")
            self.assertEqual(run_supernova.BACKGROUND_MODE, "all_sky")
            self.assertEqual(prompt.call_count, 4)


if __name__ == "__main__":
    unittest.main()
