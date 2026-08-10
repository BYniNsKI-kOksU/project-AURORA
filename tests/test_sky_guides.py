import unittest
from unittest.mock import patch

from core.aurora_sky_guides import prompt_reference_overlays


class ReferenceOverlayPromptTests(unittest.TestCase):
    def test_resolves_constellations_coordinates_and_poland_limits(self):
        with (
            patch(
                "core.aurora_sky_guides.prompt_constellation_overlay",
                return_value=True,
            ) as constellation_prompt,
            patch(
                "core.aurora_sky_guides.prompt_coordinate_grid_overlay",
                return_value=False,
            ) as coordinate_prompt,
            patch(
                "core.aurora_sky_guides.prompt_poland_limits_overlay",
                return_value=True,
            ) as poland_prompt,
        ):
            selected = prompt_reference_overlays(
                constellations="yes",
                coordinate_grid="no",
                poland_limits="ask",
            )

        self.assertEqual(selected, (True, False, True))
        constellation_prompt.assert_called_once_with("yes")
        coordinate_prompt.assert_called_once_with("no")
        poland_prompt.assert_called_once_with("ask")


if __name__ == "__main__":
    unittest.main()
