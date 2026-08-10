import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.aurora_console import console
from core.aurora_resolution import (
    available_sky_map_backgrounds,
    get_resolution,
    prompt_sky_map_background,
)


class SkyMapBackgroundSelectionTests(unittest.TestCase):
    def test_discovers_only_existing_background_files(self):
        resolution = get_resolution(16)
        with tempfile.TemporaryDirectory() as directory:
            maps_dir = Path(directory)
            (maps_dir / resolution.hammer_map_name).touch()
            selected_name = resolution.hammer_background_map_name(
                "constellations_coordinates"
            )
            (maps_dir / selected_name).touch()
            (maps_dir / "unrelated.png").touch()

            self.assertEqual(
                available_sky_map_backgrounds(resolution, maps_dir),
                ("plain", "constellations_coordinates"),
            )

    def test_option_five_selects_constellations_and_coordinates(self):
        printed_lines = []
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys.stdin, "isatty", return_value=True),
            patch.object(console, "prompt", return_value="5"),
            patch.object(
                console,
                "print",
                side_effect=lambda message: printed_lines.append(str(message)),
            ),
        ):
            selected = prompt_sky_map_background(
                available_backgrounds=(
                    "plain",
                    "constellations_coordinates",
                )
            )

        self.assertEqual(selected, "constellations_coordinates")
        menu_text = "\n".join(printed_lines)
        self.assertIn("1 = plain", menu_text)
        self.assertIn("5 = constellations + coordinates", menu_text)
        self.assertNotIn("2 = constellations", menu_text)


if __name__ == "__main__":
    unittest.main()
