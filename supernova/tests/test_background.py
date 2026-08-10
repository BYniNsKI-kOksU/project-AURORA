from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from supernova.background import (
    BackgroundConfig,
    background_native_dimensions,
    image_dimensions,
    resolve_background,
)


def _png_header(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


class BackgroundTests(unittest.TestCase):
    def test_png_dimensions_preserve_native_16k_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "map.png"
            image.write_bytes(_png_header(16384, 8192))
            self.assertEqual(image_dimensions(image), (16384, 8192))
            config = BackgroundConfig(mode="custom", image_path=image)
            self.assertEqual(background_native_dimensions(config), (16384, 8192))

    def test_region_png_and_layout_are_paired_and_keep_native_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "aurora_test_region_16k.png"
            layout = root / "aurora_test_region_16k_layout.npz"
            image.write_bytes(_png_header(16384, 8192))
            np.savez(
                layout,
                projection=np.asarray("rectangular"),
                dimensions=np.asarray([16384, 8192]),
                l_center_deg=np.asarray(279.0),
                b_center_deg=np.asarray(-32.0),
                l_width_deg=np.asarray(12.0),
                b_height_deg=np.asarray(6.0),
                map_filename=np.asarray(image.name),
            )
            config = BackgroundConfig(
                mode="region",
                image_path=image,
                layout_path=layout,
                region_name="aurora_test_region_16k",
            )
            resolved = resolve_background(config)
            self.assertEqual(resolved.image_path.resolve(), image.resolve())
            self.assertEqual(resolved.layout_path.resolve(), layout.resolve())
            self.assertEqual(background_native_dimensions(resolved), (16384, 8192))

    def test_region_never_accepts_only_one_half_of_pair(self):
        with self.assertRaises(ValueError):
            resolve_background(
                BackgroundConfig(mode="region", image_path=Path("region.png"))
            )

    def test_full_sky_uses_the_selected_standard_aurora_background(self):
        resolved = resolve_background(
            BackgroundConfig(
                mode="all_sky",
                aurora_resolution="16k",
                sky_map_background="coordinates",
            )
        )
        self.assertEqual(
            resolved.image_path.name,
            "aurora_sky_map_hammer_16k_coordinates.png",
        )


if __name__ == "__main__":
    unittest.main()
