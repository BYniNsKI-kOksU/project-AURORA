from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.aurora_memory import MemoryController
from supernova.animation import (
    AnimationConfig,
    MEMORY,
    MultiScenarioAnimator,
    SupernovaAnimator,
    build_supernova_hevc_command,
)
from supernova.background import BackgroundConfig, BackgroundFrame
from supernova.light_curve import SupernovaLightCurveModel
from supernova import run_supernova
from supernova.scenarios import load_scenario


class AnimationTests(unittest.TestCase):
    def setUp(self):
        scenario = load_scenario("sn1987a")
        model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
        self.config = AnimationConfig(
            width=640,
            height=320,
            fps=10,
            duration_seconds=8.0,
            pre_roll_seconds=1.0,
            post_roll_seconds=1.0,
            simulated_days=500.0,
        )
        self.animator = SupernovaAnimator(model, self.config)

    def test_uses_shared_aurora_memory_controller(self):
        self.assertIsInstance(MEMORY, MemoryController)

    def test_memory_controller_throttles_every_encoded_frame(self):
        config = AnimationConfig(
            width=64,
            height=32,
            fps=2,
            duration_seconds=2.0,
            pre_roll_seconds=0.5,
            post_roll_seconds=0.5,
            simulated_days=300.0,
            show_labels=False,
        )
        animator = SupernovaAnimator(self.animator.model, config)
        background = BackgroundFrame(
            np.zeros((32, 64, 3), dtype=np.uint8), 32.0, 16.0
        )
        process = MagicMock()
        process.stdin = MagicMock()
        process.wait.return_value = 0

        def create_remux_output(command, **_kwargs):
            Path(command[-1]).write_bytes(b"mp4")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "test.mp4"
            with (
                patch("supernova.animation.shutil.which", return_value="ffmpeg"),
                patch("supernova.animation.load_background", return_value=background),
                patch.object(SupernovaAnimator, "_write_preview"),
                patch("supernova.animation.subprocess.Popen", return_value=process),
                patch("supernova.animation.subprocess.run", side_effect=create_remux_output),
                patch.object(MEMORY, "throttle", return_value=False) as throttle,
            ):
                animator.render_video(
                    BackgroundConfig(mode="catalog", catalog_path="unused.csv"),
                    output,
                )

        self.assertGreaterEqual(
            throttle.call_count,
            config.video_config.frame_count + 3,
        )

    def test_pre_and_post_roll_are_background_only(self):
        first = self.animator.frame_state(0)
        last = self.animator.frame_state(self.config.video_config.frame_count - 1)
        self.assertEqual(first.core_alpha, 0.0)
        self.assertEqual(last.core_alpha, 0.0)

    def test_timeline_contains_visible_time_before_and_after_explosion(self):
        self.assertLess(float(np.min(self.animator.frame_times)), 0.0)
        self.assertGreater(float(np.max(self.animator.frame_times)), 0.0)
        self.assertTrue(np.all(np.diff(self.animator.frame_times) >= 0.0))
        active = self.animator.frame_opacity > 0.0
        pre_explosion = active & (self.animator.frame_times < 0.0)
        self.assertGreaterEqual(
            int(np.count_nonzero(pre_explosion)),
            int(0.10 * np.count_nonzero(active)),
        )
        state = self.animator.state_at(-2.0)
        self.assertEqual(state.point_source_intensity, 0.0)
        self.assertEqual(state.shell_alpha, 0.0)

    def test_visual_peak_is_at_movie_midpoint_for_non_plateau_event(self):
        scenario = load_scenario("type_ia_reference")
        model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
        animator = SupernovaAnimator(model, self.config)
        self.assertEqual(animator.timeline_alignment, "peak")
        peak_frame = int(
            np.argmin(np.abs(animator.frame_times - animator.curve.peak_visual_time_days))
        )
        self.assertLessEqual(
            abs(peak_frame - self.config.video_config.frame_count // 2), 1
        )

    def test_long_plateau_uses_its_bright_phase_as_midpoint_anchor(self):
        self.assertEqual(self.animator.timeline_alignment, "plateau")
        anchor_frame = int(
            np.argmin(np.abs(self.animator.frame_times - self.animator.timeline_anchor_days))
        )
        self.assertLessEqual(
            abs(anchor_frame - self.config.video_config.frame_count // 2), 1
        )
        self.assertLessEqual(
            self.animator.bright_phase_start_days,
            self.animator.timeline_anchor_days,
        )
        self.assertGreaterEqual(
            self.animator.bright_phase_end_days,
            self.animator.timeline_anchor_days,
        )

    def test_shell_expands_monotonically_and_is_bounded(self):
        states = [self.animator.state_at(day) for day in np.linspace(0.0, 500.0, 101)]
        radii = np.asarray([state.shell_radius_px for state in states])
        self.assertTrue(np.all(np.diff(radii) >= 0.0))
        self.assertLessEqual(radii.max(), 0.28 * min(self.config.width, self.config.height))

    def test_core_scale_and_alpha_are_bounded(self):
        states = [self.animator.state_at(day) for day in np.linspace(0.0, 500.0, 101)]
        self.assertTrue(all(0.0 <= state.core_alpha <= 1.0 for state in states))
        self.assertTrue(all(0.8 <= state.core_sigma_px <= 0.055 * 320 for state in states))

    def test_event_does_not_repeat(self):
        curve = self.animator.curve
        peak = curve.peak_index
        tail = curve.luminosity_w[peak:]
        # Small component crossovers are allowed, but no second peak may regain
        # even half of the main peak after 250 observer days.
        late = curve.luminosity_w[curve.observer_time_days >= 250.0]
        self.assertLess(float(np.max(late)), 0.5 * curve.peak_luminosity_w)
        self.assertGreater(tail[0], tail[-1])

    def test_default_launcher_generates_mp4(self):
        self.assertIn(run_supernova.RUN_MODE, {"render", "simulate_and_render", "all"})
        default_output = run_supernova.OUTPUT_DIR / "sn1987a_supernova.mp4"
        self.assertEqual(default_output.suffix, ".mp4")
        self.assertTrue(run_supernova.VIDEO_MATCH_BACKGROUND_SIZE)
        self.assertEqual((run_supernova.VIDEO_WIDTH, run_supernova.VIDEO_HEIGHT), (16384, 8192))

    def test_archival_hevc_quality_is_explicit(self):
        command = build_supernova_hevc_command(
            run_supernova.OUTPUT_DIR / "quality_test.mp4",
            self.config.width,
            self.config.height,
            self.config,
        )
        self.assertEqual(command[command.index("-crf") + 1], "10")
        self.assertEqual(command[command.index("-preset") + 1], "slow")
        self.assertEqual(command[command.index("-tune") + 1], "grain")
        self.assertIn("format=yuv420p10le", command)

    def test_lossless_hevc_mode_disables_crf(self):
        config = AnimationConfig(
            width=640,
            height=320,
            fps=10,
            duration_seconds=8.0,
            pre_roll_seconds=1.0,
            post_roll_seconds=1.0,
            lossless_video=True,
        )
        command = build_supernova_hevc_command(
            run_supernova.OUTPUT_DIR / "lossless_test.mp4", 640, 320, config
        )
        self.assertNotIn("-crf", command)
        params = command[command.index("-x265-params") + 1]
        self.assertIn("lossless=1", params)

    def test_shell_does_not_change_point_source_brightness(self):
        model = self.animator.model
        with_shell = SupernovaAnimator(model, self.config)
        without_shell = SupernovaAnimator(
            model,
            AnimationConfig(
                width=640,
                height=320,
                fps=10,
                duration_seconds=8.0,
                pre_roll_seconds=1.0,
                post_roll_seconds=1.0,
                simulated_days=500.0,
                show_shell=False,
            ),
        )
        for day in (10.0, 80.0, 300.0):
            self.assertEqual(
                with_shell.state_at(day).point_source_intensity,
                without_shell.state_at(day).point_source_intensity,
            )

    def test_render_has_no_supernova_induced_clipping(self):
        config = AnimationConfig(
            width=640,
            height=320,
            fps=10,
            duration_seconds=8.0,
            pre_roll_seconds=1.0,
            post_roll_seconds=1.0,
            simulated_days=300.0,
            show_labels=False,
        )
        animator = SupernovaAnimator(self.animator.model, config)
        peak_frame = int(np.argmin(np.abs(animator.frame_times - animator.curve.peak_visual_time_days)))
        background = BackgroundFrame(np.zeros((320, 640, 3), dtype=np.uint8), 320.0, 160.0)
        rendered = animator.render_frame(background, peak_frame)
        self.assertEqual(int(np.count_nonzero(rendered == 255)), 0)

    def test_relative_point_brightness_is_resolution_independent(self):
        intensities = []
        for width, height in ((640, 320), (1920, 960)):
            config = AnimationConfig(
                width=width,
                height=height,
                fps=10,
                duration_seconds=8.0,
                pre_roll_seconds=1.0,
                post_roll_seconds=1.0,
                simulated_days=300.0,
                show_labels=False,
            )
            animator = SupernovaAnimator(self.animator.model, config)
            intensities.append(animator.state_at(animator.curve.peak_visual_time_days).point_source_intensity)
        self.assertAlmostEqual(intensities[0], intensities[1], places=12)

    def test_combined_animation_contains_multiple_synchronized_events(self):
        scenarios = (load_scenario("sn1987a"), load_scenario("betelgeuse"))
        config = AnimationConfig(
            width=640,
            height=320,
            fps=10,
            duration_seconds=8.0,
            pre_roll_seconds=1.0,
            post_roll_seconds=1.0,
            simulated_days=450.0,
            show_labels=False,
        )
        combined = MultiScenarioAnimator(scenarios, config)
        zero_frames = tuple(
            int(np.argmin(np.abs(animator.frame_times)))
            for animator in combined.animators
        )
        self.assertEqual(len(set(zero_frames)), 1)
        background = BackgroundFrame(
            np.zeros((320, 640, 3), dtype=np.uint8), 0.0, 0.0
        )
        positions = ((180.0, 160.0), (460.0, 160.0))
        frame = config.video_config.frame_count // 2
        rendered = combined.render_frame(background, positions, frame)
        self.assertGreater(int(rendered[150:171, 170:191].sum()), 0)
        self.assertGreater(int(rendered[150:171, 450:471].sum()), 0)

    def test_combined_animation_keeps_individual_apparent_fluxes(self):
        scenarios = (load_scenario("sn1987a"), load_scenario("betelgeuse"))
        combined = MultiScenarioAnimator(scenarios, self.config)
        frame = self.config.video_config.frame_count // 2
        intensities = tuple(
            animator.frame_state(frame).point_source_intensity
            for animator in combined.animators
        )
        self.assertNotEqual(intensities[0], intensities[1])


if __name__ == "__main__":
    unittest.main()
