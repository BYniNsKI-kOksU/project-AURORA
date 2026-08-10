"""Render one non-repeating explosion with an expanding display shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import numpy as np

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_paths import VIDEOS_DIR
from core.aurora_render_core import temperature_to_rgb
from core.aurora_video_core import (
    VideoRenderConfig,
    build_editorial_timeline,
    build_hevc_command,
    build_hvc1_remux_command,
)

from .background import (
    BackgroundConfig,
    BackgroundFrame,
    load_background,
    project_galactic_position,
)
from .light_curve import LightCurve, SupernovaLightCurveModel
from .scenarios import Scenario


# Shared adaptive RAM limiter, identical to variable_animation and
# microlensing_render. By default it keeps a safe reserve for the OS; batch
# runs may use the standard AURORA memory environment variables.
MEMORY = MemoryController.from_environment()


@dataclass(frozen=True)
class AnimationConfig:
    width: int = 16384
    height: int = 8192
    fps: int = 25
    duration_seconds: float = 24.0
    pre_roll_seconds: float = 0.5
    post_roll_seconds: float = 0.5
    edge_fade_seconds: float = 0.35
    simulated_days: float = 450.0
    pre_explosion_days: float = 10.0
    explosion_position_fraction: float = 0.18
    timeline_alignment: str = "auto"
    plateau_threshold_days: float = 45.0
    plateau_magnitude_window: float = 0.5
    filter_name: str = "V"
    show_shell: bool = True
    show_halo: bool = True
    show_labels: bool = True
    point_psf_sigma_px_960: float = 2.4
    halo_radius_px_960: float = 5.0
    illustrative_shell_max_radius_px_960: float = 10.0
    video_crf: int = 10
    encoder_preset: str = "slow"
    preserve_star_field_detail: bool = True
    lossless_video: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("animation dimensions must be positive even integers")
        if self.fps <= 0 or self.duration_seconds <= 0.0 or self.simulated_days <= 0.0:
            raise ValueError("FPS, duration and simulated days must be positive")
        if self.pre_explosion_days <= 0.0:
            raise ValueError("pre_explosion_days must be positive")
        if not 0.0 < self.explosion_position_fraction < 0.45:
            raise ValueError("explosion_position_fraction must be between 0 and 0.45")
        if self.timeline_alignment not in {"auto", "peak", "plateau"}:
            raise ValueError("timeline_alignment must be auto, peak, or plateau")
        if self.plateau_threshold_days <= 0.0 or self.plateau_magnitude_window <= 0.0:
            raise ValueError("plateau timeline thresholds must be positive")
        if self.filter_name not in {"UV", "U", "B", "V", "R", "I", "IR"}:
            raise ValueError("unsupported animation filter")
        if min(
            self.point_psf_sigma_px_960,
            self.halo_radius_px_960,
            self.illustrative_shell_max_radius_px_960,
        ) <= 0.0:
            raise ValueError("PSF, halo and shell display radii must be positive")
        if not 0 <= self.video_crf <= 51:
            raise ValueError("video_crf must be between 0 and 51")
        if self.encoder_preset not in {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        }:
            raise ValueError("unsupported x265 encoder preset")
        # Delegate the remaining timeline invariants to the shared AURORA type.
        self.video_config

    @property
    def video_config(self) -> VideoRenderConfig:
        return VideoRenderConfig(
            duration_seconds=self.duration_seconds,
            fps=self.fps,
            width=self.width,
            height=self.height,
            pre_roll_seconds=self.pre_roll_seconds,
            post_roll_seconds=self.post_roll_seconds,
            edge_fade_seconds=self.edge_fade_seconds,
            save_debug_frames=False,
            encoder_preset=self.encoder_preset,
        )


def build_supernova_hevc_command(
    output: Path,
    width: int,
    height: int,
    config: AnimationConfig,
) -> list[str]:
    """Build the AURORA HEVC command with an explicit archival-quality rate."""
    command = build_hevc_command(output, width, height, config.video_config)
    output_token = command.pop()
    params_index = command.index("-x265-params") + 1
    if config.lossless_video:
        command[params_index] += ":lossless=1"
    else:
        command.extend(["-crf", str(config.video_crf)])
    if config.preserve_star_field_detail and not config.lossless_video:
        command.extend(["-tune", "grain"])
    command.append(output_token)
    return command


@dataclass(frozen=True)
class VisualState:
    observer_time_days: float
    normalized_brightness: float
    core_sigma_px: float
    core_alpha: float
    shell_radius_px: float
    shell_width_px: float
    shell_alpha: float
    color: np.ndarray
    apparent_magnitude: float
    bolometric_magnitude: float
    visual_flux_scale: float
    angular_shell_radius: float
    halo_intensity: float
    halo_radius: float
    point_source_intensity: float
    intrinsic_apparent_magnitude: float
    echo_intensity: float
    echo_radius_px: float
    display_opacity: float


class SupernovaAnimator:
    """Adapt physical output to the established AURORA additive renderer."""

    def __init__(
        self,
        model: SupernovaLightCurveModel,
        config: AnimationConfig,
        *,
        scenario_status: str = "model",
    ) -> None:
        self.model = model
        self.config = config
        self.scenario_status = scenario_status
        frames = config.video_config.frame_count
        # A denser independent grid prevents animation FPS from changing physics.
        sample_count = max(1200, frames * 2)
        self.curve = model.evaluate(np.linspace(0.0, config.simulated_days, sample_count))
        self.bright_phase_start_days, self.bright_phase_end_days = self._bright_phase_bounds()
        bright_phase_width = self.bright_phase_end_days - self.bright_phase_start_days
        use_plateau = config.timeline_alignment == "plateau" or (
            config.timeline_alignment == "auto"
            and bright_phase_width >= config.plateau_threshold_days
        )
        if use_plateau:
            self.timeline_anchor_days = 0.5 * (
                self.bright_phase_start_days + self.bright_phase_end_days
            )
            self.timeline_alignment = "plateau"
        else:
            self.timeline_anchor_days = self.curve.peak_visual_time_days
            self.timeline_alignment = "peak"
        self.timeline_start_days = -config.pre_explosion_days
        midpoint_progress = (
            0.5 * config.video_config.actual_duration_seconds
            - config.pre_roll_seconds
        ) / config.video_config.active_duration_seconds
        if not 0.0 < midpoint_progress < 1.0:
            raise ValueError("video midpoint must lie inside the active timeline")
        if config.explosion_position_fraction >= midpoint_progress:
            raise ValueError("explosion must occur before the video midpoint")
        if not self.timeline_anchor_days < config.simulated_days:
            raise ValueError("simulated_days must extend beyond the peak or plateau anchor")
        self.timeline_end_days = config.simulated_days
        _, self.frame_opacity, self.active_progress = build_editorial_timeline(
            0.0, 1.0, config.video_config
        )
        # This is an editorial, piecewise-linear mapping of physical observer
        # time.  It reserves a visible pre-explosion segment, fixes the event
        # anchor at the movie midpoint and still retains the late tail.
        self.frame_times = np.interp(
            self.active_progress,
            np.asarray(
                [0.0, config.explosion_position_fraction, midpoint_progress, 1.0]
            ),
            np.asarray(
                [
                    self.timeline_start_days,
                    0.0,
                    self.timeline_anchor_days,
                    self.timeline_end_days,
                ]
            ),
        ).astype(np.float32)
        self._peak_luminosity = self.curve.peak_luminosity_w

    def _bright_phase_bounds(self) -> tuple[float, float]:
        magnitudes = self.curve.bands[self.config.filter_name].apparent_magnitude
        peak_index = int(np.argmin(magnitudes))
        bright = magnitudes <= (
            magnitudes[peak_index] + self.config.plateau_magnitude_window
        )
        first = peak_index
        while first > 0 and bool(bright[first - 1]):
            first -= 1
        last = peak_index
        while last + 1 < bright.size and bool(bright[last + 1]):
            last += 1
        times = self.curve.observer_time_days
        return float(times[first]), float(times[last])

    def state_at(self, observer_time_days: float, opacity: float = 1.0) -> VisualState:
        time = float(
            np.clip(
                observer_time_days,
                self.timeline_start_days,
                self.timeline_end_days,
            )
        )
        if time < 0.0:
            return VisualState(
                observer_time_days=time,
                normalized_brightness=0.0,
                core_sigma_px=max(
                    0.8,
                    self.config.point_psf_sigma_px_960 * self.config.height / 960.0,
                ),
                core_alpha=0.0,
                shell_radius_px=0.0,
                shell_width_px=0.0,
                shell_alpha=0.0,
                color=np.ones(3, dtype=np.float32),
                apparent_magnitude=float("inf"),
                bolometric_magnitude=float("inf"),
                visual_flux_scale=0.0,
                angular_shell_radius=0.0,
                halo_intensity=0.0,
                halo_radius=0.0,
                point_source_intensity=0.0,
                intrinsic_apparent_magnitude=float("inf"),
                echo_intensity=0.0,
                echo_radius_px=0.0,
                display_opacity=float(np.clip(opacity, 0.0, 1.0)),
            )
        luminosity = float(np.interp(time, self.curve.observer_time_days, self.curve.luminosity_w))
        temperature = float(np.interp(time, self.curve.observer_time_days, self.curve.observed_temperature_k))
        relative = float(np.clip(luminosity / self._peak_luminosity, 0.0, 1.0))
        display = relative
        resolution_scale = self.config.height / 960.0
        band = self.curve.bands[self.config.filter_name]
        apparent_magnitude = float(
            np.interp(time, self.curve.observer_time_days, band.apparent_magnitude)
        )
        intrinsic_magnitude = float(
            np.interp(time, self.curve.observer_time_days, band.intrinsic_apparent_magnitude)
        )
        bolometric_magnitude = float(
            np.interp(time, self.curve.observer_time_days, self.curve.bolometric_magnitude)
        )
        visual_flux_scale = float(10.0 ** (-0.4 * apparent_magnitude))

        # Absolute observed flux controls intensity.  Source size is a fixed PSF
        # and never grows with luminosity or with the physical ejecta radius.
        point_intensity = float(
            np.clip(opacity * -np.expm1(-6.0 * visual_flux_scale), 0.0, 0.92)
        )
        core_sigma = max(0.8, self.config.point_psf_sigma_px_960 * resolution_scale)
        core_alpha = point_intensity
        halo_intensity = (
            float(np.clip(opacity * 0.08 * -np.expm1(-2.0 * visual_flux_scale), 0.0, 0.08))
            if self.config.show_halo
            else 0.0
        )
        halo_radius = self.config.halo_radius_px_960 * resolution_scale

        rest_days = time / (1.0 + self.curve.redshift)
        expansion_progress = np.sqrt(max(rest_days, 0.0) / max(self.config.simulated_days, 1.0))
        shell_radius = float(
            self.config.illustrative_shell_max_radius_px_960
            * resolution_scale
            * expansion_progress
        )
        shell_width = max(0.55 * resolution_scale, 0.10 * shell_radius)
        shell_alpha = (
            float(np.clip(opacity * 0.055 * relative**0.45, 0.0, 0.055))
            if self.config.show_shell
            else 0.0
        )
        angular_shell_radius = float(
            np.interp(time, self.curve.observer_time_days, self.curve.angular_shell_radius)
        )
        echo_luminosity = float(
            np.interp(time, self.curve.observer_time_days, self.curve.dust_echo_luminosity_w)
        )
        echo_fraction = echo_luminosity / max(self._peak_luminosity, 1.0)
        echo_intensity = float(np.clip(opacity * 0.04 * np.sqrt(echo_fraction), 0.0, 0.025))
        echo_radius = 1.35 * self.config.illustrative_shell_max_radius_px_960 * resolution_scale
        color = np.asarray(temperature_to_rgb(np.asarray([temperature]))[0], dtype=np.float32)
        return VisualState(
            time,
            display,
            core_sigma,
            core_alpha,
            shell_radius,
            shell_width,
            shell_alpha,
            color,
            apparent_magnitude,
            bolometric_magnitude,
            visual_flux_scale,
            angular_shell_radius,
            halo_intensity,
            halo_radius,
            point_intensity,
            intrinsic_magnitude,
            echo_intensity,
            echo_radius,
            float(np.clip(opacity, 0.0, 1.0)),
        )

    def frame_state(self, frame: int) -> VisualState:
        if not 0 <= frame < self.config.video_config.frame_count:
            raise IndexError("frame outside animation")
        return self.state_at(float(self.frame_times[frame]), float(self.frame_opacity[frame]))

    def render_frame(
        self,
        background: BackgroundFrame,
        frame: int,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        if canvas is None:
            canvas = background.pixels.copy()
        else:
            if canvas.shape != background.pixels.shape or canvas.dtype != np.uint8:
                raise ValueError("reusable canvas must match the uint8 background")
            np.copyto(canvas, background.pixels)
        state = self.frame_state(frame)
        self.draw_event(canvas, background.target_x, background.target_y, state)
        if self.config.show_labels and state.display_opacity > 0.0:
            self._draw_labels(canvas, state)
        return canvas

    def draw_event(
        self,
        canvas: np.ndarray,
        center_x: float,
        center_y: float,
        state: VisualState,
    ) -> None:
        """Add only this event's physical point and illustrative layers."""
        if state.core_alpha <= 0.0:
            return
        self._draw_halo(canvas, center_x, center_y, state)
        self._draw_shell(canvas, center_x, center_y, state)
        self._screen_gaussian(
            canvas,
            center_x,
            center_y,
            state.core_sigma_px,
            state.point_source_intensity,
            state.color,
        )

    @staticmethod
    def _screen_gaussian(
        canvas: np.ndarray,
        center_x: float,
        center_y: float,
        sigma: float,
        intensity: float,
        color: np.ndarray,
    ) -> None:
        if sigma <= 0.0 or intensity <= 0.0:
            return
        cut = int(4.0 * sigma) + 2
        x0, x1 = max(0, int(center_x) - cut), min(canvas.shape[1], int(center_x) + cut + 1)
        y0, y1 = max(0, int(center_y) - cut), min(canvas.shape[0], int(center_y) + cut + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        profile = np.exp(-0.5 * ((xx - center_x) ** 2 + (yy - center_y) ** 2) / sigma**2)
        emission = np.clip(profile[..., None] * intensity * color, 0.0, 0.999)
        background = canvas[y0:y1, x0:x1].astype(np.float32) / 255.0
        canvas[y0:y1, x0:x1] = np.rint(
            255.0 * (1.0 - (1.0 - background) * (1.0 - emission))
        ).astype(np.uint8)

    @classmethod
    def _draw_halo(
        cls, canvas: np.ndarray, center_x: float, center_y: float, state: VisualState
    ) -> None:
        cls._screen_gaussian(
            canvas,
            center_x,
            center_y,
            state.halo_radius,
            state.halo_intensity,
            state.color,
        )
        cls._screen_gaussian(
            canvas,
            center_x,
            center_y,
            state.echo_radius_px,
            state.echo_intensity,
            state.color,
        )

    @staticmethod
    def _draw_shell(canvas: np.ndarray, center_x: float, center_y: float, state: VisualState) -> None:
        if state.shell_radius_px < 1.0 or state.shell_alpha <= 0.0:
            return
        cut = int(state.shell_radius_px + 4.0 * state.shell_width_px) + 2
        x0, x1 = max(0, int(center_x) - cut), min(canvas.shape[1], int(center_x) + cut + 1)
        y0, y1 = max(0, int(center_y) - cut), min(canvas.shape[0], int(center_y) + cut + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        ring = np.exp(-0.5 * ((radius - state.shell_radius_px) / state.shell_width_px) ** 2)
        # A faint filled interior conveys optically thinning ejecta.
        interior = np.exp(-0.5 * (radius / max(state.shell_radius_px, 1.0)) ** 2) * 0.12
        alpha = np.clip((ring + interior) * state.shell_alpha, 0.0, 1.0)
        emission = np.clip(alpha[..., None] * state.color, 0.0, 0.25)
        patch = canvas[y0:y1, x0:x1].astype(np.float32) / 255.0
        canvas[y0:y1, x0:x1] = np.rint(
            255.0 * (1.0 - (1.0 - patch) * (1.0 - emission))
        ).astype(np.uint8)

    def _draw_labels(self, canvas: np.ndarray, state: VisualState) -> None:
        from .text_overlay import draw_text_block

        status = (
            "OBSERVED HISTORICALLY"
            if self.scenario_status == "historical_observation"
            else "HYPOTHETICAL - NOT A DATE PREDICTION"
            if self.scenario_status == "hypothetical"
            else self.scenario_status.upper()
        )
        if state.observer_time_days < 0.0:
            lines = (
                f"FILTER {self.config.filter_name}  PRE-EXPLOSION BASELINE",
                f"t={state.observer_time_days:.1f} D  TYPE {self.model.progenitor.supernova_type}",
                status,
                "EXPLOSION AT t=0 D",
            )
        else:
            lines = (
                f"FILTER {self.config.filter_name}  m={state.apparent_magnitude:+.2f} MAG",
                f"t=+{state.observer_time_days:.1f} D  TYPE {self.model.progenitor.supernova_type}",
                status,
                "POINT: OBSERVED FLUX  SHELL/HALO: ILLUSTRATIVE",
            )
        draw_text_block(canvas, lines)

    def render_video(
        self,
        background_config: BackgroundConfig,
        output: Path,
        *,
        preview: Path | None = None,
    ) -> tuple[Path, Path]:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg with libx265 is required to render MP4 video")
        MEMORY.throttle()
        background = load_background(
            background_config,
            self.config.width,
            self.config.height,
            target_longitude_deg=self.model.progenitor.galactic_longitude_deg,
            target_latitude_deg=self.model.progenitor.galactic_latitude_deg,
        )
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if preview is None:
            preview = output.with_name(output.stem + "_preview.png")
        peak_frame = int(np.argmin(np.abs(self.frame_times - self.curve.peak_visual_time_days)))
        MEMORY.throttle()
        self._write_preview(self.render_frame(background, peak_frame), preview)

        raw_output = output.with_name(output.stem + "_raw.mp4")
        process = subprocess.Popen(
            build_supernova_hevc_command(
                raw_output, self.config.width, self.config.height, self.config
            ),
            stdin=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("FFmpeg stdin pipe could not be created")
        MEMORY.throttle()
        frame_buffer = np.empty_like(background.pixels)
        try:
            for frame in console.progress(
                range(self.config.video_config.frame_count), desc="Supernova frames", unit="frame"
            ):
                MEMORY.throttle()
                rendered = self.render_frame(background, frame, canvas=frame_buffer)
                process.stdin.write(memoryview(rendered).cast("B"))
            process.stdin.close()
            return_code = process.wait()
        except Exception:
            process.stdin.close()
            process.terminate()
            process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(f"FFmpeg encoding failed with exit code {return_code}")
        subprocess.run(build_hvc1_remux_command(raw_output, output), check=True)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not create a valid MP4 output file")
        raw_output.unlink(missing_ok=True)
        return output, preview

    @staticmethod
    def _write_preview(frame: np.ndarray, output: Path) -> None:
        height, width = frame.shape[:2]
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-frames:v",
            "1",
            str(output),
        ]
        subprocess.run(command, input=memoryview(frame).cast("B"), check=True)


class MultiScenarioAnimator:
    """Render several independently calibrated events on one synchronized map."""

    def __init__(self, scenarios: tuple[Scenario, ...], config: AnimationConfig) -> None:
        if len(scenarios) < 2:
            raise ValueError("a combined animation requires at least two scenarios")
        self.scenarios = tuple(scenarios)
        self.config = config
        self.animators = tuple(
            SupernovaAnimator(
                SupernovaLightCurveModel(scenario.progenitor, scenario.overrides),
                config,
                scenario_status=scenario.status,
            )
            for scenario in self.scenarios
        )

    def _load_background_and_positions(
        self, background_config: BackgroundConfig
    ) -> tuple[BackgroundFrame, tuple[tuple[float, float], ...]]:
        first = self.scenarios[0].progenitor
        background = load_background(
            background_config,
            self.config.width,
            self.config.height,
            target_longitude_deg=first.galactic_longitude_deg,
            target_latitude_deg=first.galactic_latitude_deg,
        )
        positions: list[tuple[float, float]] = []
        outside: list[str] = []
        for scenario in self.scenarios:
            progenitor = scenario.progenitor
            x, y, visible = project_galactic_position(
                background_config,
                self.config.width,
                self.config.height,
                progenitor.galactic_longitude_deg,
                progenitor.galactic_latitude_deg,
            )
            if not visible:
                outside.append(scenario.key)
            positions.append((x, y))
        if outside:
            raise ValueError(
                "combined region does not contain: " + ", ".join(outside)
            )
        return background, tuple(positions)

    def render_frame(
        self,
        background: BackgroundFrame,
        positions: tuple[tuple[float, float], ...],
        frame: int,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(positions) != len(self.animators):
            raise ValueError("each combined scenario requires one map position")
        if canvas is None:
            canvas = background.pixels.copy()
        else:
            if canvas.shape != background.pixels.shape or canvas.dtype != np.uint8:
                raise ValueError("reusable canvas must match the uint8 background")
            np.copyto(canvas, background.pixels)
        states = tuple(animator.frame_state(frame) for animator in self.animators)
        for animator, (x, y), state in zip(self.animators, positions, states):
            animator.draw_event(canvas, x, y, state)
        if self.config.show_labels and any(state.display_opacity > 0.0 for state in states):
            self._draw_collection_labels(canvas, positions, states, frame)
        return canvas

    def _draw_collection_labels(
        self,
        canvas: np.ndarray,
        positions: tuple[tuple[float, float], ...],
        states: tuple[VisualState, ...],
        frame: int,
    ) -> None:
        from .text_overlay import draw_text_block

        first_animator = self.animators[0]
        progress = float(first_animator.active_progress[frame])
        midpoint = 0.5
        if progress < self.config.explosion_position_fraction:
            phase = "PRE-EXPLOSION"
        elif progress < midpoint:
            phase = "RISE"
        elif progress < 0.78:
            phase = "PEAK / PLATEAU"
        else:
            phase = "RADIOACTIVE TAIL"
        group_statuses = {scenario.status for scenario in self.scenarios}
        group = (
            "HISTORICAL"
            if group_statuses == {"historical_observation"}
            else "HYPOTHETICAL"
            if group_statuses == {"hypothetical"}
            else "ALL SCENARIOS"
        )
        draw_text_block(
            canvas,
            (
                f"COMBINED {group}  {len(self.scenarios)} EVENTS  FILTER {self.config.filter_name}",
                f"VIDEO t={frame / self.config.fps:.1f} S  PHASE {phase}",
                "EXPLOSIONS SYNCHRONIZED AT t=0",
                "EACH EVENT KEEPS ITS OWN OBSERVER-TIME MODEL AND APPARENT FLUX",
            ),
        )
        label_scale = max(1, int(round(canvas.shape[0] / 2048.0)))
        for scenario, (x, y), state in zip(self.scenarios, positions, states):
            label = scenario.key.replace("_", " ").upper()
            if state.observer_time_days >= 0.0 and np.isfinite(state.apparent_magnitude):
                label += f" {state.apparent_magnitude:+.1f} MAG"
            label_width = (len(label) * 6 + 8) * label_scale
            label_height = 16 * label_scale
            draw_x = int(round(x)) + 3 * label_scale
            if draw_x + label_width >= canvas.shape[1]:
                draw_x = int(round(x)) - label_width - 3 * label_scale
            draw_y = int(round(y)) - label_height - 3 * label_scale
            draw_x = int(np.clip(draw_x, 0, max(canvas.shape[1] - label_width, 0)))
            draw_y = int(np.clip(draw_y, 0, max(canvas.shape[0] - label_height, 0)))
            draw_text_block(
                canvas,
                (label,),
                x=draw_x,
                y=draw_y,
                scale=label_scale,
            )

    def render_video(
        self,
        background_config: BackgroundConfig,
        output: Path,
        *,
        preview: Path | None = None,
    ) -> tuple[Path, Path]:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg with libx265 is required to render MP4 video")
        MEMORY.throttle()
        background, positions = self._load_background_and_positions(background_config)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if preview is None:
            preview = output.with_name(output.stem + "_preview.png")
        midpoint_frame = self.config.video_config.frame_count // 2
        MEMORY.throttle()
        SupernovaAnimator._write_preview(
            self.render_frame(background, positions, midpoint_frame), preview
        )

        raw_output = output.with_name(output.stem + "_raw.mp4")
        process = subprocess.Popen(
            build_supernova_hevc_command(
                raw_output, self.config.width, self.config.height, self.config
            ),
            stdin=subprocess.PIPE,
        )
        if process.stdin is None:
            raise RuntimeError("FFmpeg stdin pipe could not be created")
        MEMORY.throttle()
        frame_buffer = np.empty_like(background.pixels)
        try:
            for frame in console.progress(
                range(self.config.video_config.frame_count),
                desc="Combined supernova frames",
                unit="frame",
            ):
                MEMORY.throttle()
                rendered = self.render_frame(
                    background, positions, frame, canvas=frame_buffer
                )
                process.stdin.write(memoryview(rendered).cast("B"))
            process.stdin.close()
            return_code = process.wait()
        except Exception:
            process.stdin.close()
            process.terminate()
            process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(f"FFmpeg encoding failed with exit code {return_code}")
        subprocess.run(build_hvc1_remux_command(raw_output, output), check=True)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not create a valid MP4 output file")
        raw_output.unlink(missing_ok=True)
        return output, preview


def default_video_path(scenario_key: str) -> Path:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    return VIDEOS_DIR / f"aurora_supernova_{scenario_key}_hvc1.mp4"
