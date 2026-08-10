"""Shared video, editorial-timeline, and cache settings for AURORA renders."""

from dataclasses import dataclass

import numpy as np

from core.aurora_paths import map_cache_path


@dataclass(frozen=True)
class VideoRenderConfig:
    """One project-wide profile for AURORA animation renderers."""

    duration_seconds: float = 40.0
    fps: int = 25
    width: int = 16384
    height: int = 8192
    pre_roll_seconds: float = 3.0
    post_roll_seconds: float = 2.0
    edge_fade_seconds: float = 0.5
    save_debug_frames: bool = True
    mobile_width: int = 7680
    mobile_height: int = 4320
    mobile_crf: int = 14
    encoder_preset: str = "slow"

    def __post_init__(self):
        if self.duration_seconds <= 0.0:
            raise ValueError("video duration must be positive")
        if self.fps <= 0:
            raise ValueError("video FPS must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video dimensions must be positive")
        if self.width % 2 or self.height % 2:
            raise ValueError("video dimensions must be even for YUV 4:2:0")
        if self.pre_roll_seconds < 0.0 or self.post_roll_seconds < 0.0:
            raise ValueError("video pre-roll and post-roll cannot be negative")
        if self.edge_fade_seconds <= 0.0:
            raise ValueError("video edge fade must be positive")
        if self.active_duration_seconds <= 0.0:
            raise ValueError("pre-roll and post-roll leave no active video time")

    @property
    def frame_count(self):
        return int(round(self.duration_seconds * self.fps))

    @property
    def actual_duration_seconds(self):
        return self.frame_count / self.fps

    @property
    def active_duration_seconds(self):
        return (
            self.actual_duration_seconds
            - self.pre_roll_seconds
            - self.post_roll_seconds
        )

    @property
    def active_end_seconds(self):
        return self.actual_duration_seconds - self.post_roll_seconds

    @property
    def duration_cache_tag(self):
        return f"{self.actual_duration_seconds:g}s".replace(".", "p")


# Change duration_seconds here to retime both animation renderers together.
VIDEO_CONFIG = VideoRenderConfig(duration_seconds=40.0)


def build_editorial_timeline(
    activity_start,
    activity_end,
    config=VIDEO_CONFIG,
):
    """Map one activity interval into the shared pre-roll/active/post-roll."""
    activity_start = float(activity_start)
    activity_end = float(activity_end)
    if not activity_end > activity_start:
        raise ValueError("activity interval must have positive duration")

    video_seconds = (
        np.arange(config.frame_count, dtype=np.float32)
        / np.float32(config.fps)
    )
    active_progress = np.clip(
        (video_seconds - config.pre_roll_seconds)
        / config.active_duration_seconds,
        0.0,
        1.0,
    )
    activity_times = (
        activity_start
        + active_progress * (activity_end - activity_start)
    ).astype(np.float32, copy=False)

    fade_in = np.clip(
        (video_seconds - config.pre_roll_seconds)
        / config.edge_fade_seconds,
        0.0,
        1.0,
    )
    fade_out = np.clip(
        (config.active_end_seconds - video_seconds)
        / config.edge_fade_seconds,
        0.0,
        1.0,
    )
    opacity = np.minimum(fade_in, fade_out)
    opacity[
        (video_seconds < config.pre_roll_seconds)
        | (video_seconds >= config.active_end_seconds)
    ] = 0.0
    return (
        activity_times,
        opacity.astype(np.float32, copy=False),
        active_progress.astype(np.float32, copy=False),
    )


def cache_artifact_paths(
    artifact_name,
    overlay_name,
    version,
    config=VIDEO_CONFIG,
):
    """Return duration-aware preview and sparse-overlay cache paths."""
    suffix = f"v{int(version)}_{config.duration_cache_tag}"
    video_cache = map_cache_path("videos")
    return (
        video_cache / f"{artifact_name}_preview_{suffix}.png",
        video_cache / f"{overlay_name}_{suffix}",
    )


def build_hevc_command(output, width, height, config=VIDEO_CONFIG):
    """Build the shared raw-RGB to 10-bit HEVC FFmpeg command."""
    return [
        "ffmpeg",
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
        f"{int(width)}x{int(height)}",
        "-r",
        str(config.fps),
        "-i",
        "-",
        "-vf",
        "format=yuv420p10le",
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p10le",
        "-preset",
        config.encoder_preset,
        "-x265-params",
        "log-level=none",
        str(output),
    ]


def build_hvc1_remux_command(source, output):
    """Build the shared stream-copy command for Apple-compatible HEVC tags."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-c:v",
        "copy",
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
        str(output),
    ]


def build_mobile_hevc_command(source, output, config=VIDEO_CONFIG):
    """Build the shared optional 8K mobile transcode command."""
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        (
            f"scale={config.mobile_width}:{config.mobile_height}:"
            "flags=lanczos"
        ),
        "-c:v",
        "libx265",
        "-tag:v",
        "hvc1",
        "-pix_fmt",
        "yuv420p10le",
        "-crf",
        str(config.mobile_crf),
        "-preset",
        config.encoder_preset,
        "-x265-params",
        "log-level=none",
        "-movflags",
        "+faststart",
        str(output),
    ]
