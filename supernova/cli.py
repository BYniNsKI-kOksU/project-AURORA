"""Command-line interface for simulation, calibration, Monte Carlo, and video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core.aurora_console import console

from .light_curve import SupernovaLightCurveModel
from .photometry import FILTERS, photometry_metadata
from .plotting import (
    build_comparison,
    write_comparison_data,
    write_comparison_svg,
    write_monte_carlo_svg,
)
from .scenarios import available_scenarios, load_scenario, run_monte_carlo


BACKGROUND_MODES = ("all_sky", "region", "custom", "catalog")


def _time_grid(days: float, samples: int) -> np.ndarray:
    if days <= 0.0 or samples < 2:
        raise ValueError("days must be positive and samples must be at least 2")
    return np.linspace(0.0, days, samples)


def _write_curve(path: Path, curve) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "observer_time_days", "rest_time_days", "luminosity_w",
        "radioactive_luminosity_w", "shock_luminosity_w",
        "interaction_luminosity_w", "photospheric_radius_m",
        "rest_temperature_k", "observed_temperature_k",
        "apparent_flux_w_m2", "absolute_bolometric_magnitude",
        "apparent_bolometric_magnitude",
        "shock_breakout_luminosity_w", "dust_echo_luminosity_w",
        "apparent_magnitude", "bolometric_magnitude", "visual_flux_scale",
        "angular_shell_radius",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        band_columns = tuple(
            f"{name}_apparent_magnitude" for name in FILTERS
        )
        writer.writerow(columns + band_columns)
        arrays = [getattr(curve, name) for name in columns]
        arrays.extend(curve.bands[name].apparent_magnitude for name in FILTERS)
        for row in zip(*arrays):
            writer.writerow((f"{float(value):.10g}" for value in row))


def _print_scenario_notice(scenario) -> None:
    console.detail(f"Scenario: {scenario.key} ({scenario.status})")
    console.detail(scenario.description)
    if scenario.disclaimer:
        console.warning(scenario.disclaimer)


def command_simulate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    _print_scenario_notice(scenario)
    model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
    curve = model.evaluate(_time_grid(args.days, args.samples))
    output = args.output or Path("supernova") / "output" / f"{scenario.key}_light_curve.csv"
    _write_curve(output, curve)
    summary = {
        "scenario": scenario.key,
        "status": scenario.status,
        "disclaimer": scenario.disclaimer,
        "category": scenario.category,
        "historical_metadata": scenario.historical_metadata,
        "variants": scenario.variants,
        "supernova_type": scenario.progenitor.supernova_type,
        "redshift": curve.redshift,
        "peak_observer_time_days": curve.peak_time_days,
        "peak_luminosity_w": curve.peak_luminosity_w,
        "peak_apparent_bolometric_magnitude": float(np.min(curve.apparent_bolometric_magnitude)),
        "peak_apparent_visual_magnitude": curve.peak_apparent_magnitude,
        "peak_visual_time_days": curve.peak_visual_time_days,
        "distance_parsec": scenario.progenitor.distance.parsecs,
        "distance_modulus": scenario.progenitor.distance.distance_modulus,
        "extinction_av_mag": scenario.progenitor.extinction_av_mag,
        "photometry": photometry_metadata(),
        "energy_foe": model.explosion.energy_foe,
        "ejecta_mass_solar": model.explosion.ejecta_mass_solar,
        "nickel56_mass_solar": model.explosion.nickel56_mass_solar,
        "velocity_km_s": model.explosion.characteristic_velocity_m_s / 1000.0,
        "diffusion_time_days": model.explosion.diffusion_time_days,
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console.success(f"Light curve: {output}")
    console.detail(f"Metadata: {metadata}")
    return 0


def command_monte_carlo(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    _print_scenario_notice(scenario)
    envelope = run_monte_carlo(
        scenario,
        _time_grid(args.days, args.time_samples),
        samples=args.samples,
        seed=args.seed,
    )
    output = args.output or Path("supernova") / "output" / f"{scenario.key}_monte_carlo.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["observer_time_days"]
            + [f"luminosity_w_p{p:g}" for p in envelope.percentiles]
            + [f"apparent_bolometric_mag_p{p:g}" for p in envelope.percentiles]
            + [f"{args.filter}_apparent_mag_p{p:g}" for p in envelope.percentiles]
        )
        for index, time in enumerate(envelope.observer_time_days):
            writer.writerow(
                [f"{time:.10g}"]
                + [f"{row[index]:.10g}" for row in envelope.luminosity_percentiles_w]
                + [f"{row[index]:.10g}" for row in envelope.apparent_magnitude_percentiles]
                + [f"{row[index]:.10g}" for row in envelope.filter_magnitude_percentiles[args.filter]]
            )
    metadata = output.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "scenario": scenario.key,
                "filter": args.filter,
                "percentiles": envelope.percentiles,
                "peak_time_percentiles_days": envelope.peak_time_percentiles_days.tolist(),
                "peak_magnitude_percentiles": envelope.peak_filter_magnitude_percentiles[args.filter].tolist(),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    plot = write_monte_carlo_svg(
        envelope,
        output.with_suffix(".svg"),
        scenario_key=scenario.key,
        filter_name=args.filter,
    )
    console.success(f"Monte Carlo envelope: {output}")
    console.detail(
        "Peak-time percentiles [days]: "
        + ", ".join(f"{value:.2f}" for value in envelope.peak_time_percentiles_days)
    )
    console.detail(f"Metadata: {metadata}")
    console.detail(f"Plot: {plot}")
    return 0


def command_render(args: argparse.Namespace) -> int:
    from .animation import AnimationConfig, SupernovaAnimator, default_video_path
    from .background import BackgroundConfig

    scenario = load_scenario(args.scenario)
    _print_scenario_notice(scenario)
    model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
    animation = SupernovaAnimator(
        model,
        AnimationConfig(
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration_seconds=args.duration,
            simulated_days=args.days,
            filter_name=args.filter,
        ),
        scenario_status=scenario.status,
    )
    background = BackgroundConfig(
        mode=args.background,
        image_path=args.background_image,
        layout_path=args.region_layout,
        catalog_path=args.star_catalog,
        aurora_resolution=args.aurora_resolution,
        constellations=args.constellations,
    )
    output = args.output or default_video_path(scenario.key)
    video, preview = animation.render_video(background, output)
    console.success(f"Video: {video}")
    console.detail(f"Preview: {preview}")
    return 0


def command_compare(args: argparse.Namespace) -> int:
    series = build_comparison(
        args.group,
        _time_grid(args.days, args.samples),
        filter_name=args.filter,
        scenario_names=tuple(args.scenarios or ()),
    )
    output = args.output or Path("supernova") / "output" / f"comparison_{args.group}_{args.filter.lower()}"
    csv_path, json_path = write_comparison_data(series, output, filter_name=args.filter)
    svg_path = write_comparison_svg(series, output.with_suffix(".svg"), filter_name=args.filter)
    console.success(f"Comparison: {svg_path}")
    console.detail(f"Data: {csv_path}")
    console.detail(f"Metadata: {json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m supernova",
        description="AURORA semi-analytic, non-predictive supernova simulator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list-scenarios", help="list bundled calibrations")
    listing.set_defaults(handler=lambda args: print("\n".join(available_scenarios())) or 0)

    simulate = subparsers.add_parser("simulate", help="write one deterministic light curve")
    simulate.add_argument("--scenario", default="sn1987a")
    simulate.add_argument("--days", type=float, default=500.0)
    simulate.add_argument("--samples", type=int, default=1201)
    simulate.add_argument("--output", type=Path)
    simulate.set_defaults(handler=command_simulate)

    monte_carlo = subparsers.add_parser("monte-carlo", help="sample a scenario uncertainty model")
    monte_carlo.add_argument("--scenario", default="betelgeuse")
    monte_carlo.add_argument("--days", type=float, default=500.0)
    monte_carlo.add_argument("--time-samples", type=int, default=501)
    monte_carlo.add_argument("--samples", type=int, default=200)
    monte_carlo.add_argument("--seed", type=int, default=42)
    monte_carlo.add_argument("--filter", choices=tuple(FILTERS), default="V")
    monte_carlo.add_argument("--output", type=Path)
    monte_carlo.set_defaults(handler=command_monte_carlo)

    render = subparsers.add_parser("render", help="render one explosion and its fading shell")
    render.add_argument("--scenario", default="sn1987a")
    render.add_argument("--background", choices=BACKGROUND_MODES, default="all_sky")
    render.add_argument("--background-image", type=Path)
    render.add_argument("--region-layout", type=Path)
    render.add_argument("--star-catalog", type=Path)
    render.add_argument("--aurora-resolution", choices=("8k", "16k", "32k", "64k"), default="8k")
    render.add_argument("--constellations", action="store_true")
    render.add_argument("--width", type=int, default=1920)
    render.add_argument("--height", type=int, default=960)
    render.add_argument("--fps", type=int, default=25)
    render.add_argument("--duration", type=float, default=24.0)
    render.add_argument("--days", type=float, default=450.0)
    render.add_argument("--filter", choices=tuple(FILTERS), default="V")
    render.add_argument("--output", type=Path)
    render.set_defaults(handler=command_render)

    compare = subparsers.add_parser("compare", help="compare historical or hypothetical scenarios")
    compare.add_argument("--group", choices=("historical", "hypothetical", "all", "scenario-by-scenario"), default="historical")
    compare.add_argument("--scenarios", nargs="*")
    compare.add_argument("--filter", choices=tuple(FILTERS), default="V")
    compare.add_argument("--days", type=float, default=500.0)
    compare.add_argument("--samples", type=int, default=1201)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
