"""Single-file launcher for the AURORA supernova simulator.

Edit the settings in the first block, then run:

    python -m supernova

or:

    python supernova/run_supernova.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.aurora_console import console

from supernova.background import (
    BACKGROUND_MODES,
    BackgroundConfig,
    background_native_dimensions,
    resolve_background,
)
from supernova.light_curve import SupernovaLightCurveModel
from supernova.photometry import FILTERS, photometry_metadata
from supernova.plotting import (
    build_comparison,
    write_comparison_data,
    write_comparison_svg,
    write_light_curve_svg,
    write_monte_carlo_svg,
)
from supernova.scenarios import available_scenarios, load_scenario, run_monte_carlo


# =============================================================================
# USTAWIENIA STARTOWE
# =============================================================================
#
# Po uruchomieniu kreator pyta o wszystkie ustawienia potrzebne w wybranym
# trybie. Ponizsze wartosci sa odpowiedziami domyslnymi po nacisnieciu Enter.

# True uruchamia kreator pytan po komendzie `python run_supernova.py`.
# W procesie bez terminala ponizsze wartosci pozostaja ustawieniami wsadowymi.
INTERACTIVE_SETUP = True

# Co ma zrobic program?
# Domyslnie jedna komenda oblicza krzywa blasku i generuje gotowe wideo MP4.
# Dostepne: "simulate", "monte_carlo", "render", "simulate_and_render", "compare", "all"
RUN_MODE = "render"

# Jaki scenariusz uruchomic?
# Wbudowane: "sn1987a", "betelgeuse", "eta_carinae", "type_ia_reference"
# Mozesz tez podac sciezke do wlasnego pliku JSON.
SCENARIO = "sn1987a"

# Co renderowac w trybach zawierajacych wideo:
# - "single"       - tylko SCENARIO powyzej,
# - "historical"   - wszystkie obserwowane historycznie,
# - "hypothetical" - wszystkie hipotetyczne,
# - "all"          - absolutnie wszystkie, lacznie z modelem referencyjnym Ia.
RENDER_SCENARIO_GROUP = "single"
RENDER_SCENARIO_GROUPS = ("single", "historical", "hypothetical", "all")
# "separate" zachowuje osobny film dla kazdego scenariusza;
# "combined" umieszcza cala wybrana grupe na jednej animacji.
RENDER_GROUP_LAYOUT = "separate"
RENDER_GROUP_LAYOUTS = ("separate", "combined")

# Katalog wynikow CSV/JSON. Wideo trafia domyslnie do katalogu video projektu,
# chyba ze ustawisz VIDEO_OUTPUT_PATH ponizej.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Siatka czasu dla krzywej blasku.
SIMULATED_DAYS = 500.0
TIME_SAMPLES = 1201
OBSERVATION_FILTER = "V"

# Porownania: "historical", "hypothetical", "all", "scenario-by-scenario".
COMPARE_MODE = "historical"
COMPARE_SCENARIOS = ("sn1987a", "sn1885a")

# Monte Carlo dla scenariuszy hipotetycznych.
MONTE_CARLO_SAMPLES = 200
MONTE_CARLO_SEED = 42
MONTE_CARLO_TIME_SAMPLES = 501

# Tlo obserwacyjne dla renderu.
# Dostepne: "all_sky", "region", "custom", "catalog"
BACKGROUND_MODE = "all_sky"
AURORA_RESOLUTION = "16k"
SHOW_CONSTELLATIONS = False
SKY_MAP_BACKGROUND = "plain"

# Sciezki potrzebne tylko dla wybranych typow tla:
# - "region": BACKGROUND_IMAGE_PATH + REGION_LAYOUT_PATH
# - "custom": BACKGROUND_IMAGE_PATH
# - "catalog": STAR_CATALOG_PATH
BACKGROUND_IMAGE_PATH: Path | None = None
REGION_LAYOUT_PATH: Path | None = None
STAR_CATALOG_PATH: Path | None = None
# Dla BACKGROUND_MODE = "region" kreator pokazuje poprawne pary PNG+NPZ z
# maps/regions. Ponizsze wartosci sa tylko domyslne dla trybu wsadowego.
REGION_NAME: str | None = None
REGION_INDEX = 0

# Parametry animacji.
# Domyslnie film ma dokladnie taki rozmiar jak mapa (zwykle 16384x8192).
# Ustaw False tylko wtedy, gdy swiadomie chcesz przeskalowac obraz.
VIDEO_MATCH_BACKGROUND_SIZE = True
VIDEO_WIDTH = 16384
VIDEO_HEIGHT = 8192
VIDEO_FPS = 25
VIDEO_DURATION_SECONDS = 24.0
# Fizyczny zakres osi czasu. Wybuch ma t=0; ujemne dni sa pokazywane przed nim.
# 0.18 oznacza, ze wybuch nastapi po ok. 18% aktywnej czesci filmu.
VIDEO_SIMULATED_DAYS = 450.0
VIDEO_PRE_EXPLOSION_DAYS = 10.0
VIDEO_EXPLOSION_POSITION_FRACTION = 0.18
# "auto": maksimum w srodku, a dla dlugiego plateau - srodek jasnej fazy.
# Dostepne takze: "peak" i "plateau".
VIDEO_TIMELINE_ALIGNMENT = "auto"
VIDEO_PLATEAU_THRESHOLD_DAYS = 45.0
VIDEO_PLATEAU_MAGNITUDE_WINDOW = 0.5
# CRF 10 jest wizualnie bliski bezstratnemu i zachowuje drobne gwiazdy.
# VIDEO_LOSSLESS = True daje HEVC bezstratne kosztem bardzo duzego pliku.
VIDEO_CRF = 10
VIDEO_ENCODER_PRESET = "slow"
VIDEO_PRESERVE_STAR_FIELD_DETAIL = True
VIDEO_LOSSLESS = False
VIDEO_SHOW_SHELL = True
VIDEO_SHOW_HALO = True
VIDEO_SHOW_LABELS = True
# None oznacza: supernova/output/<scenariusz>_supernova.mp4
VIDEO_OUTPUT_PATH: Path | None = None


_SKY_BACKGROUND_LABELS = {
    "plain": "plain",
    "constellations": "constellations",
    "coordinates": "coordinate grid",
    "poland_limits": "Poland limits",
    "constellations_coordinates": "constellations + coordinates",
    "constellations_poland_limits": "constellations + Poland limits",
    "coordinates_poland_limits": "coordinates + Poland limits",
    "constellations_coordinates_poland_limits": (
        "constellations + coordinates + Poland limits"
    ),
}


def _hammer_background_map_name(profile, background: str) -> str:
    """Return a full-sky filename with old/new aurora_resolution APIs."""
    method = getattr(profile, "hammer_background_map_name", None)
    if callable(method):
        return method(background)
    suffix = "" if background == "plain" else f"_{background}"
    return f"aurora_sky_map_hammer_{profile.tag}{suffix}.png"


def _available_sky_backgrounds(resolution_api, profile, maps_dir: Path) -> tuple[str, ...]:
    """Discover ready maps even when older AURORA core lacks this helper."""
    discover = getattr(resolution_api, "available_sky_map_backgrounds", None)
    if callable(discover):
        return tuple(discover(profile, maps_dir))
    return tuple(
        background
        for background in _SKY_BACKGROUND_LABELS
        if (Path(maps_dir) / _hammer_background_map_name(profile, background)).is_file()
    )


def _prompt_sky_background(resolution_api, *, default: str, available: tuple[str, ...]) -> str:
    """Use the shared prompt when supported, otherwise show the same short menu."""
    shared_prompt = getattr(resolution_api, "prompt_sky_map_background", None)
    if callable(shared_prompt):
        try:
            return shared_prompt(default=default, available_backgrounds=available)
        except TypeError:
            # Older core versions offered this prompt without the map-filtering
            # argument. Use the local equivalent so missing files are hidden.
            pass
    selected_default = default if default in available else available[0]
    return prompt_choice(
        "Background map",
        tuple(
            (background, _SKY_BACKGROUND_LABELS.get(background, background))
            for background in available
        ),
        selected_default,
    )


def _background_constellation_flag(resolution_api, background: str) -> bool:
    flags = getattr(resolution_api, "sky_map_background_flags", None)
    if callable(flags):
        return bool(flags(background)[0])
    return "constellations" in background.split("_")


def prompt_choice(
    title: str,
    choices: tuple[tuple[str, str], ...],
    default: str,
) -> str:
    """Ask one numbered question and accept a number or canonical value."""
    values = {value for value, _label in choices}
    normalized_values = {value.casefold(): value for value in values}
    if default not in values:
        raise ValueError(f"invalid default choice: {default}")
    default_number = next(
        index for index, (value, _label) in enumerate(choices, start=1)
        if value == default
    )
    console.print(f"\n{title}")
    for index, (value, label) in enumerate(choices, start=1):
        marker = " (default)" if value == default else ""
        console.print(f"  {index}. {label}{marker}")
    while True:
        try:
            response = console.prompt(
                f"Selection [Enter = {default_number}]"
            ).strip().casefold()
        except EOFError:
            return default
        if not response:
            return default
        if response.isdigit() and 1 <= int(response) <= len(choices):
            return choices[int(response) - 1][0]
        if response in normalized_values:
            return normalized_values[response]
        console.warning("Choose one of the displayed numbers")


def prompt_number(
    label: str,
    default: float | int,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | int:
    """Ask for a validated numeric setting, using the current value on Enter."""
    while True:
        try:
            response = console.prompt(f"{label} [Enter = {default}]").strip()
        except EOFError:
            return default
        if not response:
            return default
        try:
            value = int(response) if integer else float(response)
        except ValueError:
            console.warning("Enter a valid number")
            continue
        if minimum is not None and value < minimum:
            console.warning(f"Minimum value: {minimum}")
            continue
        if maximum is not None and value > maximum:
            console.warning(f"Maximum value: {maximum}")
            continue
        return value


def prompt_yes_no(label: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        try:
            response = console.prompt(f"{label} [{hint}]").strip().casefold()
        except EOFError:
            return default
        if not response:
            return default
        if response in {"t", "tak", "y", "yes"}:
            return True
        if response in {"n", "nie", "no"}:
            return False
        console.warning("Answer yes or no")


def prompt_path(label: str, default: Path | None = None) -> Path | None:
    default_text = str(default) if default is not None else "automatycznie"
    try:
        response = console.prompt(f"{label} [Enter = {default_text}]").strip()
    except EOFError:
        return default
    return Path(response).expanduser() if response else default


def prompt_scenario(default: str, *, require_uncertainty: bool = False) -> str:
    scenarios = tuple(load_scenario(name) for name in available_scenarios())
    if require_uncertainty:
        scenarios = tuple(scenario for scenario in scenarios if scenario.uncertainty)
    status_labels = {
        "historical_observation": "historically observed",
        "hypothetical": "hypothetical",
        "reference": "reference model",
    }
    choices = tuple(
        (
            scenario.key,
            f"{scenario.key} — {status_labels.get(scenario.status, scenario.status)}",
        )
        for scenario in scenarios
    )
    default_key = Path(default).stem
    return prompt_choice(
        "Scenario",
        choices,
        default_key if default_key in {value for value, _ in choices} else choices[0][0],
    )


def configure_from_questions() -> bool:
    """Ask only the short, project-standard set of runtime questions."""
    global RUN_MODE, SCENARIO, RENDER_SCENARIO_GROUP, RENDER_GROUP_LAYOUT
    global OBSERVATION_FILTER, COMPARE_MODE
    global BACKGROUND_MODE, AURORA_RESOLUTION, SHOW_CONSTELLATIONS
    global SKY_MAP_BACKGROUND, BACKGROUND_IMAGE_PATH, REGION_LAYOUT_PATH
    global STAR_CATALOG_PATH, REGION_NAME, REGION_INDEX

    from core.aurora_paths import MAPS_DIR, region_map_path
    from core.aurora_region_selection import select_sky_region
    from core import aurora_resolution as resolution_api

    get_resolution = resolution_api.get_resolution

    console.section("SUPERNOVA SETUP")
    RUN_MODE = prompt_choice(
        "Operation",
        (
            ("render", "Render video"),
            ("simulate_and_render", "Light curve and video"),
            ("simulate", "Light curve only"),
            ("monte_carlo", "Monte Carlo"),
            ("compare", "Compare scenarios"),
        ),
        RUN_MODE if RUN_MODE != "all" else "render",
    )
    has_render = RUN_MODE in {"render", "simulate_and_render"}
    has_simulation = RUN_MODE in {"simulate", "simulate_and_render"}
    has_monte_carlo = RUN_MODE == "monte_carlo"
    has_comparison = RUN_MODE == "compare"

    if has_render:
        RENDER_SCENARIO_GROUP = prompt_choice(
            "Scenarios",
            (
                ("single", "One scenario"),
                ("historical", "All historical scenarios"),
                ("hypothetical", "All hypothetical scenarios"),
                ("all", "All scenarios"),
            ),
            RENDER_SCENARIO_GROUP,
        )
        if RENDER_SCENARIO_GROUP == "single":
            RENDER_GROUP_LAYOUT = "separate"
        else:
            RENDER_GROUP_LAYOUT = prompt_choice(
                "Group output",
                (
                    ("separate", "One video per scenario"),
                    ("combined", "All events in one video"),
                ),
                RENDER_GROUP_LAYOUT,
            )

    if RENDER_SCENARIO_GROUP == "single" or has_simulation or has_monte_carlo:
        SCENARIO = prompt_scenario(
            SCENARIO,
            require_uncertainty=has_monte_carlo,
        )
    if has_comparison:
        COMPARE_MODE = prompt_choice(
            "Comparison group",
            (
                ("historical", "Historical"),
                ("hypothetical", "Hypothetical"),
                ("all", "All"),
                ("scenario-by-scenario", "Selected scenarios"),
            ),
            COMPARE_MODE,
        )

    OBSERVATION_FILTER = prompt_choice(
        "Observation filter",
        tuple((name, name) for name in FILTERS),
        OBSERVATION_FILTER,
    )

    if has_render:
        profile = resolution_api.prompt_resolution(
            default=get_resolution(AURORA_RESOLUTION).k
        )
        AURORA_RESOLUTION = profile.tag
        BACKGROUND_IMAGE_PATH = None
        REGION_LAYOUT_PATH = None
        STAR_CATALOG_PATH = None

        if RENDER_GROUP_LAYOUT == "combined" and RENDER_SCENARIO_GROUP != "single":
            BACKGROUND_MODE = "all_sky"
            console.detail("Combined animations use the full-sky map")
        else:
            coverage = resolution_api.prompt_sky_map_mode(
                default="region" if BACKGROUND_MODE == "region" else "full"
            )
            BACKGROUND_MODE = "region" if coverage == "region" else "all_sky"

        if BACKGROUND_MODE == "all_sky":
            available = _available_sky_backgrounds(
                resolution_api, profile, MAPS_DIR
            )
            if not available:
                raise FileNotFoundError(
                    f"No ready {profile.tag} full-sky maps found in {MAPS_DIR}"
                )
            SKY_MAP_BACKGROUND = _prompt_sky_background(
                resolution_api,
                default=SKY_MAP_BACKGROUND,
                available=available,
            )
            SHOW_CONSTELLATIONS = _background_constellation_flag(
                resolution_api, SKY_MAP_BACKGROUND
            )
        else:
            SKY_MAP_BACKGROUND = "plain"
            selection = select_sky_region(
                region_map_path(profile.region_map_name),
                region_map_path(profile.region_layout_name),
            )
            BACKGROUND_IMAGE_PATH = selection.map_path
            REGION_LAYOUT_PATH = selection.layout_path
            REGION_NAME = selection.map_path.stem
            REGION_INDEX = 0

    return True


def _time_grid(days: float, samples: int) -> np.ndarray:
    if days <= 0.0:
        raise ValueError("SIMULATED_DAYS must be positive")
    if samples < 2:
        raise ValueError("TIME_SAMPLES must be at least 2")
    return np.linspace(0.0, days, samples)


def _scenario_output_stem(scenario_key: str, suffix: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{scenario_key}_{suffix}"


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
        "total_luminosity_w", "total_apparent_flux_w_m2",
        "apparent_magnitude", "bolometric_magnitude", "visual_flux_scale",
        "angular_shell_radius",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        band_columns = tuple(
            f"{name}_{quantity}"
            for name in FILTERS
            for quantity in ("intrinsic_flux_w_m2", "observed_flux_w_m2", "apparent_magnitude")
        )
        writer.writerow(columns + band_columns)
        arrays = [getattr(curve, name) for name in columns]
        arrays.extend(
            getattr(curve.bands[name], quantity)
            for name in FILTERS
            for quantity in ("intrinsic_flux_w_m2", "observed_flux_w_m2", "apparent_magnitude")
        )
        for row in zip(*arrays):
            writer.writerow((f"{float(value):.10g}" for value in row))


def _print_scenario_notice(scenario) -> None:
    console.detail(f"Scenario: {scenario.key} ({scenario.status})")
    console.detail(scenario.description)
    if scenario.disclaimer:
        console.warning(scenario.disclaimer)


def _run_simulation(scenario) -> tuple[Path, Path]:
    model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
    curve = model.evaluate(_time_grid(SIMULATED_DAYS, TIME_SAMPLES))
    output = _scenario_output_stem(scenario.key, "light_curve.csv")
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
        "observation_filter": OBSERVATION_FILTER,
        "peak_apparent_visual_magnitude": curve.peak_apparent_magnitude,
        "peak_visual_time_days": curve.peak_visual_time_days,
        "distance_parsec": scenario.progenitor.distance.parsecs,
        "distance_modulus": scenario.progenitor.distance.distance_modulus,
        "extinction_av_mag": scenario.progenitor.extinction_av_mag,
        "energy_foe": model.explosion.energy_foe,
        "ejecta_mass_solar": model.explosion.ejecta_mass_solar,
        "nickel56_mass_solar": model.explosion.nickel56_mass_solar,
        "velocity_km_s": model.explosion.characteristic_velocity_m_s / 1000.0,
        "diffusion_time_days": model.explosion.diffusion_time_days,
        "photometry": photometry_metadata(),
        "shock_breakout": {
            "included_in_classical_visual_curve": False,
            "peak_luminosity_w": float(np.max(curve.shock_breakout_luminosity_w)),
            "temperature_k": scenario.overrides.shock_breakout_temperature_k,
            "duration_hours": scenario.overrides.shock_breakout_duration_hours,
        },
        "visual_layers": {
            "point_source": "scaled from extincted band flux",
            "halo": "display-only PSF halo; does not change photometry",
            "shell": "illustrative bounded overlay; physical angular radius is stored separately",
        },
    }
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plot = write_light_curve_svg(
        curve, scenario, _scenario_output_stem(scenario.key, "filters.svg")
    )
    console.success(f"Light curve: {output}")
    console.detail(f"Metadata: {metadata}")
    console.detail(f"Plot: {plot}")
    return output, metadata


def _run_monte_carlo(scenario) -> Path:
    envelope = run_monte_carlo(
        scenario,
        _time_grid(SIMULATED_DAYS, MONTE_CARLO_TIME_SAMPLES),
        samples=MONTE_CARLO_SAMPLES,
        seed=MONTE_CARLO_SEED,
    )
    output = _scenario_output_stem(scenario.key, "monte_carlo.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["observer_time_days"]
            + [f"luminosity_w_p{p:g}" for p in envelope.percentiles]
            + [f"apparent_bolometric_mag_p{p:g}" for p in envelope.percentiles]
            + [f"{OBSERVATION_FILTER}_apparent_mag_p{p:g}" for p in envelope.percentiles]
        )
        for index, time in enumerate(envelope.observer_time_days):
            writer.writerow(
                [f"{time:.10g}"]
                + [f"{row[index]:.10g}" for row in envelope.luminosity_percentiles_w]
                + [f"{row[index]:.10g}" for row in envelope.apparent_magnitude_percentiles]
                + [f"{row[index]:.10g}" for row in envelope.filter_magnitude_percentiles[OBSERVATION_FILTER]]
            )
    metadata = output.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "scenario": scenario.key,
                "filter": OBSERVATION_FILTER,
                "percentiles": envelope.percentiles,
                "peak_time_percentiles_days": envelope.peak_time_percentiles_days.tolist(),
                "peak_magnitude_percentiles": envelope.peak_filter_magnitude_percentiles[OBSERVATION_FILTER].tolist(),
                "nominal_peak_magnitude": float(np.min(envelope.nominal_filter_magnitudes[OBSERVATION_FILTER])),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    plot = write_monte_carlo_svg(
        envelope,
        _scenario_output_stem(scenario.key, "monte_carlo.svg"),
        scenario_key=scenario.key,
        filter_name=OBSERVATION_FILTER,
    )
    console.success(f"Monte Carlo envelope: {output}")
    console.detail(
        "Peak-time percentiles [days]: "
        + ", ".join(f"{value:.2f}" for value in envelope.peak_time_percentiles_days)
    )
    console.detail(f"Monte Carlo metadata: {metadata}")
    console.detail(f"Monte Carlo plot: {plot}")
    return output


def render_video_output_path(scenario_key: str, artifact: str, *, batch: bool) -> Path:
    """Return a collision-free output path for single and grouped renders."""
    if VIDEO_OUTPUT_PATH is None:
        return _scenario_output_stem(scenario_key, artifact + ".mp4")
    if batch:
        suffix = VIDEO_OUTPUT_PATH.suffix or ".mp4"
        return VIDEO_OUTPUT_PATH.with_name(
            f"{VIDEO_OUTPUT_PATH.stem}_{scenario_key}{suffix}"
        )
    return VIDEO_OUTPUT_PATH


def _render_background_and_size() -> tuple[BackgroundConfig, int, int]:
    background = resolve_background(
        BackgroundConfig(
            mode=BACKGROUND_MODE,
            image_path=BACKGROUND_IMAGE_PATH,
            layout_path=REGION_LAYOUT_PATH,
            catalog_path=STAR_CATALOG_PATH,
            aurora_resolution=AURORA_RESOLUTION,
            constellations=SHOW_CONSTELLATIONS,
            sky_map_background=SKY_MAP_BACKGROUND,
            region_name=REGION_NAME,
            region_index=REGION_INDEX,
        )
    )
    width, height = (
        background_native_dimensions(background)
        if VIDEO_MATCH_BACKGROUND_SIZE
        else (VIDEO_WIDTH, VIDEO_HEIGHT)
    )
    return background, width, height


def _animation_config(width: int, height: int):
    from supernova.animation import AnimationConfig

    return AnimationConfig(
        width=width,
        height=height,
        fps=VIDEO_FPS,
        duration_seconds=VIDEO_DURATION_SECONDS,
        simulated_days=VIDEO_SIMULATED_DAYS,
        pre_explosion_days=VIDEO_PRE_EXPLOSION_DAYS,
        explosion_position_fraction=VIDEO_EXPLOSION_POSITION_FRACTION,
        timeline_alignment=VIDEO_TIMELINE_ALIGNMENT,
        plateau_threshold_days=VIDEO_PLATEAU_THRESHOLD_DAYS,
        plateau_magnitude_window=VIDEO_PLATEAU_MAGNITUDE_WINDOW,
        filter_name=OBSERVATION_FILTER,
        show_shell=VIDEO_SHOW_SHELL,
        show_halo=VIDEO_SHOW_HALO,
        show_labels=VIDEO_SHOW_LABELS,
        video_crf=VIDEO_CRF,
        encoder_preset=VIDEO_ENCODER_PRESET,
        preserve_star_field_detail=VIDEO_PRESERVE_STAR_FIELD_DETAIL,
        lossless_video=VIDEO_LOSSLESS,
    )


def _run_render(scenario, *, batch: bool = False) -> tuple[Path, Path]:
    from supernova.animation import SupernovaAnimator

    background, width, height = _render_background_and_size()
    model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
    animation = SupernovaAnimator(
        model,
        _animation_config(width, height),
        scenario_status=scenario.status,
    )
    artifact = "supernova"
    if background.mode == "region" and background.image_path is not None:
        artifact += "_" + background.image_path.stem.casefold()
    output = render_video_output_path(scenario.key, artifact, batch=batch)
    quality = "lossless" if VIDEO_LOSSLESS else f"CRF {VIDEO_CRF}"
    size_source = "background native size" if VIDEO_MATCH_BACKGROUND_SIZE else "manual size"
    console.detail(f"Video canvas: {width} x {height} ({size_source})")
    console.detail(f"HEVC quality: {quality}, preset {VIDEO_ENCODER_PRESET}")
    anchor_label = "bright-phase midpoint" if animation.timeline_alignment == "plateau" else "visual peak"
    console.detail(
        f"Timeline: {animation.timeline_start_days:+.1f} to "
        f"{animation.timeline_end_days:+.1f} days; explosion t=0 at "
        f"{100.0 * VIDEO_EXPLOSION_POSITION_FRACTION:.0f}% of active video"
    )
    console.detail(
        f"Movie midpoint anchor: {anchor_label} at "
        f"t={animation.timeline_anchor_days:+.1f} days"
    )
    if background.mode == "region":
        console.detail(f"Region map: {background.image_path}")
        console.detail(f"Region layout: {background.layout_path}")
    video, preview = animation.render_video(background, output)
    console.success(f"Video: {video}")
    console.detail(f"Preview: {preview}")
    return video, preview


def _run_combined_render(scenarios: tuple) -> tuple[Path, Path]:
    from supernova.animation import MultiScenarioAnimator

    if BACKGROUND_MODE == "region":
        raise ValueError(
            "combined group animation requires all_sky, custom, or catalog background"
        )
    background, width, height = _render_background_and_size()
    animation = MultiScenarioAnimator(
        tuple(scenarios),
        _animation_config(width, height),
    )
    artifact_key = f"combined_{RENDER_SCENARIO_GROUP}"
    if VIDEO_OUTPUT_PATH is None:
        output = _scenario_output_stem(
            artifact_key,
            f"supernova_{OBSERVATION_FILTER.casefold()}.mp4",
        )
    else:
        suffix = VIDEO_OUTPUT_PATH.suffix or ".mp4"
        output = VIDEO_OUTPUT_PATH.with_name(
            f"{VIDEO_OUTPUT_PATH.stem}_{artifact_key}{suffix}"
        )
    console.section(
        f"Combined animation: {RENDER_SCENARIO_GROUP} "
        f"({len(scenarios)} synchronized events)"
    )
    console.detail(f"Video canvas: {width} x {height}")
    console.detail(f"Filter: {OBSERVATION_FILTER}; shared explosion epoch t=0")
    console.detail("Each event keeps its own distance, flux, peak, and plateau")
    video, preview = animation.render_video(background, output)
    console.success(f"Combined video: {video}")
    console.detail(f"Preview: {preview}")
    return video, preview


def render_scenarios_for_group(primary_scenario) -> tuple:
    """Resolve the code-configured render selection without changing CLI APIs."""
    if RENDER_SCENARIO_GROUP == "single":
        return (primary_scenario,)
    return tuple(
        load_scenario(name) for name in available_scenarios(RENDER_SCENARIO_GROUP)
    )


def _run_render_group(primary_scenario) -> tuple[tuple[Path, Path], ...]:
    scenarios = render_scenarios_for_group(primary_scenario)
    if RENDER_GROUP_LAYOUT == "combined" and len(scenarios) > 1:
        return (_run_combined_render(scenarios),)
    batch = len(scenarios) > 1 or RENDER_SCENARIO_GROUP != "single"
    if batch:
        console.section(
            f"Sequential video render: {RENDER_SCENARIO_GROUP} "
            f"({len(scenarios)} scenarios)"
        )
    outputs: list[tuple[Path, Path]] = []
    for index, scenario in enumerate(scenarios, start=1):
        if batch:
            console.section(f"Render {index}/{len(scenarios)}: {scenario.key}")
            _print_scenario_notice(scenario)
        outputs.append(_run_render(scenario, batch=batch))
    if batch:
        console.success(
            f"Completed {len(outputs)}/{len(scenarios)} scenario videos"
        )
    return tuple(outputs)


def _run_comparison() -> tuple[Path, Path, Path]:
    names = COMPARE_SCENARIOS if COMPARE_MODE == "scenario-by-scenario" else ()
    series = build_comparison(
        COMPARE_MODE,
        _time_grid(SIMULATED_DAYS, TIME_SAMPLES),
        filter_name=OBSERVATION_FILTER,
        scenario_names=names,
    )
    stem = _scenario_output_stem(f"comparison_{COMPARE_MODE}", OBSERVATION_FILTER.lower())
    csv_path, json_path = write_comparison_data(series, stem, filter_name=OBSERVATION_FILTER)
    plot = write_comparison_svg(series, stem.with_suffix(".svg"), filter_name=OBSERVATION_FILTER)
    console.success(f"Comparison plot: {plot}")
    console.detail(f"Comparison data: {csv_path}")
    console.detail(f"Comparison metadata: {json_path}")
    return plot, csv_path, json_path


def main() -> int:
    if INTERACTIVE_SETUP and sys.stdin.isatty():
        if not configure_from_questions():
            console.warning("Run cancelled")
            return 0
    if RUN_MODE not in {"simulate", "monte_carlo", "render", "simulate_and_render", "compare", "all"}:
        raise ValueError("RUN_MODE must be simulate, monte_carlo, render, simulate_and_render, compare, or all")
    if BACKGROUND_MODE not in BACKGROUND_MODES:
        raise ValueError(f"BACKGROUND_MODE must be one of: {', '.join(BACKGROUND_MODES)}")
    if OBSERVATION_FILTER not in FILTERS:
        raise ValueError(f"OBSERVATION_FILTER must be one of: {', '.join(FILTERS)}")
    if RENDER_SCENARIO_GROUP not in RENDER_SCENARIO_GROUPS:
        raise ValueError(
            "RENDER_SCENARIO_GROUP must be single, historical, hypothetical, or all"
        )
    if RENDER_GROUP_LAYOUT not in RENDER_GROUP_LAYOUTS:
        raise ValueError("RENDER_GROUP_LAYOUT must be separate or combined")

    scenario = load_scenario(SCENARIO)
    _print_scenario_notice(scenario)
    console.detail("Available scenarios: " + ", ".join(available_scenarios()))

    if RUN_MODE in {"simulate", "simulate_and_render", "all"}:
        _run_simulation(scenario)
    if RUN_MODE in {"monte_carlo", "all"}:
        _run_monte_carlo(scenario)
    if RUN_MODE in {"render", "simulate_and_render", "all"}:
        _run_render_group(scenario)
    if RUN_MODE in {"compare", "all"}:
        _run_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
