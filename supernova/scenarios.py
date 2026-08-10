"""JSON scenarios, uncertainty sampling, and Monte Carlo envelopes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .light_curve import SupernovaLightCurveModel
from .progenitor import ModelOverrides, Progenitor


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
HISTORICAL_SCENARIO_DIR = SCENARIO_DIR / "historical"
HYPOTHETICAL_SCENARIO_DIR = SCENARIO_DIR / "hypothetical"


@dataclass(frozen=True)
class Scenario:
    key: str
    description: str
    status: str
    disclaimer: str
    progenitor: Progenitor
    overrides: ModelOverrides
    uncertainty: Mapping[str, Mapping[str, float | str]]
    source_notes: tuple[str, ...]
    category: str
    historical_metadata: Mapping[str, Any]
    variants: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def is_hypothetical(self) -> bool:
        return self.status == "hypothetical"

    @property
    def is_historical(self) -> bool:
        return self.status == "historical_observation"


def _scenario_index() -> dict[str, Path]:
    paths = list(sorted(SCENARIO_DIR.glob("*.json")))
    # Grouped catalogues are canonical and override legacy root-level files
    # with the same key while preserving direct-path compatibility.
    paths.extend(sorted(HISTORICAL_SCENARIO_DIR.glob("*.json")))
    paths.extend(sorted(HYPOTHETICAL_SCENARIO_DIR.glob("*.json")))
    return {path.stem: path for path in paths}


def available_scenarios(group: str | None = None) -> tuple[str, ...]:
    scenarios = []
    for key, path in sorted(_scenario_index().items()):
        if group is None or group == "all":
            scenarios.append(key)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        status = str(raw.get("status", ""))
        if group == "historical" and status == "historical_observation":
            scenarios.append(key)
        elif group == "hypothetical" and status == "hypothetical":
            scenarios.append(key)
    return tuple(scenarios)


def load_scenario(name_or_path: str | Path) -> Scenario:
    path = Path(name_or_path)
    if not path.exists():
        path = _scenario_index().get(path.stem, SCENARIO_DIR / f"{path.stem}.json")
    if not path.exists():
        choices = ", ".join(available_scenarios())
        raise FileNotFoundError(f"unknown scenario {name_or_path!r}; available: {choices}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    progenitor = Progenitor.from_dict(raw["progenitor"])
    historical_metadata = dict(raw.get("historical_metadata", {}))
    if str(raw.get("status")) == "historical_observation":
        historical_metadata.setdefault("name", str(raw.get("key", path.stem)))
        historical_metadata.setdefault("supernova_type", progenitor.supernova_type)
        historical_metadata.setdefault("distance", progenitor.distance.to_dict())
    return Scenario(
        key=str(raw.get("key", path.stem)),
        description=str(raw["description"]),
        status=str(raw["status"]),
        disclaimer=str(raw.get("disclaimer", "")),
        progenitor=progenitor,
        overrides=ModelOverrides.from_dict(raw.get("model_overrides")),
        uncertainty=raw.get("uncertainty", {}),
        source_notes=tuple(raw.get("source_notes", ())),
        category=str(raw.get("category", "uncategorized")),
        historical_metadata=historical_metadata,
        variants=raw.get("variants", {}),
        raw=raw,
    )


def _set_dotted(data: dict[str, Any], dotted_key: str, value: float) -> None:
    parts = dotted_key.split(".")
    target = data
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = float(value)


def sample_scenario(scenario: Scenario, rng: np.random.Generator) -> tuple[Progenitor, ModelOverrides]:
    sampled = deepcopy(dict(scenario.raw))
    for dotted_key, specification in scenario.uncertainty.items():
        distribution = str(specification.get("distribution", "normal"))
        if distribution == "normal":
            value = rng.normal(float(specification["mean"]), float(specification["sigma"]))
        elif distribution == "uniform":
            value = rng.uniform(float(specification["min"]), float(specification["max"]))
        elif distribution == "lognormal":
            value = rng.lognormal(float(specification["log_mean"]), float(specification["log_sigma"]))
        else:
            raise ValueError(f"unsupported uncertainty distribution: {distribution}")
        if "min" in specification:
            value = max(value, float(specification["min"]))
        if "max" in specification:
            value = min(value, float(specification["max"]))
        _set_dotted(sampled, dotted_key, value)
    # Independent priors can draw a final mass just above the initial mass.
    # Project such a draw back into the physically allowed region.
    progenitor_data = sampled["progenitor"]
    if progenitor_data["final_mass_solar"] > progenitor_data["initial_mass_solar"]:
        progenitor_data["final_mass_solar"] = 0.97 * progenitor_data["initial_mass_solar"]
    return (
        Progenitor.from_dict(sampled["progenitor"]),
        ModelOverrides.from_dict(sampled.get("model_overrides")),
    )


@dataclass(frozen=True)
class MonteCarloEnvelope:
    observer_time_days: np.ndarray
    luminosity_percentiles_w: np.ndarray
    apparent_magnitude_percentiles: np.ndarray
    peak_time_percentiles_days: np.ndarray
    peak_magnitude_percentiles: np.ndarray
    filter_magnitude_percentiles: Mapping[str, np.ndarray]
    peak_filter_magnitude_percentiles: Mapping[str, np.ndarray]
    nominal_luminosity_w: np.ndarray
    nominal_filter_magnitudes: Mapping[str, np.ndarray]
    percentiles: tuple[float, ...]


def run_monte_carlo(
    scenario: Scenario,
    observer_time_days: np.ndarray,
    *,
    samples: int = 200,
    seed: int = 42,
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> MonteCarloEnvelope:
    if samples < 2:
        raise ValueError("Monte Carlo requires at least two samples")
    if not scenario.uncertainty:
        raise ValueError("scenario has no uncertainty model")
    time = np.asarray(observer_time_days, dtype=np.float64)
    rng = np.random.default_rng(seed)
    luminosities = np.empty((samples, time.size), dtype=np.float64)
    magnitudes = np.empty_like(luminosities)
    peak_times = np.empty(samples, dtype=np.float64)
    peak_magnitudes = np.empty(samples, dtype=np.float64)
    filter_magnitudes = {
        name: np.empty_like(luminosities)
        for name in ("UV", "U", "B", "V", "R", "I", "IR")
    }
    peak_filter_magnitudes = {
        name: np.empty(samples, dtype=np.float64) for name in filter_magnitudes
    }
    for index in range(samples):
        progenitor, overrides = sample_scenario(scenario, rng)
        curve = SupernovaLightCurveModel(progenitor, overrides).evaluate(time)
        luminosities[index] = curve.luminosity_w
        magnitudes[index] = curve.apparent_bolometric_magnitude
        peak_times[index] = curve.peak_time_days
        peak_magnitudes[index] = np.min(curve.apparent_bolometric_magnitude)
        for name in filter_magnitudes:
            values = curve.bands[name].apparent_magnitude
            filter_magnitudes[name][index] = values
            peak_filter_magnitudes[name][index] = np.min(values)
    nominal = SupernovaLightCurveModel(
        scenario.progenitor, scenario.overrides
    ).evaluate(time)
    return MonteCarloEnvelope(
        observer_time_days=time,
        luminosity_percentiles_w=np.percentile(luminosities, percentiles, axis=0),
        apparent_magnitude_percentiles=np.percentile(magnitudes, percentiles, axis=0),
        peak_time_percentiles_days=np.percentile(peak_times, percentiles),
        peak_magnitude_percentiles=np.percentile(peak_magnitudes, percentiles),
        filter_magnitude_percentiles={
            name: np.percentile(values, percentiles, axis=0)
            for name, values in filter_magnitudes.items()
        },
        peak_filter_magnitude_percentiles={
            name: np.percentile(values, percentiles)
            for name, values in peak_filter_magnitudes.items()
        },
        nominal_luminosity_w=nominal.luminosity_w,
        nominal_filter_magnitudes={
            name: nominal.bands[name].apparent_magnitude for name in filter_magnitudes
        },
        percentiles=percentiles,
    )


def scenarios_for_comparison(
    mode: str,
    scenario_names: tuple[str, ...] | list[str] = (),
) -> tuple[Scenario, ...]:
    if mode not in {"historical", "hypothetical", "all", "scenario-by-scenario"}:
        raise ValueError(
            "comparison mode must be historical, hypothetical, all, or scenario-by-scenario"
        )
    if mode == "scenario-by-scenario":
        if not scenario_names:
            raise ValueError("scenario-by-scenario comparison requires scenario names")
        names = tuple(scenario_names)
    else:
        names = available_scenarios(mode)
    return tuple(load_scenario(name) for name in names)
