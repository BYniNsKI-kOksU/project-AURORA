"""Dependency-free SVG plots for light curves, comparisons and Monte Carlo."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .light_curve import LightCurve, SupernovaLightCurveModel
from .scenarios import MonteCarloEnvelope, Scenario, scenarios_for_comparison


_COLORS = ("#67d8ff", "#ffc857", "#ff6b8a", "#9cff95", "#c7a6ff", "#ff9e64", "#e8f1ff")


@dataclass(frozen=True)
class ComparisonSeries:
    key: str
    status: str
    supernova_type: str
    distance_pc: float
    energy_foe: float
    observer_time_days: np.ndarray
    apparent_magnitude: np.ndarray
    absolute_magnitude: np.ndarray
    observed_flux_w_m2: np.ndarray
    temperature_k: np.ndarray
    photospheric_radius_m: np.ndarray


def build_comparison(
    mode: str,
    observer_time_days: np.ndarray,
    *,
    filter_name: str = "V",
    scenario_names: tuple[str, ...] | list[str] = (),
) -> tuple[ComparisonSeries, ...]:
    series = []
    for scenario in scenarios_for_comparison(mode, scenario_names):
        model = SupernovaLightCurveModel(scenario.progenitor, scenario.overrides)
        curve = model.evaluate(observer_time_days)
        band = curve.bands[filter_name]
        series.append(
            ComparisonSeries(
                key=scenario.key,
                status=scenario.status,
                supernova_type=scenario.progenitor.supernova_type,
                distance_pc=scenario.progenitor.distance.parsecs,
                energy_foe=model.explosion.energy_foe,
                observer_time_days=curve.observer_time_days,
                apparent_magnitude=band.apparent_magnitude,
                absolute_magnitude=band.absolute_magnitude,
                observed_flux_w_m2=band.observed_flux_w_m2,
                temperature_k=curve.observed_temperature_k,
                photospheric_radius_m=curve.photospheric_radius_m,
            )
        )
    return tuple(series)


def write_comparison_data(
    series: Iterable[ComparisonSeries], output: Path, *, filter_name: str
) -> tuple[Path, Path]:
    series = tuple(series)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    json_path = output.with_suffix(".json")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        header = ["scenario", "observer_time_days", "filter", "apparent_magnitude", "absolute_magnitude", "observed_flux_w_m2", "temperature_k", "photospheric_radius_m"]
        handle.write(",".join(header) + "\n")
        for item in series:
            for values in zip(item.observer_time_days, item.apparent_magnitude, item.absolute_magnitude, item.observed_flux_w_m2, item.temperature_k, item.photospheric_radius_m):
                handle.write(
                    f"{item.key},{values[0]:.10g},{filter_name},"
                    + ",".join(f"{value:.10g}" for value in values[1:])
                    + "\n"
                )
    metadata = {
        "filter": filter_name,
        "series": [
            {
                "scenario": item.key,
                "status": item.status,
                "supernova_type": item.supernova_type,
                "distance_pc": item.distance_pc,
                "energy_foe": item.energy_foe,
            }
            for item in series
        ],
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _polyline(x: np.ndarray, y: np.ndarray, color: str, *, dash: str = "") -> str:
    points = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in zip(x, y))
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{dashed}/>'


def _magnitude_axes(time: np.ndarray, magnitudes: list[np.ndarray], width: int, height: int):
    left, right, top, bottom = 92.0, width - 34.0, 72.0, height - 86.0
    t_min, t_max = float(np.min(time)), float(np.max(time))
    finite = np.concatenate([values[np.isfinite(values)] for values in magnitudes])
    m_min, m_max = float(np.min(finite)), float(np.max(finite))
    padding = max(0.4, 0.06 * (m_max - m_min))
    m_min, m_max = m_min - padding, m_max + padding
    x = lambda values: left + (np.asarray(values) - t_min) / max(t_max - t_min, 1e-9) * (right - left)
    # Smaller magnitude is brighter and therefore appears higher.
    y = lambda values: top + (np.asarray(values) - m_min) / max(m_max - m_min, 1e-9) * (bottom - top)
    return left, right, top, bottom, x, y, m_min, m_max


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#07111d"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#e8f1ff}.axis{stroke:#8292a6;stroke-width:1}.grid{stroke:#294057;stroke-width:1;opacity:.55}.muted{fill:#9eacbc;font-size:13px}.label{font-size:14px}.title{font-size:23px;font-weight:650}</style>',
        f'<text x="32" y="38" class="title">{escape(title)}</text>',
    ]


def _draw_axes(parts: list[str], time: np.ndarray, left: float, right: float, top: float, bottom: float, x, y, m_min: float, m_max: float) -> None:
    parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="#0b1827" stroke="#40556c"/>')
    for value in np.linspace(float(time[0]), float(time[-1]), 6):
        xx = float(x(value)); parts.append(f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{bottom}" class="grid"/><text x="{xx}" y="{bottom+24}" text-anchor="middle" class="muted">{value:.0f}</text>')
    for value in np.linspace(m_min, m_max, 6):
        yy = float(y(value)); parts.append(f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" class="grid"/><text x="{left-12}" y="{yy+4}" text-anchor="end" class="muted">{value:.1f}</text>')
    parts.append(f'<text x="{(left+right)/2}" y="{bottom+57}" text-anchor="middle" class="label">Time since explosion [observer days]</text>')
    parts.append(f'<text x="25" y="{(top+bottom)/2}" transform="rotate(-90 25 {(top+bottom)/2})" text-anchor="middle" class="label">Apparent AB magnitude (brighter upward)</text>')


def write_light_curve_svg(curve: LightCurve, scenario: Scenario, output: Path) -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    names = ("UV", "U", "B", "V", "R", "I", "IR")
    values = [curve.bands[name].apparent_magnitude for name in names]
    width, height = 1200, 720
    left, right, top, bottom, x, y, m_min, m_max = _magnitude_axes(curve.observer_time_days, values, width, height)
    parts = _svg_header(width, height, f"{scenario.key}: multi-band observed light curves")
    _draw_axes(parts, curve.observer_time_days, left, right, top, bottom, x, y, m_min, m_max)
    for index, name in enumerate(names):
        color = _COLORS[index]
        parts.append(_polyline(x(curve.observer_time_days), y(values[index]), color))
        parts.append(f'<text x="{left+index*72}" y="{height-18}" fill="{color}" class="label">{name}</text>')
    intrinsic_v = curve.bands["V"].intrinsic_apparent_magnitude
    parts.append(_polyline(x(curve.observer_time_days), y(intrinsic_v), _COLORS[3], dash="6 5"))
    observed = scenario.historical_metadata.get("observed_peak_visual_magnitude")
    if observed is not None:
        peak_x = float(x(curve.peak_visual_time_days)); peak_y = float(y(float(observed)))
        parts.append(f'<circle cx="{peak_x}" cy="{peak_y}" r="5" fill="none" stroke="#ffffff" stroke-width="2"/><text x="{peak_x+10}" y="{peak_y-9}" class="label">historical V ≈ {float(observed):.1f}</text>')
    parts.append(f'<text x="{right-360}" y="45" class="muted">solid: extincted model · dashed V: without dust · circle: observation</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def write_comparison_svg(series: Iterable[ComparisonSeries], output: Path, *, filter_name: str = "V") -> Path:
    series = tuple(series)
    if not series: raise ValueError("comparison requires at least one scenario")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 1250
    time = series[0].observer_time_days
    parts = _svg_header(width, height, f"AURORA supernova comparison — filter {filter_name}")
    left, right = 92.0, width - 34.0
    x = lambda values: left + np.asarray(values) / max(float(time[-1]), 1e-9) * (right - left)
    metrics = (
        ("apparent_magnitude", f"Apparent {filter_name} [mag]", False, False),
        ("absolute_magnitude", f"Absolute {filter_name} [mag]", False, False),
        ("observed_flux_w_m2", "log10 observed flux [W m^-2]", True, True),
        ("temperature_k", "Observed temperature [K]", False, True),
        ("photospheric_radius_m", "log10 photospheric radius [m]", True, True),
    )
    panel_height, gap, first_top = 155.0, 43.0, 62.0
    for panel, (attribute, label, logarithmic, high_is_up) in enumerate(metrics):
        top = first_top + panel * (panel_height + gap)
        bottom = top + panel_height
        raw_values = [np.asarray(getattr(item, attribute), dtype=np.float64) for item in series]
        values = [np.log10(np.maximum(value, np.finfo(float).tiny)) if logarithmic else value for value in raw_values]
        finite = np.concatenate([value[np.isfinite(value)] for value in values])
        low, high = float(np.min(finite)), float(np.max(finite))
        padding = max(0.05, 0.05 * max(high - low, 1.0))
        low, high = low - padding, high + padding
        if high_is_up:
            y = lambda value, lo=low, hi=high, a=top, b=bottom: b - (np.asarray(value) - lo) / (hi - lo) * (b - a)
        else:
            y = lambda value, lo=low, hi=high, a=top, b=bottom: a + (np.asarray(value) - lo) / (hi - lo) * (b - a)
        parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{panel_height}" fill="#0b1827" stroke="#40556c"/>')
        for tick in np.linspace(low, high, 4):
            yy = float(y(tick)); parts.append(f'<line x1="{left}" y1="{yy}" x2="{right}" y2="{yy}" class="grid"/><text x="{left-10}" y="{yy+4}" text-anchor="end" class="muted">{tick:.2g}</text>')
        for index, item in enumerate(series):
            color = _COLORS[index % len(_COLORS)]
            parts.append(_polyline(x(item.observer_time_days), y(values[index]), color))
        parts.append(f'<text x="25" y="{(top+bottom)/2}" transform="rotate(-90 25 {(top+bottom)/2})" text-anchor="middle" class="label">{escape(label)}</text>')
        if panel == len(metrics) - 1:
            for tick in np.linspace(0.0, float(time[-1]), 6):
                xx = float(x(tick)); parts.append(f'<text x="{xx}" y="{bottom+23}" text-anchor="middle" class="muted">{tick:.0f}</text>')
            parts.append(f'<text x="{(left+right)/2}" y="{bottom+49}" text-anchor="middle" class="label">Time since explosion [observer days]</text>')
    legend_top = 1100.0
    for index, item in enumerate(series):
        color = _COLORS[index % len(_COLORS)]
        lx = left + (index % 3) * 360; ly = legend_top + (index // 3) * 24
        parts.append(
            f'<text x="{lx}" y="{ly}" fill="{color}" class="muted">'
            f'{escape(item.key)} · {item.supernova_type} · {item.distance_pc:g} pc · {item.energy_foe:.2g} foe</text>'
        )
    parts.append(f'<text x="{left}" y="{height-18}" class="muted">All panels: semi-analytic model · filter {filter_name}; status and catalog provenance are stored in JSON.</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def write_monte_carlo_svg(envelope: MonteCarloEnvelope, output: Path, *, scenario_key: str, filter_name: str = "V") -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    percentile_values = envelope.filter_magnitude_percentiles[filter_name]
    nominal = envelope.nominal_filter_magnitudes[filter_name]
    all_values = [percentile_values[0], percentile_values[-1], nominal]
    width, height = 1200, 720
    time = envelope.observer_time_days
    left, right, top, bottom, x, y, m_min, m_max = _magnitude_axes(time, all_values, width, height)
    parts = _svg_header(width, height, f"{scenario_key}: Monte Carlo uncertainty — filter {filter_name}")
    _draw_axes(parts, time, left, right, top, bottom, x, y, m_min, m_max)
    upper = list(zip(x(time), y(percentile_values[0])))
    lower = list(zip(x(time)[::-1], y(percentile_values[-1])[::-1]))
    polygon = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in upper + lower)
    parts.append(f'<polygon points="{polygon}" fill="#67d8ff" opacity="0.20"/>')
    parts.append(_polyline(x(time), y(percentile_values[len(percentile_values)//2]), "#67d8ff", dash="5 4"))
    parts.append(_polyline(x(time), y(nominal), "#ffffff"))
    p = envelope.percentiles
    parts.append(f'<text x="{right-400}" y="45" class="muted">band: P{p[0]:g}–P{p[-1]:g} · dashed: median · white: nominal model</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output
