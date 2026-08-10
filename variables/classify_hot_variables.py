"""Resolve Gaia DR3 ``BE|GCAS|SDOR|WR`` candidates into subclasses.

This module deliberately produces *candidate* classifications.  Its
multi-feature rules use Gaia DR3 astrophysical parameters and variability
statistics; they do not replace a literature or spectroscopic classification.
The scoring code is independent of FITS input/output so that the rules can be
reviewed and recalibrated without changing catalogue handling.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.table import Column, Table
from astropy.units import UnitsWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "assets" / "lbv.fits"
CLASS_NAMES = ("BE", "GCAS", "SDOR", "WR")
OUTPUT_COLUMNS = (
    "bp_rp_intrinsic",
    "reddening_quality",
    "extinction_flags",
    "resolved_class",
    "classification_score",
    "classification_margin",
    "classification_method",
    "score_be",
    "score_gcas",
    "score_sdor",
    "score_wr",
    "classification_flags",
)
LITERATURE_COLUMNS = (
    "subclass",
    "resolved_class",
    "literature_class",
    "spectral_class",
    "object_type",
)
SUMMARY_COLUMNS = (
    "teff_gspphot",
    "logg_gspphot",
    "lum_flame",
    "radius_gspphot",
    "mass_flame",
    "vbroad",
    "trimmed_range_mag_g_fov",
    "std_dev_mag_g_fov",
)


@dataclass(frozen=True)
class ClassificationThresholds:
    """All tunable boundaries for the Gaia candidate classifier.

    These broad boundaries are intentionally conservative.  They describe
    overlapping candidate profiles, not hard spectroscopic class definitions.
    Values use Gaia catalogue units: K, dex, solar luminosities/radii/masses,
    km/s, and magnitudes.
    """

    # B-star temperature interval used as evidence for BE/GCAS candidates.
    b_teff_min: float = 10_000.0
    b_teff_max: float = 30_000.0
    # A temperature above this is supporting (never sufficient) WR evidence.
    wr_teff_hot: float = 35_000.0
    # A second, stronger WR temperature contribution starts here.
    wr_teff_very_hot: float = 50_000.0
    # Hot-supergiant temperatures that are compatible with an SDOR candidate.
    sdor_teff_min: float = 8_000.0
    sdor_teff_max: float = 35_000.0

    # Luminosity boundaries in L_sun; SDOR requires corroborating giant traits.
    moderate_lum_min: float = 20.0
    moderate_lum_max: float = 10_000.0
    high_luminosity: float = 10_000.0
    extreme_luminosity: float = 100_000.0
    # Radius boundaries in R_sun separating compact/hot and extended profiles.
    compact_radius_min: float = 1.0
    compact_radius_max: float = 20.0
    extended_radius: float = 25.0
    very_extended_radius: float = 50.0
    # Surface gravity boundaries (log10(cm s^-2)).
    main_sequence_logg: float = 3.2
    low_logg: float = 3.0
    very_low_logg: float = 2.0
    # Mass boundaries in M_sun.
    be_mass_min: float = 3.0
    massive_star_mass: float = 10.0
    sdor_mass: float = 20.0
    very_massive_star_mass: float = 40.0

    # Extinction-corrected BP-RP below this is supporting hot-star evidence.
    blue_intrinsic_bp_rp: float = 0.20
    # Raw BP-RP is accepted only as much weaker blue-star evidence.
    raw_blue_bp_rp: float = -0.10
    # Reliability limits for Gaia E(BP-RP) and A_G interval estimates.
    reddening_absolute_uncertainty_floor: float = 0.15
    reddening_relative_uncertainty_limit: float = 0.75
    ag_absolute_uncertainty_floor: float = 0.30
    ag_relative_uncertainty_limit: float = 0.75
    # Values above these boundaries are described as strong extinction.
    strong_reddening: float = 0.50
    strong_ag: float = 1.00
    # Astrometric luminosity is used only for positive >=5-sigma parallaxes.
    minimum_parallax_snr: float = 5.0
    luminous_absolute_g: float = -4.0
    extreme_absolute_g: float = -7.0
    moderate_absolute_g: float = 2.0
    # Gaia vbroad boundaries; rapid rotation supports BE more strongly than WR.
    elevated_vbroad: float = 50.0
    rapid_vbroad: float = 100.0
    # Radial-velocity range is weak, non-exclusive supporting evidence only.
    high_rv_amplitude: float = 50.0

    # Variability ranges distinguish quiet BE/WR from irregular GCAS/SDOR.
    quiet_trimmed_range: float = 0.15
    active_trimmed_range: float = 0.50
    large_trimmed_range: float = 1.00
    quiet_std_dev: float = 0.05
    active_std_dev: float = 0.15
    irregular_abs_skewness: float = 1.0
    extreme_abs_skewness: float = 2.0
    irregular_kurtosis: float = 3.0
    extreme_kurtosis: float = 6.0

    # Resolution gates: weak or close scores remain UNKNOWN.
    minimum_winning_score: float = 4.0
    minimum_margin: float = 1.0
    minimum_confidence: float = 0.48
    strong_score_scale: float = 9.0
    minimum_evidence_features: int = 3
    minimum_gcas_variability_points: float = 2.0
    minimum_sdor_physical_groups: int = 2
    minimum_wr_physical_groups: int = 2
    # The Gaia parent-class score modulates confidence but never selects a type.
    low_parent_class_score: float = 0.35

    # Uncertain Gaia intervals reduce, rather than discard, their contribution.
    good_fractional_uncertainty: float = 0.25
    poor_fractional_uncertainty: float = 1.00
    good_logg_uncertainty: float = 0.30
    poor_logg_uncertainty: float = 1.00
    minimum_uncertainty_weight: float = 0.35


DEFAULT_THRESHOLDS = ClassificationThresholds()


@dataclass(frozen=True)
class ObjectClassification:
    resolved_class: str
    classification_score: float
    classification_margin: float
    classification_method: str
    score_be: float
    score_gcas: float
    score_sdor: float
    score_wr: float
    classification_flags: str


@dataclass(frozen=True)
class ReddeningAssessment:
    bp_rp_intrinsic: float
    quality: str
    flags: tuple[str, ...]
    ebpminrp_uncertainty: float
    ag_reliable: bool


def _normalise_label(value: object) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    text = str(value).strip().upper()
    if text in {"", "--", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return ""
    return re.sub(r"[^A-Z0-9]+", "", text)


def _label_class(value: object, *, spectral: bool = False) -> str | None:
    """Return only an unambiguous class encoded in a text value."""
    raw = "" if value is None or np.ma.is_masked(value) else str(value)
    normal = _normalise_label(raw)
    if not normal:
        return None

    if (
        normal in {"WR", "WOLF RAYET".replace(" ", "")}
        or re.fullmatch(r"W[CNOR][A-Z]*\d*[A-Z]*", normal)
    ):
        return "WR"
    if normal in {"LBV", "SDOR", "SDORADUS", "LUMINOUSBLUEVARIABLE"}:
        return "SDOR"
    if normal in {"GCAS", "GAMMACAS", "GAMMACASSIOPEIAE"}:
        return "GCAS"
    if normal in {
        "BE",
        "BEMISSION",
        "BEMISSIONLINE",
        "BTYPEEMISSIONLINE",
        "EMISSIONLINEB",
    }:
        return "BE"

    if spectral:
        lower = raw.casefold()
        if re.search(r"\b(lbv|s\s*dor(?:adus)?)\b", lower):
            return "SDOR"
        if re.search(r"\b(?:gamma|γ)\s*cas(?:siopeiae)?\b", lower):
            return "GCAS"
        if re.search(r"\bb\s*\[\s*e\s*\]", lower):
            return "BE"
        if re.search(r"\bbe(?:\d|\b)", lower):
            return "BE"
    return None


def _float_value(row, name: str) -> float:
    if name not in row.colnames:
        return np.nan
    value = row[name]
    if value is None or np.ma.is_masked(value):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _text_value(row, name: str) -> str:
    if name not in row.colnames:
        return ""
    value = row[name]
    if value is None or np.ma.is_masked(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "--", "nan", "none", "null"} else text


def _interval_uncertainty(row, name: str) -> tuple[float, bool]:
    """Return interval half-width and whether both valid bounds exist."""
    lower = _float_value(row, f"{name}_lower")
    upper = _float_value(row, f"{name}_upper")
    valid = np.isfinite(lower) and np.isfinite(upper) and upper >= lower
    return (0.5 * (upper - lower), True) if valid else (np.nan, False)


def assess_reddening(
    row,
    config: ClassificationThresholds = DEFAULT_THRESHOLDS,
) -> ReddeningAssessment:
    """Assess Gaia reddening estimates without assuming zero extinction."""
    bp_rp = _float_value(row, "bp_rp")
    reddening = _float_value(row, "ebpminrp_gspphot")
    ag = _float_value(row, "ag_gspphot")
    flags: list[str] = []
    reddening_uncertainty = np.nan
    invalidate_intrinsic = True
    if not np.isfinite(reddening):
        quality = "missing"
        flags.append("missing_reddening")
    elif reddening < 0.0:
        quality = "invalid"
        flags.append("invalid_reddening")
    else:
        invalidate_intrinsic = False
        quality = "good"
        (
            reddening_uncertainty,
            valid_reddening_interval,
        ) = _interval_uncertainty(
            row,
            "ebpminrp_gspphot",
        )
        if not valid_reddening_interval:
            quality = "uncertain"
            flags.extend(
                ("uncertain_reddening", "missing_reddening_interval")
            )
        else:
            reddening_limit = max(
                config.reddening_absolute_uncertainty_floor,
                config.reddening_relative_uncertainty_limit * reddening,
            )
            if reddening_uncertainty > reddening_limit:
                quality = "uncertain"
                invalidate_intrinsic = True
                flags.append("uncertain_reddening")

    ag_reliable = False
    if np.isfinite(ag):
        if ag < 0.0:
            flags.append("invalid_extinction")
            if quality == "good":
                quality = "uncertain"
                flags.append("uncertain_reddening")
        else:
            ag_uncertainty, valid_ag_interval = _interval_uncertainty(
                row, "ag_gspphot"
            )
            if valid_ag_interval:
                ag_limit = max(
                    config.ag_absolute_uncertainty_floor,
                    config.ag_relative_uncertainty_limit * ag,
                )
                ag_reliable = ag_uncertainty <= ag_limit
                if not ag_reliable:
                    if quality == "good":
                        quality = "uncertain"
                    invalidate_intrinsic = True
                    flags.extend(("uncertain_reddening", "uncertain_extinction"))
            else:
                flags.append("missing_extinction_interval")
    if reddening >= config.strong_reddening or (
        np.isfinite(ag) and ag >= config.strong_ag
    ):
        flags.append("strong_extinction")

    intrinsic = (
        bp_rp - reddening
        if np.isfinite(bp_rp)
        and np.isfinite(reddening)
        and reddening >= 0.0
        else np.nan
    )
    if not np.isfinite(bp_rp):
        flags.append("missing_bp_rp")
    if invalidate_intrinsic:
        intrinsic = np.nan
    return ReddeningAssessment(
        float(intrinsic) if np.isfinite(intrinsic) else np.nan,
        quality,
        tuple(dict.fromkeys(flags)),
        (
            float(reddening_uncertainty)
            if np.isfinite(reddening_uncertainty)
            else np.nan
        ),
        ag_reliable,
    )


def _uncertainty_weight(
    row,
    name: str,
    value: float,
    config: ClassificationThresholds,
) -> tuple[float, bool]:
    lower = _float_value(row, f"{name}_lower")
    upper = _float_value(row, f"{name}_upper")
    if not (np.isfinite(lower) and np.isfinite(upper) and upper >= lower):
        error = _float_value(row, f"{name}_error")
        if not np.isfinite(error) or error < 0.0:
            return 1.0, False
        half_width = error
    else:
        half_width = 0.5 * (upper - lower)

    if name == "logg_gspphot":
        good = config.good_logg_uncertainty
        poor = config.poor_logg_uncertainty
        uncertainty = half_width
    else:
        good = config.good_fractional_uncertainty
        poor = config.poor_fractional_uncertainty
        uncertainty = half_width / max(abs(value), 1.0e-12)
    fraction = np.clip((uncertainty - good) / max(poor - good, 1.0e-12), 0.0, 1.0)
    weight = 1.0 - fraction * (1.0 - config.minimum_uncertainty_weight)
    return float(weight), bool(uncertainty >= poor)


def _add(scores: dict[str, float], class_name: str, points: float, weight=1.0):
    scores[class_name] = max(0.0, scores[class_name] + points * weight)


def _score_object(row, config: ClassificationThresholds):
    scores = {name: 0.0 for name in CLASS_NAMES}
    flags: list[str] = []
    available_features = 0
    uncertain = False
    sdor_groups: set[str] = set()
    wr_groups: set[str] = set()
    gcas_variability_points = 0.0
    reddening_assessment = assess_reddening(row, config)
    flags.extend(reddening_assessment.flags)

    spectral_text = _text_value(row, "spectraltype_esphs")
    spectral_normal = _normalise_label(spectral_text)
    if not spectral_text:
        flags.append("missing_spectral_type")
    else:
        available_features += 1
        if spectral_normal == "B":
            _add(scores, "BE", 1.5)
            _add(scores, "GCAS", 1.2)
            _add(scores, "SDOR", 0.4)
        elif spectral_normal == "O":
            _add(scores, "WR", 1.2)
            _add(scores, "SDOR", 0.4)
            wr_groups.add("hot_spectral_profile")

    teff = _float_value(row, "teff_gspphot")
    if not np.isfinite(teff):
        flags.append("missing_teff")
    else:
        weight, is_uncertain = _uncertainty_weight(
            row, "teff_gspphot", teff, config
        )
        uncertain |= is_uncertain
        if not is_uncertain:
            available_features += 1
            if config.b_teff_min <= teff <= config.b_teff_max:
                _add(scores, "BE", 2.0, weight)
                _add(scores, "GCAS", 1.7, weight)
            if config.sdor_teff_min <= teff <= config.sdor_teff_max:
                _add(scores, "SDOR", 0.7, weight)
            if teff >= config.wr_teff_hot:
                _add(scores, "WR", 2.0, weight)
                wr_groups.add("very_hot")
                _add(scores, "BE", -0.8, weight)
                _add(scores, "GCAS", -0.5, weight)
            if teff >= config.wr_teff_very_hot:
                _add(scores, "WR", 1.0, weight)

    logg = _float_value(row, "logg_gspphot")
    if np.isfinite(logg):
        available_features += 1
        weight, is_uncertain = _uncertainty_weight(
            row, "logg_gspphot", logg, config
        )
        uncertain |= is_uncertain
        if logg >= config.main_sequence_logg:
            _add(scores, "BE", 0.8, weight)
            _add(scores, "GCAS", 0.6, weight)
        if logg <= config.low_logg:
            _add(scores, "SDOR", 2.0, weight)
            sdor_groups.add("low_gravity")
        if logg <= config.very_low_logg:
            _add(scores, "SDOR", 1.0, weight)
    else:
        flags.append("missing_logg")

    luminosity = _float_value(row, "lum_flame")
    if not np.isfinite(luminosity):
        flags.append("missing_luminosity")
    else:
        available_features += 1
        weight, is_uncertain = _uncertainty_weight(
            row, "lum_flame", luminosity, config
        )
        uncertain |= is_uncertain
        if config.moderate_lum_min <= luminosity < config.moderate_lum_max:
            _add(scores, "BE", 1.0, weight)
            _add(scores, "GCAS", 0.8, weight)
        if luminosity >= config.high_luminosity:
            _add(scores, "SDOR", 1.3, weight)
            _add(scores, "WR", 1.0, weight)
            sdor_groups.add("high_luminosity")
            wr_groups.add("high_luminosity")
        if luminosity >= config.extreme_luminosity:
            _add(scores, "SDOR", 2.0, weight)
            _add(scores, "WR", 1.0, weight)
        elif luminosity < config.moderate_lum_min:
            _add(scores, "SDOR", -1.0, weight)
            _add(scores, "WR", -0.8, weight)

    parallax = _float_value(row, "parallax")
    parallax_error = _float_value(row, "parallax_error")
    apparent_g = _float_value(row, "phot_g_mean_mag")
    ag = _float_value(row, "ag_gspphot")
    if (
        np.isfinite(parallax)
        and parallax > 0.0
        and np.isfinite(parallax_error)
        and parallax_error > 0.0
        and parallax / parallax_error >= config.minimum_parallax_snr
        and np.isfinite(apparent_g)
    ):
        available_features += 1
        distance_pc = 1_000.0 / parallax
        absolute_g = apparent_g - 5.0 * np.log10(distance_pc / 10.0)
        if reddening_assessment.ag_reliable:
            absolute_g -= max(ag, 0.0)
        if absolute_g <= config.luminous_absolute_g:
            _add(scores, "SDOR", 1.0)
            _add(scores, "WR", 0.7)
            sdor_groups.add("astrometric_luminosity")
            wr_groups.add("astrometric_luminosity")
        elif absolute_g <= config.moderate_absolute_g:
            _add(scores, "BE", 0.5)
            _add(scores, "GCAS", 0.4)
        if absolute_g <= config.extreme_absolute_g:
            _add(scores, "SDOR", 1.0)
            _add(scores, "WR", 0.5)

    radius = _float_value(row, "radius_gspphot")
    if not np.isfinite(radius):
        flags.append("missing_radius")
    else:
        available_features += 1
        weight, is_uncertain = _uncertainty_weight(
            row, "radius_gspphot", radius, config
        )
        uncertain |= is_uncertain
        if config.compact_radius_min <= radius <= config.compact_radius_max:
            _add(scores, "BE", 0.7, weight)
            _add(scores, "GCAS", 0.6, weight)
            _add(scores, "WR", 1.0, weight)
            wr_groups.add("compact_radius")
        if radius >= config.extended_radius:
            _add(scores, "SDOR", 2.5, weight)
            _add(scores, "WR", -1.0, weight)
            sdor_groups.add("extended_radius")
        if radius >= config.very_extended_radius:
            _add(scores, "SDOR", 1.0, weight)

    mass = _float_value(row, "mass_flame")
    if np.isfinite(mass):
        available_features += 1
        weight, is_uncertain = _uncertainty_weight(
            row, "mass_flame", mass, config
        )
        uncertain |= is_uncertain
        if config.be_mass_min <= mass < config.sdor_mass:
            _add(scores, "BE", 0.7, weight)
            _add(scores, "GCAS", 0.5, weight)
        if mass >= config.massive_star_mass:
            _add(scores, "WR", 1.0, weight)
            wr_groups.add("massive")
        if mass >= config.sdor_mass:
            _add(scores, "SDOR", 2.0, weight)
            sdor_groups.add("massive")
        if mass >= config.very_massive_star_mass:
            _add(scores, "SDOR", 1.0, weight)

    intrinsic_colour = reddening_assessment.bp_rp_intrinsic
    if reddening_assessment.quality == "good" and np.isfinite(intrinsic_colour):
        available_features += 1
        if intrinsic_colour <= config.blue_intrinsic_bp_rp:
            _add(scores, "BE", 0.8)
            _add(scores, "GCAS", 0.7)
            _add(scores, "SDOR", 0.3)
            _add(scores, "WR", 0.5)
    else:
        # An observed colour can be heavily reddened.  Only an unusually blue
        # raw value supplies a small amount of supporting, never excluding,
        # evidence when a reliable correction is unavailable.
        bp_rp = _float_value(row, "bp_rp")
        if np.isfinite(bp_rp) and bp_rp <= config.raw_blue_bp_rp:
            _add(scores, "BE", 0.2)
            _add(scores, "GCAS", 0.2)
            _add(scores, "SDOR", 0.1)
            _add(scores, "WR", 0.1)

    vbroad = _float_value(row, "vbroad")
    if np.isfinite(vbroad):
        available_features += 1
        weight, is_uncertain = _uncertainty_weight(row, "vbroad", vbroad, config)
        uncertain |= is_uncertain
        if vbroad >= config.elevated_vbroad:
            _add(scores, "BE", 1.5, weight)
            _add(scores, "GCAS", 0.7, weight)
        if vbroad >= config.rapid_vbroad:
            _add(scores, "BE", 1.2, weight)
            _add(scores, "GCAS", 0.5, weight)
            _add(scores, "WR", 0.8, weight)
            wr_groups.add("broad_lines")

    rv_amplitude = _float_value(row, "rv_amplitude_robust")
    if np.isfinite(rv_amplitude):
        available_features += 1
        if rv_amplitude >= config.high_rv_amplitude:
            _add(scores, "GCAS", 0.5)
            _add(scores, "WR", 0.3)

    trimmed_range = _float_value(row, "trimmed_range_mag_g_fov")
    if np.isfinite(trimmed_range):
        available_features += 1
        if trimmed_range <= config.quiet_trimmed_range:
            _add(scores, "BE", 1.5)
            _add(scores, "WR", 0.7)
        elif trimmed_range < config.active_trimmed_range:
            _add(scores, "GCAS", 1.5)
            _add(scores, "SDOR", 0.5)
            gcas_variability_points += 1.5
        else:
            _add(scores, "GCAS", 1.5)
            _add(scores, "SDOR", 1.5)
            gcas_variability_points += 1.5
        if trimmed_range >= config.large_trimmed_range:
            _add(scores, "GCAS", 0.5)
            _add(scores, "SDOR", 1.5)
            gcas_variability_points += 0.5

    std_dev = _float_value(row, "std_dev_mag_g_fov")
    if np.isfinite(std_dev):
        available_features += 1
        if std_dev <= config.quiet_std_dev:
            _add(scores, "BE", 1.0)
            _add(scores, "WR", 0.5)
        elif std_dev < config.active_std_dev:
            _add(scores, "GCAS", 1.0)
            gcas_variability_points += 1.0
        else:
            _add(scores, "GCAS", 1.0)
            _add(scores, "SDOR", 1.0)
            gcas_variability_points += 1.0

    skewness = _float_value(row, "skewness_mag_g_fov")
    if np.isfinite(skewness):
        available_features += 1
        absolute_skewness = abs(skewness)
        if absolute_skewness >= config.irregular_abs_skewness:
            _add(scores, "GCAS", 1.2)
            gcas_variability_points += 1.2
        if absolute_skewness >= config.extreme_abs_skewness:
            _add(scores, "GCAS", 0.7)
            gcas_variability_points += 0.7

    kurtosis = _float_value(row, "kurtosis_mag_g_fov")
    if np.isfinite(kurtosis):
        available_features += 1
        if kurtosis >= config.irregular_kurtosis:
            _add(scores, "GCAS", 1.2)
            gcas_variability_points += 1.2
        if kurtosis >= config.extreme_kurtosis:
            _add(scores, "GCAS", 0.7)
            gcas_variability_points += 0.7

    if uncertain:
        flags.append("large_parameter_uncertainty")
    return (
        scores,
        flags,
        available_features,
        gcas_variability_points,
        sdor_groups,
        wr_groups,
    )


def _winner_values(scores: dict[str, float]):
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    winner, top = ordered[0]
    second = ordered[1][1]
    return winner, float(top), float(second), float(top - second)


def _heuristic_confidence(
    top: float,
    margin: float,
    parent_score: float,
    config: ClassificationThresholds,
) -> float:
    strength = np.clip(top / config.strong_score_scale, 0.0, 1.0)
    separation = np.clip(margin / max(top, 1.0e-12), 0.0, 1.0)
    confidence = 0.65 * strength + 0.35 * separation
    if np.isfinite(parent_score):
        confidence *= 0.85 + 0.15 * np.clip(parent_score, 0.0, 1.0)
    return float(np.clip(confidence, 0.0, 1.0))


def classify_object(
    row,
    config: ClassificationThresholds = DEFAULT_THRESHOLDS,
) -> ObjectClassification:
    """Classify one table row, respecting literature and spectral precedence."""
    scores, flags, evidence_count, gcas_var, sdor_groups, wr_groups = _score_object(
        row, config
    )
    parent_score = _float_value(row, "best_class_score")
    if np.isfinite(parent_score) and parent_score < config.low_parent_class_score:
        flags.append("low_parent_class_score")

    literature_class = None
    for column_name in LITERATURE_COLUMNS:
        # An existing output column is only input evidence if it predates this run.
        if column_name in row.colnames:
            literature_class = _label_class(_text_value(row, column_name))
            if literature_class:
                break
    if literature_class:
        scores[literature_class] += 12.0
        _, top, second, margin = _winner_values(scores)
        return ObjectClassification(
            literature_class,
            1.0,
            margin,
            "literature_label",
            scores["BE"],
            scores["GCAS"],
            scores["SDOR"],
            scores["WR"],
            ";".join(dict.fromkeys(flags)),
        )

    spectral_text = _text_value(row, "spectraltype_esphs")
    spectral_class = _label_class(spectral_text, spectral=True)
    if spectral_class:
        # A Be spectrum and GCAS variability can coexist; only strong irregular
        # evidence promotes this ambiguous spectral indication to GCAS.
        resolved = spectral_class
        if spectral_class == "BE" and gcas_var >= config.minimum_gcas_variability_points:
            resolved = "GCAS"
        scores[resolved] += 8.0
        _, top, second, margin = _winner_values(scores)
        return ObjectClassification(
            resolved,
            0.95 if resolved != "BE" else 0.90,
            margin,
            "spectral_type",
            scores["BE"],
            scores["GCAS"],
            scores["SDOR"],
            scores["WR"],
            ";".join(dict.fromkeys(flags)),
        )

    winner, top, _second, margin = _winner_values(scores)
    confidence = _heuristic_confidence(top, margin, parent_score, config)
    heuristic_teff = _float_value(row, "teff_gspphot")
    teff_uncertain = False
    if np.isfinite(heuristic_teff):
        _teff_weight, teff_uncertain = _uncertainty_weight(
            row, "teff_gspphot", heuristic_teff, config
        )
    broad_spectral_type = _normalise_label(spectral_text)
    sdor_hot_profile = (
        (
            not teff_uncertain
            and config.sdor_teff_min <= heuristic_teff <= config.sdor_teff_max
        )
        if np.isfinite(heuristic_teff)
        else broad_spectral_type in {"B", "O"}
    )
    ambiguous_pair = None
    if abs(scores["BE"] - scores["GCAS"]) < config.minimum_margin:
        if max(scores["BE"], scores["GCAS"]) >= config.minimum_winning_score:
            ambiguous_pair = "ambiguous_be_gcas"
    if abs(scores["SDOR"] - scores["WR"]) < config.minimum_margin:
        if max(scores["SDOR"], scores["WR"]) >= config.minimum_winning_score:
            ambiguous_pair = "ambiguous_sdor_wr"

    method = "multi_feature_score"
    resolved = winner
    if evidence_count < config.minimum_evidence_features:
        resolved = "UNKNOWN"
        method = "insufficient_data"
        flags.append("insufficient_data")
    elif winner == "GCAS" and gcas_var < config.minimum_gcas_variability_points:
        resolved = "UNKNOWN"
        method = "ambiguous"
        flags.append("ambiguous_be_gcas")
    elif winner == "SDOR" and len(sdor_groups) < config.minimum_sdor_physical_groups:
        resolved = "UNKNOWN"
        method = "ambiguous"
        flags.append("sdor_without_multiple_physical_indicators")
    elif winner == "SDOR" and not sdor_hot_profile:
        resolved = "UNKNOWN"
        method = "ambiguous"
        flags.append("sdor_without_hot_star_profile")
    elif winner == "WR" and len(wr_groups) < config.minimum_wr_physical_groups:
        resolved = "UNKNOWN"
        method = "ambiguous"
        flags.append("wr_without_multiple_physical_indicators")
    elif (
        top < config.minimum_winning_score
        or margin < config.minimum_margin
        or confidence < config.minimum_confidence
    ):
        resolved = "UNKNOWN"
        method = "ambiguous"
        if ambiguous_pair:
            flags.append(ambiguous_pair)

    if resolved == "UNKNOWN":
        if ambiguous_pair:
            flags.append(ambiguous_pair)
        flags.append("low_confidence")
    return ObjectClassification(
        resolved,
        confidence,
        margin,
        method,
        scores["BE"],
        scores["GCAS"],
        scores["SDOR"],
        scores["WR"],
        ";".join(dict.fromkeys(flags)),
    )


def classify_table(
    table: Table,
    config: ClassificationThresholds = DEFAULT_THRESHOLDS,
) -> Table:
    """Return all input rows and columns followed by classification outputs."""
    output = table.copy(copy_data=True)
    # Classification reads the original rows so an input ``resolved_class``
    # can legitimately act as the requested high-priority catalogue label.
    results = [classify_object(row, config) for row in table]
    reddening = [assess_reddening(row, config) for row in table]
    columns = {
        "bp_rp_intrinsic": Column(
            [item.bp_rp_intrinsic for item in reddening],
            dtype=np.float32,
            name="bp_rp_intrinsic",
        ),
        "reddening_quality": Column(
            [item.quality for item in reddening],
            dtype="U9",
            name="reddening_quality",
        ),
        "extinction_flags": Column(
            [";".join(item.flags) for item in reddening],
            dtype="U128",
            name="extinction_flags",
        ),
        "resolved_class": Column(
            [r.resolved_class for r in results], dtype="U7", name="resolved_class"
        ),
        "classification_score": Column(
            [r.classification_score for r in results],
            dtype=np.float32,
            name="classification_score",
        ),
        "classification_margin": Column(
            [r.classification_margin for r in results],
            dtype=np.float32,
            name="classification_margin",
        ),
        "classification_method": Column(
            [r.classification_method for r in results],
            dtype="U19",
            name="classification_method",
        ),
    }
    for name in ("score_be", "score_gcas", "score_sdor", "score_wr"):
        columns[name] = Column(
            [getattr(r, name) for r in results], dtype=np.float32, name=name
        )
    columns["classification_flags"] = Column(
        [r.classification_flags for r in results],
        dtype="U256",
        name="classification_flags",
    )
    for name in OUTPUT_COLUMNS:
        if name in output.colnames:
            output.replace_column(name, columns[name])
        else:
            output.add_column(columns[name])
    return output


def validate_catalogues(input_table: Table, classified: Table) -> None:
    if len(input_table) != len(classified):
        raise RuntimeError("classification lost or added input records")
    if "source_id" in classified.colnames:
        source_ids = np.asarray(classified["source_id"])
        if len(np.unique(source_ids)) != len(source_ids):
            raise RuntimeError("duplicate source_id values found")
    classes = np.asarray(classified["resolved_class"]).astype(str)
    valid_classes = set(CLASS_NAMES) | {"UNKNOWN"}
    unexpected = set(classes) - valid_classes
    if unexpected:
        raise RuntimeError(f"unexpected resolved classes: {sorted(unexpected)}")
    if sum(np.count_nonzero(classes == name) for name in valid_classes) != len(input_table):
        raise RuntimeError("class totals do not match the number of input records")


def _atomic_write(table: Table, path: Path, overwrite: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.write(temporary, format="fits", overwrite=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_catalogues(classified: Table, output_dir: Path, overwrite=False):
    classes = np.asarray(classified["resolved_class"]).astype(str)
    paths = {
        "BE": Path(output_dir) / "be.fits",
        "GCAS": Path(output_dir) / "gcas.fits",
        "SDOR": Path(output_dir) / "sdor.fits",
        "WR": Path(output_dir) / "wr.fits",
        "UNKNOWN": Path(output_dir) / "be_gcas_sdor_wr_unknown.fits",
        "ALL": Path(output_dir) / "be_gcas_sdor_wr_classified.fits",
    }
    for class_name in (*CLASS_NAMES, "UNKNOWN"):
        _atomic_write(classified[classes == class_name], paths[class_name], overwrite)
    _atomic_write(classified, paths["ALL"], overwrite)
    return paths


def _finite_median(table: Table, name: str) -> str:
    if name not in table.colnames:
        return "n/a"
    column = table[name]
    values = np.asarray(column.filled(np.nan) if hasattr(column, "filled") else column, dtype=float)
    finite = values[np.isfinite(values)]
    return f"{np.median(finite):.6g}" if len(finite) else "n/a"


def print_report(classified: Table, stream=sys.stdout) -> None:
    classes = np.asarray(classified["resolved_class"]).astype(str)
    methods = np.asarray(classified["classification_method"]).astype(str)
    print(f"Input objects: {len(classified)}", file=stream)
    for class_name in (*CLASS_NAMES, "UNKNOWN"):
        print(f"{class_name}: {np.count_nonzero(classes == class_name)}", file=stream)
    print(f"Classified by literature label: {np.count_nonzero(methods == 'literature_label')}", file=stream)
    print(f"Classified by spectral type: {np.count_nonzero(methods == 'spectral_type')}", file=stream)
    print(f"Classified by multi-feature score: {np.count_nonzero(methods == 'multi_feature_score')}", file=stream)
    print(f"Ambiguous: {np.count_nonzero(methods == 'ambiguous')}", file=stream)
    print(f"Insufficient data: {np.count_nonzero(methods == 'insufficient_data')}", file=stream)
    print(f"Total written: {len(classified)}", file=stream)
    if "reddening_quality" in classified.colnames:
        reddening_quality = np.asarray(classified["reddening_quality"]).astype(str)
        print("Reddening quality:", file=stream)
        for quality in ("good", "uncertain", "missing", "invalid"):
            print(
                f"  {quality}: {np.count_nonzero(reddening_quality == quality)}",
                file=stream,
            )
    print("\nMedian physical parameters by resolved class:", file=stream)
    print("class " + " ".join(SUMMARY_COLUMNS), file=stream)
    for class_name in (*CLASS_NAMES, "UNKNOWN"):
        subset = classified[classes == class_name]
        medians = " ".join(_finite_median(subset, name) for name in SUMMARY_COLUMNS)
        print(f"{class_name} {medians}", file=stream)


def print_thresholds(
    config: ClassificationThresholds = DEFAULT_THRESHOLDS,
    stream=sys.stdout,
) -> None:
    for field in fields(config):
        print(f"{field.name}={getattr(config, field.name)}", file=stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Candidate subdivision of Gaia DR3 BE|GCAS|SDOR|WR objects. "
            "Heuristic results do not replace spectroscopy."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-thresholds", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.input.parent
    if args.show_thresholds:
        print_thresholds()
    # The downloaded FITS metadata contains several valid Gaia/VOTable unit
    # strings that are not strict FITS units.  Preserve them without flooding
    # the classification report with repeated Astropy conversion warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        source = Table.read(args.input)
        classified = classify_table(source)
        validate_catalogues(source, classified)
        write_catalogues(classified, output_dir, overwrite=args.overwrite)
    print_report(classified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
