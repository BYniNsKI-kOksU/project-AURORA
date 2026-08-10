"""Semi-analytic supernova physics and AURORA-compatible animation."""

from .explosion import ExplosionParameters, derive_explosion_parameters
from .light_curve import LightCurve, SupernovaLightCurveModel
from .photometry import BandPhotometry, FILTERS, FilterBand
from .progenitor import (
    ChemicalComposition,
    ModelOverrides,
    Progenitor,
    SUPPORTED_SUPERNOVA_TYPES,
)
from .units import Distance

__all__ = [
    "ChemicalComposition",
    "BandPhotometry",
    "Distance",
    "ExplosionParameters",
    "FILTERS",
    "FilterBand",
    "LightCurve",
    "ModelOverrides",
    "Progenitor",
    "SUPPORTED_SUPERNOVA_TYPES",
    "SupernovaLightCurveModel",
    "derive_explosion_parameters",
]
