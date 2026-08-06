"""
Download Gaia DR3 variable-star catalogues used by AURORA.

RR Lyrae and Cepheids come from their dedicated Specific Object Study tables.
The remaining catalogues use Gaia's general variability classifier. Downloads
are resumable at the catalogue-file level and are written atomically as FITS.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_paths import ASSETS_DIR


DEFAULT_ROW_LIMIT = 500_000
DEFAULT_QUERY_ATTEMPTS = 5
DEFAULT_RETRY_DELAY = 30.0
DEFAULT_MIN_CLASSIFIER_SCORE = 0.0
CORE_CATALOG_KEYS = ("rr_lyrae", "cepheids", "lbv", "cataclysmic", "zz_ceti")
MEMORY = MemoryController.from_environment()


@dataclass(frozen=True)
class CatalogSpec:
    key: str
    filename: str
    source_kind: str
    classifier_labels: tuple[str, ...]
    output_class: str
    default_period_days: float


CATALOG_SPECS = {
    "rr_lyrae": CatalogSpec(
        key="rr_lyrae",
        filename="rr_lyrae.fits",
        source_kind="rr_sos",
        classifier_labels=("RR",),
        output_class="RR_LYRAE",
        default_period_days=0.5,
    ),
    "cepheids": CatalogSpec(
        key="cepheids",
        filename="cepheids.fits",
        source_kind="cepheid_sos",
        classifier_labels=("CEP",),
        output_class="CEPHEIDS",
        default_period_days=5.0,
    ),
    "lbv": CatalogSpec(
        key="lbv",
        filename="lbv.fits",
        source_kind="classifier",
        classifier_labels=("BE|GCAS|SDOR|WR",),
        output_class="LBV",
        default_period_days=100.0,
    ),
    "cataclysmic": CatalogSpec(
        key="cataclysmic",
        filename="cataclysmic_variables.fits",
        source_kind="classifier",
        classifier_labels=("CV",),
        output_class="CATACLYSMIC",
        default_period_days=0.2,
    ),
    "zz_ceti": CatalogSpec(
        key="zz_ceti",
        filename="zz_ceti.fits",
        source_kind="classifier",
        classifier_labels=("WD",),
        output_class="ZZ_CETI",
        default_period_days=0.01,
    ),
    "other": CatalogSpec(
        key="other",
        filename="other_variables.fits",
        source_kind="other_classifier",
        classifier_labels=(),
        output_class="OTHER",
        default_period_days=1.0,
    ),
}


BASE_SOURCE_COLUMNS = """
            g.source_id,
            g.l,
            g.b,
            g.parallax,
            g.parallax_error,
            g.phot_g_mean_mag,
            g.teff_gspphot,
            g.teff_gspphot_lower,
            g.teff_gspphot_upper,
            g.logg_gspphot,
            g.logg_gspphot_lower,
            g.logg_gspphot_upper,
            g.radius_gspphot,
            g.radius_gspphot_lower,
            g.radius_gspphot_upper,
            g.lum_flame,
            g.lum_flame_lower,
            g.lum_flame_upper,
            g.mass_flame,
            g.mass_flame_lower,
            g.mass_flame_upper,
            g.ag_gspphot,
            g.ag_gspphot_lower,
            g.ag_gspphot_upper,
            g.ebpminrp_gspphot,
            g.ebpminrp_gspphot_lower,
            g.ebpminrp_gspphot_upper,
            g.bp_rp,
            g.vbroad,
            g.vbroad_error,
            g.rv_amplitude_robust,
            g.spectraltype_esphs,
            g.random_index
"""
VARIABILITY_SUMMARY_COLUMNS = """
            s.range_mag_g_fov,
            s.trimmed_range_mag_g_fov,
            s.std_dev_mag_g_fov,
            s.skewness_mag_g_fov,
            s.kurtosis_mag_g_fov
"""
AUXILIARY_FLOAT_COLUMNS = (
    "teff_gspphot_lower",
    "teff_gspphot_upper",
    "logg_gspphot",
    "logg_gspphot_lower",
    "logg_gspphot_upper",
    "radius_gspphot",
    "radius_gspphot_lower",
    "radius_gspphot_upper",
    "lum_flame",
    "lum_flame_lower",
    "lum_flame_upper",
    "mass_flame",
    "mass_flame_lower",
    "mass_flame_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
    "ebpminrp_gspphot_lower",
    "ebpminrp_gspphot_upper",
    "vbroad",
    "vbroad_error",
    "rv_amplitude_robust",
    "range_mag_g_fov",
    "trimmed_range_mag_g_fov",
    "std_dev_mag_g_fov",
    "skewness_mag_g_fov",
    "kurtosis_mag_g_fov",
)
AUXILIARY_TEXT_COLUMNS = ("spectraltype_esphs",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Gaia DR3 variable-star catalogues for "
            "aurora_variable_animation.py."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Output directory for FITS catalogues (default: %(default)s).",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        choices=tuple(CATALOG_SPECS),
        dest="catalogs",
        help=(
            "Catalogue to download; repeat the option for multiple catalogues. "
            "By default all five animation catalogues are downloaded."
        ),
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Also download all remaining Gaia variability classes.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help=(
            "Maximum rows per output catalogue; use 0 for no TOP limit "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_CLASSIFIER_SCORE,
        help=(
            "Minimum general-classifier score for LBV/CV/WD/other "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument("--username", default=os.environ.get("GAIA_USER"))
    parser.add_argument("--password", default=os.environ.get("GAIA_PASSWORD"))
    parser.add_argument(
        "--prompt-password",
        action="store_true",
        help="Prompt for a Gaia password when --username is supplied.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_QUERY_ATTEMPTS,
        help="Maximum query attempts per catalogue (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Base retry delay in seconds (default: %(default)s).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ADQL queries without contacting Gaia or writing files.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.row_limit < 0:
        raise ValueError("--row-limit cannot be negative")
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be between 0 and 1")
    if args.attempts <= 0:
        raise ValueError("--attempts must be greater than zero")
    if args.retry_delay < 0.0:
        raise ValueError("--retry-delay cannot be negative")
    if args.password and not args.username:
        raise ValueError("--password requires --username")
    if args.username and not args.password and not args.prompt_password:
        raise ValueError(
            "--username requires --password, GAIA_PASSWORD, or "
            "--prompt-password"
        )


def selected_specs(args: argparse.Namespace) -> list[CatalogSpec]:
    keys = list(args.catalogs) if args.catalogs else list(CORE_CATALOG_KEYS)
    if args.include_other and "other" not in keys:
        keys.append("other")
    return [CATALOG_SPECS[key] for key in dict.fromkeys(keys)]


def _top_clause(row_limit: int) -> str:
    return f"TOP {row_limit} " if row_limit else ""


def _quoted_labels(labels: tuple[str, ...]) -> str:
    return ", ".join(f"'{label}'" for label in labels)


def build_query(
    spec: CatalogSpec,
    row_limit: int,
    min_score: float,
) -> str:
    top = _top_clause(row_limit)

    if spec.source_kind == "rr_sos":
        return f"""
        SELECT {top}
{BASE_SOURCE_COLUMNS},
            v.pf AS period,
            v.p1_o AS alternate_period,
            v.epoch_g AS epoch_g,
            v.peak_to_peak_g AS amplitude,
            0.0 AS phase,
            v.best_classification AS source_subclass,
            s.trimmed_range_mag_g_fov AS summary_amplitude,
{VARIABILITY_SUMMARY_COLUMNS}
        FROM gaiadr3.vari_rrlyrae AS v
        JOIN gaiadr3.gaia_source AS g
            ON v.source_id = g.source_id
        LEFT OUTER JOIN gaiadr3.vari_summary AS s
            ON v.source_id = s.source_id
        WHERE g.l IS NOT NULL
            AND g.b IS NOT NULL
        ORDER BY g.random_index
        """.strip()

    if spec.source_kind == "cepheid_sos":
        return f"""
        SELECT {top}
{BASE_SOURCE_COLUMNS},
            v.pf AS period,
            v.p1_o AS alternate_period,
            v.epoch_g AS epoch_g,
            v.peak_to_peak_g AS amplitude,
            0.0 AS phase,
            v.type_best_classification AS source_subclass,
            s.trimmed_range_mag_g_fov AS summary_amplitude,
{VARIABILITY_SUMMARY_COLUMNS}
        FROM gaiadr3.vari_cepheid AS v
        JOIN gaiadr3.gaia_source AS g
            ON v.source_id = g.source_id
        LEFT OUTER JOIN gaiadr3.vari_summary AS s
            ON v.source_id = s.source_id
        WHERE g.l IS NOT NULL
            AND g.b IS NOT NULL
        ORDER BY g.random_index
        """.strip()

    core_labels = tuple(
        label
        for key in CORE_CATALOG_KEYS
        for label in CATALOG_SPECS[key].classifier_labels
    )
    if spec.source_kind == "other_classifier":
        class_filter = (
            f"c.best_class_name NOT IN ({_quoted_labels(core_labels)})"
        )
    else:
        class_filter = (
            f"c.best_class_name IN ({_quoted_labels(spec.classifier_labels)})"
        )

    return f"""
        SELECT {top}
{BASE_SOURCE_COLUMNS},
            c.best_class_name AS classifier_class,
            c.best_class_score AS classifier_score,
            {spec.default_period_days:.12g} AS period,
            s.trimmed_range_mag_g_fov AS amplitude,
            0.0 AS phase,
{VARIABILITY_SUMMARY_COLUMNS}
        FROM gaiadr3.vari_classifier_result AS c
        JOIN gaiadr3.gaia_source AS g
            ON c.source_id = g.source_id
        LEFT OUTER JOIN gaiadr3.vari_summary AS s
            ON c.source_id = s.source_id
        WHERE {class_filter}
            AND c.best_class_score >= {min_score:.12g}
            AND g.l IS NOT NULL
            AND g.b IS NOT NULL
        ORDER BY g.random_index
    """.strip()


def _column_array(
    table: Any,
    name: str,
    dtype: Any,
    default: float | int = np.nan,
) -> np.ndarray:
    if name not in table.colnames:
        return np.full(len(table), default, dtype=dtype)
    column = table[name]
    if hasattr(column, "filled"):
        column = column.filled(default)
    return np.asarray(column, dtype=dtype)


def _text_array(
    table: Any,
    name: str,
    default: str,
    width: int = 40,
) -> np.ndarray:
    if name not in table.colnames:
        return np.full(len(table), default, dtype=f"U{width}")
    column = table[name]
    if hasattr(column, "filled"):
        column = column.filled(default)
    result = np.asarray(column).astype(f"U{width}")
    result[result == ""] = default
    return result


def _first_positive(
    primary: np.ndarray,
    secondary: np.ndarray,
    fallback: float,
) -> np.ndarray:
    result = np.asarray(primary, dtype=np.float64).copy()
    missing = ~np.isfinite(result) | (result <= 0.0)
    result[missing] = secondary[missing]
    missing = ~np.isfinite(result) | (result <= 0.0)
    result[missing] = fallback
    return result


def prepare_catalog(table: Any, spec: CatalogSpec) -> Table:
    source_id = _column_array(table, "source_id", np.int64, default=-1)
    lon = _column_array(table, "l", np.float64)
    lat = _column_array(table, "b", np.float64)
    valid = (source_id >= 0) & np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        raise RuntimeError(f"No valid positions returned for {spec.key}")

    parallax = _column_array(table, "parallax", np.float32)
    parallax_error = _column_array(table, "parallax_error", np.float32)
    g_mag = _column_array(table, "phot_g_mean_mag", np.float32)
    teff = _column_array(table, "teff_gspphot", np.float32)
    bp_rp = _column_array(table, "bp_rp", np.float32)
    summary_amplitude = _column_array(
        table,
        "summary_amplitude",
        np.float32,
    )

    if spec.source_kind in {"rr_sos", "cepheid_sos"}:
        period = _first_positive(
            _column_array(table, "period", np.float64),
            _column_array(table, "alternate_period", np.float64),
            spec.default_period_days,
        )
        amplitude = _first_positive(
            _column_array(table, "amplitude", np.float64),
            summary_amplitude,
            1.0,
        )
        epoch = _column_array(table, "epoch_g", np.float64)
        phase = np.zeros(len(table), dtype=np.float64)
        has_epoch = np.isfinite(epoch)
        phase[has_epoch] = np.remainder(
            2.0 * np.pi * epoch[has_epoch] / period[has_epoch],
            2.0 * np.pi,
        )
        variable_class = np.full(
            len(table),
            spec.output_class,
            dtype="U40",
        )
        variable_subclass = _text_array(
            table,
            "source_subclass",
            spec.classifier_labels[0],
        )
        classifier_score = np.full(len(table), np.nan, dtype=np.float32)
    else:
        period = np.full(
            len(table),
            spec.default_period_days,
            dtype=np.float64,
        )
        amplitude = _first_positive(
            _column_array(table, "amplitude", np.float64),
            np.full(len(table), np.nan),
            1.0,
        )
        phase = (
            np.remainder(source_id, 1_000_003).astype(np.float64)
            / 1_000_003.0
            * (2.0 * np.pi)
        )
        classifier_class = _text_array(
            table,
            "classifier_class",
            spec.output_class,
        )
        variable_class = (
            classifier_class
            if spec.source_kind == "other_classifier"
            else np.full(len(table), spec.output_class, dtype="U40")
        )
        variable_subclass = classifier_class
        classifier_score = _column_array(
            table,
            "classifier_score",
            np.float32,
        )

    catalog = Table(
        {
            "source_id": source_id[valid],
            "l": lon[valid],
            "b": lat[valid],
            "parallax": parallax[valid],
            "parallax_error": parallax_error[valid],
            "phot_g_mean_mag": g_mag[valid],
            "teff_gspphot": teff[valid],
            "bp_rp": bp_rp[valid],
            "period": period[valid].astype(np.float32),
            "amplitude": amplitude[valid].astype(np.float32),
            "phase": phase[valid].astype(np.float32),
            "variable_class": variable_class[valid],
            "variable_subclass": variable_subclass[valid],
            "classification_score": classifier_score[valid],
            "best_class_score": classifier_score[valid],
        },
        copy=False,
    )
    for name in AUXILIARY_FLOAT_COLUMNS:
        catalog[name] = _column_array(table, name, np.float32)[valid]
    for name in AUXILIARY_TEXT_COLUMNS:
        catalog[name] = _text_array(table, name, "")[valid]
    catalog["l"].unit = "deg"
    catalog["b"].unit = "deg"
    catalog["parallax"].unit = "mas"
    catalog["parallax_error"].unit = "mas"
    catalog["phot_g_mean_mag"].unit = "mag"
    catalog["teff_gspphot"].unit = "K"
    catalog["ag_gspphot"].unit = "mag"
    catalog["ebpminrp_gspphot"].unit = "mag"
    catalog["period"].unit = "d"
    catalog["amplitude"].unit = "mag"
    catalog["phase"].unit = "rad"
    catalog.meta["GAIADR"] = "DR3"
    catalog.meta["VARCAT"] = spec.key
    return catalog


def save_fits_atomic(table: Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        table.write(temporary, format="fits", overwrite=True)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _login(args: argparse.Namespace) -> bool:
    from astroquery.gaia import Gaia

    if not args.username:
        console.print("  → Using anonymous Gaia Archive access")
        return False
    password = args.password
    if not password:
        password = console.prompt("Gaia Archive password", secret=True)
    if not password:
        raise RuntimeError("Gaia Archive password cannot be empty")
    console.print(f"  → Logging in to Gaia Archive as {args.username}")
    Gaia.login(user=args.username, password=password)
    console.print("  ✓ Gaia authenticated session ready")
    return True


def execute_query(
    query: str,
    attempts: int,
    retry_delay: float,
) -> Any:
    try:
        from astroquery.gaia import Gaia
    except ImportError as error:
        raise RuntimeError(
            "astroquery is required. Install project requirements first."
        ) from error

    Gaia.ROW_LIMIT = -1
    for attempt in range(1, attempts + 1):
        try:
            job = Gaia.launch_job_async(query, dump_to_file=False)
            return job.get_results()
        except Exception as error:
            console.print(f"  ! Query attempt {attempt}/{attempts} failed: {error}")
            if attempt == attempts:
                raise
            wait_seconds = retry_delay * attempt
            console.print(f"  → Retrying in {wait_seconds:g} s")
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    MEMORY.throttle()

    console.print("\n[AURORA] Variable-star catalogue downloader started")
    console.print("─" * 52)
    started = time.perf_counter()
    logged_in = False
    try:
        validate_args(args)
        specs = selected_specs(args)
        queries = {
            spec.key: build_query(spec, args.row_limit, args.min_score)
            for spec in specs
        }

        console.print(f"  → Output directory: {args.output_dir}")
        console.print(
            "  → Catalogues: "
            + ", ".join(spec.key for spec in specs)
        )
        console.print(
            "  → Row limit per catalogue: "
            + (f"{args.row_limit:,}" if args.row_limit else "none")
        )
        console.print(f"  → Minimum classifier score: {args.min_score:g}")

        if args.dry_run:
            for spec in specs:
                console.print(f"\n--- {spec.key}: {spec.filename} ---")
                console.print(queries[spec.key])
            return 0

        from astroquery.gaia import Gaia

        logged_in = _login(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for position, spec in enumerate(specs, start=1):
            MEMORY.throttle()
            output_path = args.output_dir / spec.filename
            console.print(
                f"\n[AURORA] Catalogue {position}/{len(specs)}: {spec.key}"
            )
            console.print("─" * 52)
            if output_path.exists() and not args.overwrite:
                console.print(f"  ✓ Already exists, skipping: {output_path}")
                continue

            query_started = time.perf_counter()
            console.print("  → Launching asynchronous Gaia DR3 query")
            result = execute_query(
                queries[spec.key],
                args.attempts,
                args.retry_delay,
            )
            console.print(
                f"  ✓ Downloaded {len(result):,} rows in "
                f"{time.perf_counter() - query_started:.1f}s"
            )
            catalog = prepare_catalog(result, spec)
            save_fits_atomic(catalog, output_path)
            size_mb = output_path.stat().st_size / (1024.0 * 1024.0)
            console.print(
                f"  ✓ Saved {len(catalog):,} rows: "
                f"{output_path} ({size_mb:.1f} MiB)"
            )
            del result, catalog

        console.print(
            f"\n  ✓ Completed in {time.perf_counter() - started:.1f}s"
        )
        console.print("=== AURORA VARIABLE-STAR CATALOGUES FINISHED ===")
        return 0
    except KeyboardInterrupt:
        console.print("\n  ! Interrupted by user")
        return 130
    except Exception as error:
        console.print(f"\n  ! Catalogue generation failed: {error}", file=sys.stderr)
        return 1
    finally:
        if logged_in:
            try:
                Gaia.logout()
                console.print("  ✓ Gaia Archive session closed")
            except Exception as error:
                console.print(f"  ! Could not close Gaia session cleanly: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
