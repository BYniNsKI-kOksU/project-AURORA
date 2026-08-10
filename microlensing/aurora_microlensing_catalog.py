"""
AURORA module for acquiring Gaia DR3 gravitational microlensing catalogs.

The module retrieves Gaia microlensing events and combines Paczynski model
parameters with Galactic coordinates and stellar photometry required for
AURORA microlensing rendering.

Parallax filtering removes unreliable astrometric solutions and limits the
catalog to Galactic stellar sources suitable for visualisation.

The generated FITS catalog is used as input for the AURORA microlensing
renderer.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astroquery.gaia import Gaia


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_paths import ASSETS_DIR, asset_path


# ─────────────────────────────────────────────────────────────
# Program settings
# ─────────────────────────────────────────────────────────────
OUTPUT = asset_path("aurora_microlensing.fits")
REQUIRED_OUTPUT_COLUMNS = {
    "source_id",
    "l",
    "b",
    "paczynski0_tmax",
    "paczynski0_te",
    "paczynski0_u0",
    "paczynski1_tmax",
    "paczynski1_te",
    "paczynski1_u0",
    "paczynski1_bp0",
    "paczynski1_rp0",
    "paczynski1_fs_g",
    "paczynski1_fs_bp",
    "paczynski1_fs_rp",
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_bp_mean_flux",
    "phot_rp_mean_flux",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "teff_gspphot",
    "bp_rp",
    "abp_gspphot",
    "arp_gspphot",
    "ebpminrp_gspphot",
}
MEMORY = MemoryController.from_environment()


def _configure_gaia_login():
    """Log in without storing Gaia Archive credentials in the source code."""
    user = os.environ.get("GAIA_USER")
    password = os.environ.get("GAIA_PASSWORD")

    if not user:
        user = console.prompt("Gaia Archive username").strip()
    if not password:
        password = console.prompt("Gaia Archive password", secret=True)
    if not user or not password:
        raise RuntimeError(
            "Gaia Archive credentials are required. Set GAIA_USER and "
            "GAIA_PASSWORD, or enter them at the prompt."
        )

    console.print("  → Logging in to Gaia Archive")
    try:
        Gaia.login(user=user, password=password)
    except Exception as error:
        raise RuntimeError("Gaia Archive login failed") from error
    console.print("  ✓ Gaia authenticated session ready")


def _catalog_has_required_columns(path):
    """Check whether an existing FITS uses the current catalogue schema."""
    try:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    names = {name.lower() for name in hdu.columns.names}
                    return REQUIRED_OUTPUT_COLUMNS.issubset(names)
    except OSError:
        return False
    return False


def _finite_count(column):
    """Count finite values in a possibly masked Astropy column."""
    if hasattr(column, "filled"):
        column = column.filled(np.nan)
    return int(np.count_nonzero(np.isfinite(np.asarray(column, dtype=float))))


def _normalise_fits_object_columns(table):
    """Convert TAP object-string columns to fixed-width FITS strings."""
    for name in table.colnames:
        column = table[name]
        if np.asarray(column).dtype.kind != "O":
            continue
        values = column.filled("") if hasattr(column, "filled") else column
        strings = np.asarray(
            ["" if value is None else str(value) for value in values],
            dtype=str,
        )
        table[name] = strings


# ─────────────────────────────────────────────────────────────
# Gaia DR3 microlensing catalog acquisition pipeline
# ─────────────────────────────────────────────────────────────
def download_microlensing():
    MEMORY.throttle()
    console.print("\n[AURORA] Microlensing catalog downloader started")
    console.print("─" * 45)
    ASSETS_DIR.mkdir(exist_ok=True)

    if OUTPUT.exists() and _catalog_has_required_columns(OUTPUT):
        console.print(f"  ✓ Catalog already exists: {OUTPUT}")
        return
    if OUTPUT.exists():
        console.print(
            "  → Existing catalog does not match the current schema; "
            "it will be replaced atomically"
        )

    _configure_gaia_login()
    console.print("  → Downloading Gaia DR3 microlensing catalog")

    # Level 0 is the unblended Paczynski fit.  Level 1 additionally separates
    # source and blend flux in G/BP/RP and is therefore essential both for the
    # observed amplification and for recovering the lensed source colour.
    # GSP-Phot supplies intrinsic stellar parameters and line-of-sight
    # extinction.  LEFT OUTER JOIN retains events without an AP solution.
    query = """
    SELECT
        v.solution_id,
        v.source_id,
        v.paczynski0_g0,
        v.paczynski0_g0_error,
        v.paczynski0_bp0,
        v.paczynski0_bp0_error,
        v.paczynski0_rp0,
        v.paczynski0_rp0_error,
        v.paczynski0_u0,
        v.paczynski0_u0_error,
        v.paczynski0_te,
        v.paczynski0_te_error,
        v.paczynski0_tmax,
        v.paczynski0_tmax_error,
        v.paczynski0_chi2,
        v.paczynski0_chi2_dof,
        v.paczynski1_g0,
        v.paczynski1_g0_error,
        v.paczynski1_bp0,
        v.paczynski1_bp0_error,
        v.paczynski1_rp0,
        v.paczynski1_rp0_error,
        v.paczynski1_u0,
        v.paczynski1_u0_error,
        v.paczynski1_te,
        v.paczynski1_te_error,
        v.paczynski1_tmax,
        v.paczynski1_tmax_error,
        v.paczynski1_fs_g,
        v.paczynski1_fs_g_error,
        v.paczynski1_fs_bp,
        v.paczynski1_fs_bp_error,
        v.paczynski1_fs_rp,
        v.paczynski1_fs_rp_error,
        v.paczynski1_chi2,
        v.paczynski1_chi2_dof,
        g.ra,
        g.dec,
        g.l,
        g.b,
        g.parallax,
        g.parallax_error,
        g.pmra,
        g.pmra_error,
        g.pmdec,
        g.pmdec_error,
        g.ruwe,
        g.phot_g_n_obs,
        g.phot_g_mean_flux,
        g.phot_g_mean_flux_error,
        g.phot_g_mean_mag,
        g.phot_bp_n_obs,
        g.phot_bp_mean_flux,
        g.phot_bp_mean_flux_error,
        g.phot_bp_mean_mag,
        g.phot_rp_n_obs,
        g.phot_rp_mean_flux,
        g.phot_rp_mean_flux_error,
        g.phot_rp_mean_mag,
        g.phot_bp_rp_excess_factor,
        g.phot_bp_n_contaminated_transits,
        g.phot_bp_n_blended_transits,
        g.phot_rp_n_contaminated_transits,
        g.phot_rp_n_blended_transits,
        g.phot_proc_mode,
        g.bp_rp,
        g.teff_gspphot,
        g.teff_gspphot_lower,
        g.teff_gspphot_upper,
        g.logg_gspphot,
        g.mh_gspphot,
        g.ag_gspphot,
        g.ebpminrp_gspphot,
        g.has_xp_continuous,
        g.has_xp_sampled,
        g.has_epoch_photometry,
        ap.azero_gspphot,
        ap.abp_gspphot,
        ap.arp_gspphot,
        ap.logposterior_gspphot,
        ap.mcmcaccept_gspphot,
        ap.libname_gspphot
    FROM gaiadr3.vari_microlensing AS v
    JOIN gaiadr3.gaia_source AS g
        ON v.source_id = g.source_id
    LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap
        ON v.source_id = ap.source_id
    WHERE
        g.l IS NOT NULL
        AND g.b IS NOT NULL
        AND v.paczynski0_te IS NOT NULL
        AND v.paczynski0_tmax IS NOT NULL
        AND v.paczynski0_u0 IS NOT NULL
        AND v.paczynski0_te > 0
        AND g.parallax IS NOT NULL
        AND g.parallax > 0.03
        AND g.parallax < 21
    """

    start = time.time()

    # Single query is sufficient because the microlensing event table is small.
    MEMORY.throttle()
    job = Gaia.launch_job_async(query)
    data = job.get_results()

    console.print(f"  ✓ Downloaded {len(data):,} microlensing events")
    console.print(f"  ✓ Query completed in {time.time() - start:.1f}s")
    console.print("  → Temperature and colour statistics")
    console.print(
        f"    teff_gspphot available: "
        f"{_finite_count(data['teff_gspphot']):,}"
    )
    console.print(f"    bp_rp available: {_finite_count(data['bp_rp']):,}")
    console.print(
        f"    extinction E(BP-RP) available: "
        f"{_finite_count(data['ebpminrp_gspphot']):,}"
    )
    console.print(
        f"    level-1 G blending available: "
        f"{_finite_count(data['paczynski1_fs_g']):,}"
    )
    console.print(
        f"    parallax_error available: "
        f"{_finite_count(data['parallax_error']):,}"
    )

    console.print(f"  → Saving catalog: {OUTPUT}")

    # Atomic replacement prevents an interrupted write from looking like a
    # valid cache on the next run.
    MEMORY.throttle()
    _normalise_fits_object_columns(data)
    temp_output = OUTPUT.with_name(f"{OUTPUT.name}.tmp")
    try:
        data.write(temp_output, format="fits", overwrite=True)
        os.replace(temp_output, OUTPUT)
    except Exception:
        try:
            temp_output.unlink()
        except FileNotFoundError:
            pass
        raise

    console.print("  ✓ Microlensing catalog saved")
    console.print("=== AURORA MICROLENSING CATALOG DOWNLOAD FINISHED ===")


if __name__ == "__main__":
    download_microlensing()
