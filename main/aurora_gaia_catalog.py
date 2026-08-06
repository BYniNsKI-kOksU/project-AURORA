"""
Download and assemble the Gaia DR3 catalogue used by AURORA.

Downloads are resumable. The final FITS table is assembled by copying each
chunk's binary table payload directly, avoiding an in-memory vstack or
per-record Astropy conversion for up to the configured Gaia DR3 source count.
"""

import os
import threading
import time
from pathlib import Path
import sys

from astropy.io import fits
from astroquery.gaia import Gaia


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.aurora_console import console
from core.aurora_memory import MemoryController
from core.aurora_paths import ASSETS_DIR, asset_path


OUTPUT = asset_path("aurora_gaia_catalog_900m.fits")
RECORDS_DIR = ASSETS_DIR / "gaia_chunks"
CHUNK_PREFIX = "gaia_chunk"
N_CHUNKS = 61
PER_CHUNK = 30_000_000
QUERY_TIMEOUT = 60 * 60
MAX_QUERY_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 30
N_TOTAL = 1_811_709_771
COPY_BUFFER_BYTES = 64 * 1024 * 1024
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


def _run_query_with_timeout(query, timeout):
    """Run a Gaia TAP query with a client-side timeout."""
    result_box = {}

    def worker():
        try:
            job = Gaia.launch_job_async(query)
            result_box["result"] = job.get_results()
        except Exception as error:
            result_box["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Gaia query did not complete in {timeout}s")
    if "error" in result_box:
        raise result_box["error"]
    return result_box["result"]


def _chunk_paths():
    return [
        RECORDS_DIR / f"{CHUNK_PREFIX}_{index:02d}.fits"
        for index in range(N_CHUNKS)
    ]


def _ready_chunk_paths(chunk_paths):
    """Return already downloaded FITS chunks from the Records directory."""
    return [path for path in chunk_paths if path.is_file()]


def _print_chunk_status(chunk_paths):
    ready_paths = _ready_chunk_paths(chunk_paths)
    console.print(
        f"  ✓ Ready FITS chunks in {RECORDS_DIR}: "
        f"{len(ready_paths)}/{len(chunk_paths)}"
    )
    if ready_paths:
        console.print(
            "  → Already downloaded: "
            + ", ".join(path.name for path in ready_paths)
        )
    return ready_paths


def _table_hdu(hdul):
    for hdu in hdul:
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return hdu
    raise RuntimeError("Downloaded FITS file contains no table HDU")


def _copy_exact_bytes(source_file, output_file, byte_count):
    """Copy exactly *byte_count* bytes without decoding FITS rows."""
    remaining = byte_count
    while remaining:
        MEMORY.throttle()
        block = source_file.read(min(COPY_BUFFER_BYTES, remaining))
        if not block:
            raise OSError(
                "Unexpected end of a Gaia chunk while copying its FITS data"
            )
        output_file.write(block)
        remaining -= len(block)


def combine_chunks(chunk_paths, output_path=OUTPUT):
    """Assemble equal-schema FITS chunks with direct byte-for-byte copying."""
    chunk_paths = [Path(path) for path in chunk_paths]
    output_path = Path(output_path)
    if not chunk_paths:
        raise ValueError("No Gaia chunks were supplied")

    row_counts = []
    reference_signature = None
    primary_header = None
    table_header = None
    data_offsets = []

    console.print("\n[AURORA] Combining catalog chunks")
    console.print("─" * 45)
    for path in chunk_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Gaia chunk: {path}")
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            hdu = _table_hdu(hdul)
            signature = (
                int(hdu.header["NAXIS1"]),
                tuple(hdu.columns.names),
                tuple(hdu.columns.formats),
            )
            if int(hdu.header.get("PCOUNT", 0)) != 0:
                raise ValueError(
                    f"Unsupported variable-length FITS table in {path}"
                )
            if reference_signature is None:
                reference_signature = signature
                primary_header = hdul[0].header.copy()
                table_header = hdu.header.copy()
            elif signature != reference_signature:
                raise ValueError(f"Incompatible FITS schema in {path}")
            row_counts.append(int(hdu.header["NAXIS2"]))
            file_info = hdu.fileinfo()
            data_offsets.append(int(file_info["datLoc"]))

    total_rows = sum(row_counts)
    row_size = reference_signature[0]
    table_header["NAXIS2"] = total_rows
    for keyword in ("CHECKSUM", "DATASUM"):
        primary_header.pop(keyword, None)
        table_header.pop(keyword, None)

    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        # FITS headers and data blocks are padded to 2880-byte boundaries.
        with temp_path.open("wb") as output_file:
            output_file.write(
                primary_header.tostring(
                    sep="",
                    endcard=True,
                    padding=True,
                ).encode("ascii")
            )
            output_file.write(
                table_header.tostring(
                    sep="",
                    endcard=True,
                    padding=True,
                ).encode("ascii")
            )
            for path, row_count, data_offset in zip(
                chunk_paths,
                row_counts,
                data_offsets,
            ):
                console.print(f"  → Appending chunk: {path} ({row_count:,} rows)")
                with path.open("rb") as source_file:
                    source_file.seek(data_offset)
                    _copy_exact_bytes(
                        source_file,
                        output_file,
                        row_count * row_size,
                    )

            data_bytes = total_rows * row_size
            padding = (-data_bytes) % 2880
            if padding:
                output_file.write(b"\0" * padding)

        os.replace(temp_path, output_path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    console.print(f"  ✓ Combined catalog contains {total_rows:,} stars")
    console.print(f"  ✓ Gaia catalog saved: {output_path}")


def download_gaia():
    console.print("\n[AURORA] Gaia catalog downloader started")
    console.print("─" * 45)
    ASSETS_DIR.mkdir(exist_ok=True)
    RECORDS_DIR.mkdir(exist_ok=True)
    chunk_paths = _chunk_paths()
    _print_chunk_status(chunk_paths)

    if OUTPUT.exists():
        console.print(f"  ✓ Catalog already exists: {OUTPUT}")
        return

    console.print(f"  ✓ Chunk directory ready: {RECORDS_DIR}")
    if len(_ready_chunk_paths(chunk_paths)) < len(chunk_paths):
        _configure_gaia_login()
    console.print(f"  → Downloading {N_CHUNKS} deterministic Gaia samples")
    console.print(f"  → Up to {PER_CHUNK:,} stars per sample")

    # random_index is a random permutation. Each wide range is sorted by that
    # key. With the current 61 ranges, every interval is narrower than
    # PER_CHUNK, so the query keeps all valid sources in that interval. The
    # legacy output filename is retained for compatibility.
    stride = N_TOTAL // N_CHUNKS
    for index, chunk_path in enumerate(chunk_paths):
        MEMORY.throttle()
        if chunk_path.exists():
            console.print(f"  ✓ Chunk already exists: {chunk_path}")
            continue

        random_lo = index * stride
        random_hi = (
            (index + 1) * stride
            if index < N_CHUNKS - 1
            else N_TOTAL
        )
        console.print(
            f"  → Range {index + 1}/{N_CHUNKS}: "
            f"random_index ∈ [{random_lo:,}, {random_hi:,})"
        )
        query = f"""
        SELECT TOP {PER_CHUNK}
            l,
            b,
            phot_g_mean_mag,
            teff_gspphot,
            bp_rp
        FROM gaiadr3.gaia_source
        WHERE
            random_index >= {random_lo}
            AND random_index < {random_hi}
            AND l IS NOT NULL
            AND b IS NOT NULL
            AND phot_g_mean_mag IS NOT NULL
        """
        # ORDER BY random_index


        started = time.perf_counter()
        for attempt in range(1, MAX_QUERY_ATTEMPTS + 1):
            try:
                result = _run_query_with_timeout(query, QUERY_TIMEOUT)
                break
            except Exception as error:
                console.print(
                    f"  ! Query attempt {attempt}/{MAX_QUERY_ATTEMPTS} "
                    f"failed: {error}"
                )
                if attempt == MAX_QUERY_ATTEMPTS:
                    raise
                wait_seconds = RETRY_DELAY_SECONDS * attempt
                console.print(f"  → Retrying in {wait_seconds} s")
                time.sleep(wait_seconds)

        temp_path = chunk_path.with_name(f"{chunk_path.name}.tmp")
        try:
            result.write(temp_path, format="fits", overwrite=True)
            os.replace(temp_path, chunk_path)
        except Exception:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise
        console.print(
            f"  ✓ Downloaded {len(result):,} stars in "
            f"{time.perf_counter() - started:.1f}s"
        )
        del result

    combine_chunks(chunk_paths)
    console.print("=== AURORA GAIA CATALOG DOWNLOAD FINISHED ===")


if __name__ == "__main__":
    download_gaia()
