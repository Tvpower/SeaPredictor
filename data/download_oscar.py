"""
download_oscar.py
-----------------
Downloads NOAA OSCAR V2.0 surface current data (U/V vectors) from NASA PO.DAAC
using the earthaccess library. Filters by date range and bounding box, then
extracts 30-day temporal sequences aligned to MARIDA/MADOS tile coordinates.

Requirements:
    pip install earthaccess xarray netCDF4 numpy pandas tqdm

NASA Earthdata account required (free): https://urs.earthdata.nasa.gov/
Set credentials in .env:
    EARTHDATA_USERNAME=your_username
    EARTHDATA_PASSWORD=your_password

OSCAR V2.0 dataset: https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_INTERIM_V2.0
  - Daily files, 0.25-degree global grid
  - Variables: u (zonal), v (meridional)
  - Coverage: 1993 to present
"""

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# Date range — covers MARIDA training window (2018-2020) + 30-day lookback buffer
DATE_START = "2017-12-01"   # 30-day buffer before 2018 training data
DATE_END   = "2021-12-31"   # covers full MARIDA + MADOS range

# Bounding box — global ocean (adjust tighter to save disk space)
# For North Pacific Garbage Patch only: lat 15-50, lon -180 to -120
BBOX = {
    "lat_min": -90,
    "lat_max":  90,
    "lon_min": -180,
    "lon_max":  180,
}

# Output directory
OUT_DIR = Path("data/raw/oscar")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_DIR = Path("data/processed/oscar_sequences")
SEQUENCE_DIR.mkdir(parents=True, exist_ok=True)

# OSCAR short name on PO.DAAC (V2.0 — replaces retired third-deg)
OSCAR_SHORT_NAME = "OSCAR_L4_OC_INTERIM_V2.0"

# ─── STEP 1: AUTHENTICATE ──────────────────────────────────────────────────────

def authenticate():
    """
    Authenticates with NASA Earthdata using earthaccess.
    Credentials read from .env (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
    or interactive prompt if not set.
    """
    import earthaccess
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if username and password:
        auth = earthaccess.login(strategy="environment")
    else:
        print("No credentials in .env — prompting interactively.")
        auth = earthaccess.login(strategy="interactive", persist=True)

    print(f"Authenticated: {auth.authenticated}")
    return auth


# ─── STEP 2: SEARCH & DOWNLOAD ─────────────────────────────────────────────────

def _download_one(args):
    """Worker: download a single granule via its own authenticated session."""
    import earthaccess
    granule, out_dir = args
    try:
        result = earthaccess.download([granule], local_path=out_dir)
        return (True, result[0] if result else None, None)
    except Exception as e:
        return (False, None, str(e))


def _granule_filename(granule) -> str | None:
    """Best-effort: pull the .nc filename out of a granule's data link."""
    try:
        return Path(granule.data_links()[0]).name
    except Exception:
        return None


def _search_targeted(dates, window_days: int = 30):
    """
    Search OSCAR granules for the union of [date - window_days, date]
    windows around each MARIDA acquisition date. Deduplicates by filename
    so overlapping windows don't double-fetch.

    Args:
        dates:        iterable of ISO date strings or pd.Timestamp
        window_days:  lookback window in days (default 30 to match LSTM input)

    Returns: list of unique granule objects.
    """
    import earthaccess

    seen = set()
    unique = []
    dates = sorted({pd.Timestamp(d).normalize() for d in dates})
    print(f"\nTargeted search: {len(dates)} acquisition dates × {window_days}-day windows")

    for d in tqdm(dates, desc="Searching"):
        start = (d - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end   = d.strftime("%Y-%m-%d")
        try:
            results = earthaccess.search_data(
                short_name=OSCAR_SHORT_NAME,
                temporal=(start, end),
                bounding_box=(
                    BBOX["lon_min"], BBOX["lat_min"],
                    BBOX["lon_max"], BBOX["lat_max"]
                ),
            )
        except Exception as e:
            tqdm.write(f"  search failed for {end}: {e}")
            continue

        for g in results:
            fname = _granule_filename(g)
            key = fname or id(g)
            if key in seen:
                continue
            seen.add(key)
            unique.append(g)

    print(f"Unique granules across all windows: {len(unique)}")
    return unique


def _search_full_range():
    """Search the full DATE_START → DATE_END range (legacy behavior)."""
    import earthaccess
    print(f"\nSearching OSCAR granules: {DATE_START} → {DATE_END}")
    results = earthaccess.search_data(
        short_name=OSCAR_SHORT_NAME,
        temporal=(DATE_START, DATE_END),
        bounding_box=(
            BBOX["lon_min"], BBOX["lat_min"],
            BBOX["lon_max"], BBOX["lat_max"]
        ),
    )
    print(f"Found {len(results)} granules.")
    return results


def _resolve_dates(dates_arg: str) -> list[str]:
    """
    Accept either a CSV path (must contain a `date` column) or a
    comma-separated list of ISO dates. Returns deduplicated list of strings.

    Path resolution is tried in this order:
        1. As-given (relative to cwd)
        2. Relative to the repo root (parent of this script's dir)
    """
    looks_like_path = dates_arg.endswith(".csv") or "/" in dates_arg or "\\" in dates_arg

    if looks_like_path:
        candidates = [Path(dates_arg)]
        repo_root = Path(__file__).resolve().parent.parent
        candidates.append(repo_root / dates_arg)
        candidates.append(repo_root / "data" / Path(dates_arg).name)

        for p in candidates:
            if p.exists():
                df = pd.read_csv(p)
                if "date" not in df.columns:
                    raise ValueError(
                        f"{p} has no `date` column. Found: {list(df.columns)}"
                    )
                print(f"Loaded dates from: {p}")
                return sorted(df["date"].astype(str).unique().tolist())

        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"--dates looks like a CSV path but no file found. Tried:\n  {tried}"
        )

    return sorted({s.strip() for s in dates_arg.split(",") if s.strip()})


def download_oscar_files(
    auth,
    threads: int = 16,
    dates: list[str] | None = None,
    window_days: int = 30,
):
    """
    Downloads OSCAR V2.0 granules to data/raw/oscar/ in parallel.

    Args:
        auth:         earthaccess auth object
        threads:      concurrent download workers (default 16)
        dates:        optional list of MARIDA acquisition dates. If provided,
                      only granules within [date - window_days, date] for each
                      date are fetched (deduplicated). If None, falls back to
                      the full DATE_START → DATE_END range.
        window_days:  lookback window for targeted mode (default 30)

    Returns: list of downloaded file paths.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if dates:
        results = _search_targeted(dates, window_days=window_days)
    else:
        results = _search_full_range()

    if not results:
        raise RuntimeError("No granules found. Check date range and credentials.")

    # Pre-filter granules that are already on disk to avoid HEAD requests
    out_str = str(OUT_DIR)
    pending = []
    cached = 0
    for g in results:
        fname = _granule_filename(g)
        if fname and (OUT_DIR / fname).exists():
            cached += 1
            continue
        pending.append(g)

    print(f"Cached: {cached} | To download: {len(pending)} | Threads: {threads}")

    if not pending:
        print(f"All granules already cached in {OUT_DIR}/")
        return [str(p) for p in OUT_DIR.glob("*.nc")]

    files = []
    errors = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(_download_one, (g, out_str)) for g in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            ok, path, err = fut.result()
            if ok:
                if path:
                    files.append(path)
            else:
                errors += 1
                tqdm.write(f"  ERROR: {err}")

    print(f"Downloaded {len(files)} files to {OUT_DIR}/ (errors: {errors})")
    return files


# ─── STEP 3: BUILD SEQUENCE EXTRACTOR ──────────────────────────────────────────

class OSCARSequenceExtractor:
    """
    Loads all downloaded OSCAR NetCDF files into a lazy xarray dataset
    and extracts 30-day U/V sequences for a given (lat, lon, date).

    The sequence format matches the LSTM input:
        shape: (30, 4)
        features per timestep: [U, V, lat, lon]

    Usage:
        extractor = OSCARSequenceExtractor("data/raw/oscar")
        seq = extractor.get_sequence(lat=32.1, lon=-145.3, date="2019-06-15")
        # seq.shape == (30, 4)
    """

    def __init__(self, oscar_dir: str):
        self.oscar_dir = Path(oscar_dir)
        self._ds = None

    def _load_dataset(self):
        """Lazily load all NetCDF files as a single xarray dataset."""
        if self._ds is not None:
            return

        nc_files = sorted(self.oscar_dir.glob("*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No .nc files found in {self.oscar_dir}")

        print(f"Loading {len(nc_files)} OSCAR NetCDF files...")
        self._ds = xr.open_mfdataset(
            nc_files,
            combine="by_coords",
            chunks={"time": 30},   # lazy load with dask chunks
        )
        print(f"OSCAR dataset loaded: {self._ds}")

    def get_sequence(
        self,
        lat: float,
        lon: float,
        date: str,
        window_days: int = 30,
    ) -> np.ndarray:
        """
        Extract a (window_days, 4) sequence of [U, V, lat, lon]
        ending on `date` for the nearest grid point to (lat, lon).

        Args:
            lat:         Latitude of MARIDA/MADOS tile center
            lon:         Longitude of MARIDA/MADOS tile center
            date:        ISO date string "YYYY-MM-DD" (tile acquisition date)
            window_days: Number of days to look back (default 30)

        Returns:
            np.ndarray of shape (window_days, 4), dtype float32
            Features: [U (m/s), V (m/s), lat (degrees), lon (degrees)]
            Missing timesteps filled with 0.0.
        """
        self._load_dataset()

        end_date   = pd.Timestamp(date)
        start_date = end_date - timedelta(days=window_days - 1)

        # Slice time window
        ds_window = self._ds.sel(
            time=slice(start_date, end_date),
            method=None,
        )

        # Nearest-neighbor spatial selection
        ds_point = ds_window.sel(
            latitude=lat,
            longitude=lon,
            method="nearest",
        )

        # Extract U and V arrays
        u_vals = ds_point["u"].values.flatten().astype(np.float32)
        v_vals = ds_point["v"].values.flatten().astype(np.float32)

        n_actual = len(u_vals)

        # Build output array — pad with zeros if fewer than window_days timesteps
        sequence = np.zeros((window_days, 4), dtype=np.float32)
        n_fill = min(n_actual, window_days)

        sequence[:n_fill, 0] = u_vals[:n_fill]           # U current
        sequence[:n_fill, 1] = v_vals[:n_fill]           # V current
        sequence[:n_fill, 2] = np.float32(lat)           # lat (constant)
        sequence[:n_fill, 3] = np.float32(lon)           # lon (constant)

        # Replace NaN (ocean gaps, coastlines) with 0.0
        sequence = np.nan_to_num(sequence, nan=0.0)

        return sequence


# ─── STEP 4: PRE-EXTRACT SEQUENCES FOR ALL MARIDA/MADOS TILES ─────────────────

def preextract_sequences(tile_index_path: str, extractor: OSCARSequenceExtractor):
    """
    Pre-extracts OSCAR sequences for every tile in MARIDA/MADOS and saves
    them as .npy files keyed by tile ID. This avoids repeated NetCDF seeks
    during training and speeds up DataLoader throughput significantly.

    tile_index_path: path to a CSV with columns: tile_id, lat, lon, date
    Output: data/processed/oscar_sequences/{tile_id}.npy

    CSV format example:
        tile_id,lat,lon,date
        S2_2019-06-15_32.1_-145.3,32.1,-145.3,2019-06-15
        S2_2018-09-02_20.0_-160.0,20.0,-160.0,2018-09-02
    """
    df = pd.read_csv(tile_index_path)
    print(f"\nPre-extracting OSCAR sequences for {len(df)} tiles...")

    skipped = 0
    errors  = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = SEQUENCE_DIR / f"{row['tile_id']}.npy"

        # Skip if already extracted
        if out_path.exists():
            skipped += 1
            continue

        try:
            seq = extractor.get_sequence(
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                date=str(row["date"]),
            )
            np.save(out_path, seq)
        except Exception as e:
            print(f"  ERROR {row['tile_id']}: {e}")
            errors += 1

    print(f"Done. Skipped (cached): {skipped} | Errors: {errors}")
    print(f"Sequences saved to: {SEQUENCE_DIR}/")


# ─── STEP 5: VALIDATION ────────────────────────────────────────────────────────

def validate_sequence(npy_path: str):
    """
    Quick sanity check on an extracted sequence file.
    Prints shape, value ranges, and NaN count.
    """
    seq = np.load(npy_path)
    print(f"\nFile: {npy_path}")
    print(f"  Shape:      {seq.shape}  (expected: (30, 4))")
    print(f"  U range:    [{seq[:, 0].min():.4f}, {seq[:, 0].max():.4f}] m/s")
    print(f"  V range:    [{seq[:, 1].min():.4f}, {seq[:, 1].max():.4f}] m/s")
    print(f"  Lat (const): {seq[0, 2]:.4f}")
    print(f"  Lon (const): {seq[0, 3]:.4f}")
    print(f"  NaN count:  {np.isnan(seq).sum()}")
    print(f"  Zero rows:  {(seq.sum(axis=1) == 0).sum()} (padded timesteps)")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and extract NOAA OSCAR sequences")
    parser.add_argument(
        "--mode",
        choices=["download", "extract", "validate"],
        default="download",
        help=(
            "download  — authenticate and pull OSCAR NetCDF files from PO.DAAC\n"
            "extract   — pre-extract 30-day sequences for all MARIDA/MADOS tiles\n"
            "validate  — sanity check a single extracted .npy sequence file"
        ),
    )
    parser.add_argument(
        "--tile-index",
        type=str,
        default="data/raw/marida/tile_index.csv",
        help="Path to tile index CSV (columns: tile_id, lat, lon, date)",
    )
    parser.add_argument(
        "--npy",
        type=str,
        default=None,
        help="Path to .npy file to validate (used with --mode validate)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Concurrent download workers (default 16)",
    )
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help=(
            "Targeted download: either a CSV path with a `date` column "
            "(e.g. data/raw/marida/tile_index.csv) or a comma-separated list "
            "of ISO dates (e.g. 2018-04-15,2018-07-22). When set, only "
            "[date - window-days, date] granules are fetched."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Lookback window for targeted mode (default 30)",
    )
    args = parser.parse_args()

    if args.mode == "download":
        auth = authenticate()
        date_list = _resolve_dates(args.dates) if args.dates else None
        if date_list:
            print(f"Targeted mode: {len(date_list)} unique dates, {args.window_days}-day window")
        download_oscar_files(
            auth,
            threads=args.threads,
            dates=date_list,
            window_days=args.window_days,
        )

    elif args.mode == "extract":
        extractor = OSCARSequenceExtractor(str(OUT_DIR))
        preextract_sequences(args.tile_index, extractor)

    elif args.mode == "validate":
        if not args.npy:
            # Auto-pick first available file for quick sanity check
            npy_files = list(SEQUENCE_DIR.glob("*.npy"))
            if not npy_files:
                print("No .npy files found. Run --mode extract first.")
            else:
                validate_sequence(str(npy_files[0]))
        else:
            validate_sequence(args.npy)
