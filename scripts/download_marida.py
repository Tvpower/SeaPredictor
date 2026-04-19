"""Download and extract the MARIDA dataset from Zenodo.

Usage:
    python scripts/download_marida.py

Requires: pip install requests tqdm
The dataset is ~4.5 GB. Extraction lands at data/raw/MARIDA/.
"""
import sys
import hashlib
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run:  pip install requests tqdm")
    sys.exit(1)

# Zenodo record 5151941 — MARIDA v1.0
ZENODO_URL = "https://zenodo.org/record/5151941/files/MARIDA.zip"
DEST_ZIP   = Path("data/raw/MARIDA.zip")
DEST_DIR   = Path("data/raw")

DEST_DIR.mkdir(parents=True, exist_ok=True)

if (DEST_DIR / "MARIDA" / "patches").exists():
    print("MARIDA already extracted at data/raw/MARIDA/. Nothing to do.")
    sys.exit(0)

print(f"Downloading MARIDA from Zenodo (~4.5 GB)...")
print(f"  -> {DEST_ZIP}")

with requests.get(ZENODO_URL, stream=True, timeout=60) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(DEST_ZIP, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="MARIDA.zip"
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))

print(f"\nExtracting to {DEST_DIR}/...")
with zipfile.ZipFile(DEST_ZIP, "r") as zf:
    members = zf.namelist()
    for member in tqdm(members, desc="Extracting"):
        zf.extract(member, DEST_DIR)

print(f"Removing zip...")
DEST_ZIP.unlink()

patches = DEST_DIR / "MARIDA" / "patches"
if patches.exists():
    count = sum(1 for _ in patches.glob("**/*.tif"))
    print(f"\nDone. {count} .tif files at data/raw/MARIDA/patches/")
else:
    print("\nWARNING: expected data/raw/MARIDA/patches/ not found after extraction.")
    print("Check the zip contents manually.")
