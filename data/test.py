# run from: data/
# pip install rasterio pandas tqdm

import rasterio
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

MARIDA_PATCHES = Path("raw/marida/patches")  # adjust if needed
OUT_CSV = Path("raw/marida/tile_index.csv")

records = []

patch_files = sorted(MARIDA_PATCHES.rglob("*.tif"))
# Only image patches, not masks (_cl) or confidence (_conf)
patch_files = [p for p in patch_files if "_cl" not in p.name and "_conf" not in p.name]

for tif in tqdm(patch_files, desc="Indexing MARIDA tiles"):
    try:
        # Parse date from parent folder: S2_DD-MM-YY_TILE
        folder = tif.parent.name          # e.g. S2_21-05-15_T36MXE
        parts  = folder.split("_")        # ['S2', '21-05-15', 'T36MXE']
        date_str = parts[1]               # '21-05-15'
        tile_id  = parts[2]              # 'T36MXE'

        # Parse DD-MM-YY → YYYY-MM-DD
        date = datetime.strptime(date_str, "%d-%m-%y").strftime("%Y-%m-%d")

        # Read center lat/lon from GeoTIFF CRS
        with rasterio.open(tif) as src:
            bounds = src.bounds
            cx = (bounds.left + bounds.right)  / 2
            cy = (bounds.top  + bounds.bottom) / 2
            # Transform to WGS84 if not already
            from rasterio.crs import CRS
            from pyproj import Transformer
            wgs84 = CRS.from_epsg(4326)
            if src.crs != wgs84:
                t = Transformer.from_crs(src.crs, wgs84, always_xy=True)
                cx, cy = t.transform(cx, cy)

        records.append({
            "tile_id": tif.stem,     # e.g. S2_21-05-15_T36MXE_0
            "date":    date,         # e.g. 2015-05-21
            "lat":     round(cy, 6),
            "lon":     round(cx, 6),
            "scene":   folder,
            "path":    str(tif),
        })

    except Exception as e:
        print(f"  SKIP {tif.name}: {e}")

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} tiles → {OUT_CSV}")
print(df.head())