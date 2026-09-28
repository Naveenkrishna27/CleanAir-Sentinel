"""
Copies real citizen reports (CSV + referenced photos) into data/sample/,
so they ship with the deployed app as the offline-safe fallback tier -
same purpose as data_ingestion.py's save_sample_snapshot(), but for
citizen data instead of satellite data.

Run this any time after submitting new reports through the app, before
you commit or deploy:

    python save_citizen_sample.py

Safe to run repeatedly - each run rebuilds the sample snapshot from
whatever is currently in data/citizen_reports.csv. Your original photos
and exact coordinates in data/ are never modified.

Because the sample folder is committed to a public repo, this script
sanitizes what it copies:
  - Photos are re-saved WITHOUT metadata (phones embed GPS position,
    device model and timestamps in EXIF), and resized to at most 1280 px
    on the long side, which also keeps the repo and the deployed app light.
  - Coordinates are rounded to 3 decimals (about 110 m). That is still far
    finer than the 5 km hotspot grid, so results are unchanged, but it no
    longer pinpoints a home.
  - Paths always use forward slashes, because the app runs on Windows
    locally but on Linux when deployed.
"""

import os

import pandas as pd
from PIL import Image, ImageOps

DATA_DIR = "data"
CITIZEN_REPORTS_CSV = f"{DATA_DIR}/citizen_reports.csv"

SAMPLE_DIR = f"{DATA_DIR}/sample"
SAMPLE_PHOTOS_DIR = f"{SAMPLE_DIR}/citizen_photos"
SAMPLE_REPORTS_CSV = f"{SAMPLE_DIR}/citizen_reports.csv"

MAX_SIDE_PX = 1280
COORD_DECIMALS = 3


def _clean_copy(src: str, dest: str) -> None:
    """Re-save an image with no metadata, orientation baked in, and a
    bounded size. Copies pixels only, so EXIF/GPS cannot carry over."""
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)  # apply rotation before dropping EXIF
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
        clean = Image.new("RGB", img.size)
        clean.paste(img)
        clean.save(dest, format="JPEG", quality=88)


def save_citizen_sample():
    if not os.path.exists(CITIZEN_REPORTS_CSV):
        print(f"No {CITIZEN_REPORTS_CSV} found - submit at least one report through the app first.")
        return

    df = pd.read_csv(CITIZEN_REPORTS_CSV)
    if df.empty:
        print("citizen_reports.csv is empty - nothing to snapshot.")
        return

    os.makedirs(SAMPLE_PHOTOS_DIR, exist_ok=True)

    # Clear old sample photos so removed reports don't leave stale files
    # (and their metadata) behind in the repo.
    for old in os.listdir(SAMPLE_PHOTOS_DIR):
        old_path = os.path.join(SAMPLE_PHOTOS_DIR, old)
        if os.path.isfile(old_path):
            os.remove(old_path)

    rows, missing = [], 0
    for _, row in df.iterrows():
        # The app writes paths with a mix of / and \ on Windows; treat both
        # as separators so this works no matter how the row was saved.
        src = str(row["image_path"]).replace("\\", "/")
        stem = os.path.splitext(src.split("/")[-1])[0]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
        dest_name = f"{safe_stem}.jpg"

        if not os.path.exists(src):
            missing += 1
            print(f"  missing photo: {src}")
            continue

        try:
            _clean_copy(src, os.path.join(SAMPLE_PHOTOS_DIR, dest_name))
        except Exception as e:
            missing += 1
            print(f"  could not process {src}: {e}")
            continue

        rows.append({
            **row.to_dict(),
            "image_path": f"{SAMPLE_PHOTOS_DIR}/{dest_name}",
            "lat": round(float(row["lat"]), COORD_DECIMALS),
            "lon": round(float(row["lon"]), COORD_DECIMALS),
        })

    if not rows:
        print("No photos could be processed - sample snapshot not written.")
        return

    pd.DataFrame(rows).to_csv(SAMPLE_REPORTS_CSV, index=False)

    print(f"Saved {len(rows)} reports to {SAMPLE_REPORTS_CSV} (coordinates rounded to {COORD_DECIMALS} decimals)")
    print(f"Wrote {len(rows)} metadata-free photo(s) to {SAMPLE_PHOTOS_DIR}/")
    if missing:
        print(f"Skipped {missing} report(s) that could not be processed.")


if __name__ == "__main__":
    save_citizen_sample()