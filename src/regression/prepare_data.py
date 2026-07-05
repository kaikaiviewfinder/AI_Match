"""Prepare training data for coordinate regression head.

Generates a JSON file mapping image_path → [lat, lng] from various sources.

Supported sources:
  1. Directory of geotagged images (EXIF GPS tags)
  2. Mapillary Vistas / MSV dataset
  3. Custom CSV with image_path, lat, lng columns
  4. Tuxun-style parquet files (coordinates only, no images — for reference)

Usage:
    # From geotagged images (EXIF GPS)
    python -m src.regression.prepare_data --source exif --image-dir data/images/ --output data/geotagged.json

    # From CSV
    python -m src.regression.prepare_data --source csv --csv data/images.csv --output data/geotagged.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def from_exif_gps(image_dir: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> dict:
    """Extract GPS coordinates from EXIF metadata in images.

    Many smartphone photos and some street view downloads include GPS EXIF data.
    """
    from PIL import Image
    from PIL.ExifTags import GPSTAGS

    data = {}
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(extensions):
            continue
        path = os.path.join(image_dir, fname)
        try:
            img = Image.open(path)
            exif = img._getexif()
            if not exif:
                continue

            gps_info = {}
            for tag, value in exif.items():
                tag_name = GPSTAGS.get(tag, tag)
                if tag_name in ("GPSInfo",):
                    for gps_tag, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_info[gps_tag_name] = gps_value

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
                lat = _exif_to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
                lng = _exif_to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    data[path] = [lat, lng]
        except Exception:
            continue

    print(f"Extracted GPS from {len(data)}/{sum(1 for _ in os.listdir(image_dir))} images in {image_dir}")
    return data


def _exif_to_decimal(dms_tuple, ref):
    """Convert EXIF (degrees, minutes, seconds) tuple to decimal degrees."""
    degrees, minutes, seconds = dms_tuple
    decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if ref.upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def from_csv(csv_path: str, image_base_dir: str = "") -> dict:
    """Load from a CSV file with columns: image_path, lat, lng."""
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get("image_path", row.get("path", ""))
            lat = float(row.get("lat", row.get("latitude", 0)))
            lng = float(row.get("lng", row.get("lon", row.get("longitude", 0))))
            if image_base_dir and not os.path.isabs(path):
                path = os.path.join(image_base_dir, path)
            if os.path.exists(path) and -90 <= lat <= 90 and -180 <= lng <= 180:
                data[path] = [lat, lng]
    print(f"Loaded {len(data)} entries from {csv_path}")
    return data


def from_json(json_path: str, image_base_dir: str = "") -> dict:
    """Load from JSON: {image_path: [lat, lng]} or [{path, lat, lng}]."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = {}
    if isinstance(raw, dict):
        for path, coords in raw.items():
            full_path = os.path.join(image_base_dir, path) if image_base_dir else path
            if os.path.exists(full_path):
                data[full_path] = coords[:2]
    elif isinstance(raw, list):
        for item in raw:
            path = item.get("image_path", item.get("path", ""))
            lat = item.get("lat", item.get("latitude", 0))
            lng = item.get("lng", item.get("lon", item.get("longitude", 0)))
            full_path = os.path.join(image_base_dir, path) if image_base_dir else path
            if os.path.exists(full_path) and -90 <= lat <= 90 and -180 <= lng <= 180:
                data[full_path] = [lat, lng]
    print(f"Loaded {len(data)} entries from {json_path}")
    return data


def check_tuxun_coverage(data_dir: str) -> None:
    """Print statistics about the tuxun dataset coordinates (no images needed)."""
    import glob as g
    import pandas as pd

    files = sorted(g.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        print(f"No parquet files found in {data_dir}")
        return

    import json as _json
    total_locs = 0
    lat_min, lat_max = 90, -90
    lng_min, lng_max = 180, -180

    for fp in files[:100]:  # sample first 100 files
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            try:
                obj = _json.loads(row["data"])
                for rnd in obj.get("rounds", []):
                    lat, lng = rnd.get("lat"), rnd.get("lng")
                    if lat and lng:
                        total_locs += 1
                        lat_min = min(lat_min, lat)
                        lat_max = max(lat_max, lat)
                        lng_min = min(lng_min, lng)
                        lng_max = max(lng_max, lng)
            except Exception:
                pass

    print(f"\nTuxun dataset ({len(files)} parquet files, sampled first 100):")
    print(f"  Locations found: {total_locs}")
    print(f"  Lat: [{lat_min:.2f}, {lat_max:.2f}]")
    print(f"  Lng: [{lng_min:.2f}, {lng_max:.2f}]")
    print(f"  Global coverage: {'Yes' if lat_max > 60 and lat_min < -40 else 'Partial'}")
    print(f"\n  NOTE: Tuxun has coordinates but NO images.")
    print(f"  To get training images, download Google Street View panoramas")
    print(f"  using panoIDs (VPN required within China).")
    print(f"  Alternatively, use Mapillary Vistas or Baidu Street View API.")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for regression head")
    parser.add_argument("--source", type=str, default="exif",
                        choices=["exif", "csv", "json", "tuxun-check"],
                        help="Data source type")
    parser.add_argument("--image-dir", type=str, default="primary test",
                        help="Directory containing images")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to CSV file (for --source csv)")
    parser.add_argument("--json", type=str, default="",
                        help="Path to JSON file (for --source json)")
    parser.add_argument("--tuxun-dir", type=str,
                        default="D:/AI_Cache/huggingface/hub/datasets--ShirohAO--tuxun/snapshots/1fcc0f3b1f6312ea0ad5c1837a9302af5cd7ff87/data",
                        help="Path to tuxun parquet files")
    parser.add_argument("--output", type=str, default="data/geotagged.json",
                        help="Output JSON file")
    args = parser.parse_args()

    if args.source == "tuxun-check":
        check_tuxun_coverage(args.tuxun_dir)
        return

    sources = {
        "exif": lambda: from_exif_gps(args.image_dir),
        "csv": lambda: from_csv(args.csv, args.image_dir),
        "json": lambda: from_json(args.json, args.image_dir),
    }

    data = sources[args.source]()
    if not data:
        print("WARNING: No valid image-coordinate pairs found.")
        print("For EXIF: make sure images have GPS metadata.")
        print("For CSV: ensure columns 'image_path,lat,lng' exist.")
        print("For JSON: ensure {image_path: [lat, lng]} format.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    lats = [v[0] for v in data.values()]
    lngs = [v[1] for v in data.values()]
    print(f"\nSaved {len(data)} entries to {args.output}")
    print(f"Lat: [{min(lats):.4f}, {max(lats):.4f}]")
    print(f"Lng: [{min(lngs):.4f}, {max(lngs):.4f}]")
    print(f"\nNext: python -m src.regression.train --data {args.output}")


if __name__ == "__main__":
    main()
