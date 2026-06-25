#!/usr/bin/env python3
"""Generate route_table.csv from rosbag GNSS data."""
import csv
import math
import subprocess
import sys
import os

sys.path.insert(0, "/home/kai/hiking_route_localization/scripts")
from route_tools import lla_to_local_xy, compute_route_geometry, write_route_csv

BAG = "/home/kai/AI_Data/Outdoor-1/outdoor_1.bag"
OUT = "/home/kai/AI_Data/Outdoor-1/route_table.csv"

# Extract GNSS from bag
print("Extracting GNSS from bag...")
result = subprocess.run(
    ["rostopic", "echo", "-b", BAG, "-p", "/gnss0"],
    capture_output=True, text=True
)
lines = result.stdout.strip().split("\n")
print(f"Got {len(lines)} lines (first is header)")

reader = csv.DictReader(lines)
pts = []
for row in reader:
    lat = float(row["field.latitude"])
    lon = float(row["field.longitude"])
    alt = float(row["field.altitude"])
    pts.append((lon, lat, alt))

if len(pts) < 2:
    print(f"ERROR: only {len(pts)} GNSS points")
    sys.exit(1)

print(f"Got {len(pts)} GNSS points, lat: {pts[0][1]:.6f}, lon: {pts[0][0]:.6f}, alt: {pts[0][2]:.1f}m")

# Convert to local xy and resample
lon0, lat0, _ = pts[0]
raw = []
for lon, lat, alt in pts:
    x, y = lla_to_local_xy(lon, lat, lon0, lat0)
    raw.append((lon, lat, alt, x, y))

route_points, lon0_out, lat0_out = compute_route_geometry(raw, step_m=2.0)
write_route_csv(OUT, route_points, lon0_out, lat0_out)
print(f"Wrote {len(route_points)} route points to {OUT}, length={route_points[-1].s:.1f}m")
