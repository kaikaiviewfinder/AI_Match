#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a synthetic hiking route_table.csv for demo runs."""
import argparse
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from route_tools import compute_route_geometry, lla_to_local_xy, local_xy_to_lla, write_route_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(os.path.dirname(SCRIPT_DIR), "data", "demo_route.csv"))
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--length", type=float, default=1500.0)
    parser.add_argument("--origin-lon", type=float, default=113.953200)
    parser.add_argument("--origin-lat", type=float, default=22.530000)
    parser.add_argument("--origin-alt", type=float, default=120.0)
    args = parser.parse_args()

    raw = []
    # Create a smooth route with bends and altitude changes.
    # Raw points every 20 m; route_tools will resample to args.step.
    for i in range(int(args.length // 20) + 1):
        s = i * 20.0
        x = s * 0.78 + 25.0 * math.sin(s / 120.0)
        y = s * 0.42 + 40.0 * math.sin(s / 180.0)
        z = args.origin_alt + 0.06 * s + 12.0 * math.sin(s / 90.0) + 4.0 * math.sin(s / 25.0)
        lon, lat = local_xy_to_lla(x, y, args.origin_lon, args.origin_lat)
        raw.append((lon, lat, z, x, y))

    points, lon0, lat0 = compute_route_geometry(raw, step_m=args.step)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_route_csv(args.output, points, lon0, lat0)
    print(f"Wrote {len(points)} route points to {args.output}")
    print(f"Route length: {points[-1].s:.1f} m")


if __name__ == "__main__":
    main()
