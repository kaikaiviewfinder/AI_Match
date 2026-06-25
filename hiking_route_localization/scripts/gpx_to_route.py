#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert GPX route/track into a resampled route_table.csv."""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from route_tools import compute_route_geometry, lla_to_local_xy, write_route_csv


def parse_gpx_points(path):
    tree = ET.parse(path)
    root = tree.getroot()
    pts = []
    # GPX can have namespaces; handle by tag suffix.
    for elem in root.iter():
        if elem.tag.endswith("trkpt") or elem.tag.endswith("rtept") or elem.tag.endswith("wpt"):
            lat = float(elem.attrib["lat"])
            lon = float(elem.attrib["lon"])
            ele = None
            for child in elem:
                if child.tag.endswith("ele") and child.text is not None:
                    ele = float(child.text.strip())
                    break
            if ele is None:
                ele = 0.0
            pts.append((lon, lat, ele))
    return pts


def main():
    parser = argparse.ArgumentParser(description="Convert GPX to route_table.csv")
    parser.add_argument("input_gpx", help="Input .gpx file with track points and optional elevation")
    parser.add_argument("--output", default="route_table.csv", help="Output route csv")
    parser.add_argument("--step", type=float, default=1.0, help="Resample interval in meters")
    args = parser.parse_args()

    lla = parse_gpx_points(args.input_gpx)
    if len(lla) < 2:
        raise RuntimeError("GPX has fewer than 2 track/route points")
    lon0, lat0, _ = lla[0]
    raw = []
    for lon, lat, alt in lla:
        x, y = lla_to_local_xy(lon, lat, lon0, lat0)
        raw.append((lon, lat, alt, x, y))
    points, lon0, lat0 = compute_route_geometry(raw, step_m=args.step)
    write_route_csv(args.output, points, lon0, lat0)
    print(f"Wrote {len(points)} points to {args.output}; length={points[-1].s:.1f} m")


if __name__ == "__main__":
    main()
