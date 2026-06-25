#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common tools for hiking route localization.
Coordinate convention:
- Raw GPS: WGS84 longitude/latitude/altitude.
- Route matcher: local ENU-like metric frame in meters.
  x: east, y: north, z: altitude.
This lightweight implementation uses a local tangent-plane approximation.
For hiking-scale routes it is usually adequate; for long routes replace with GeographicLib/pyproj.
"""

import csv
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

EARTH_RADIUS_M = 6378137.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def wrap_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def lla_to_local_xy(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """Approximate WGS84 lon/lat to local x/y in meters."""
    lat_rad = math.radians(lat)
    lat0_rad = math.radians(lat0)
    dlon = math.radians(lon - lon0)
    dlat = math.radians(lat - lat0)
    x = EARTH_RADIUS_M * dlon * math.cos(0.5 * (lat_rad + lat0_rad))
    y = EARTH_RADIUS_M * dlat
    return x, y


def local_xy_to_lla(x: float, y: float, lon0: float, lat0: float) -> Tuple[float, float]:
    lat = lat0 + math.degrees(y / EARTH_RADIUS_M)
    lat_mid = math.radians(0.5 * (lat + lat0))
    lon = lon0 + math.degrees(x / (EARTH_RADIUS_M * max(math.cos(lat_mid), 1e-9)))
    return lon, lat


@dataclass
class RoutePoint:
    s: float
    x: float
    y: float
    z: float
    yaw: float
    slope: float
    curvature: float
    segment_id: int
    lon: float = 0.0
    lat: float = 0.0


class RouteTable:
    def __init__(self, points: List[RoutePoint], lon0: float = 0.0, lat0: float = 0.0):
        if len(points) < 2:
            raise ValueError("RouteTable needs at least 2 points")
        self.points = points
        self.lon0 = lon0
        self.lat0 = lat0
        self.length = points[-1].s

    @classmethod
    def load_csv(cls, path: str) -> "RouteTable":
        points: List[RoutePoint] = []
        lon0 = 0.0
        lat0 = 0.0
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                if row.get("type") == "origin":
                    lon0 = float(row.get("lon", 0.0))
                    lat0 = float(row.get("lat", 0.0))
                    continue
                try:
                    points.append(RoutePoint(
                        s=float(row["s"]),
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=float(row["z"]),
                        yaw=float(row.get("yaw", 0.0)),
                        slope=float(row.get("slope", 0.0)),
                        curvature=float(row.get("curvature", 0.0)),
                        segment_id=int(float(row.get("segment_id", len(points)))),
                        lon=float(row.get("lon", 0.0)),
                        lat=float(row.get("lat", 0.0)),
                    ))
                except KeyError as exc:
                    raise ValueError(f"route csv missing required column: {exc}") from exc
        if not points:
            raise ValueError(f"no route points loaded from {path}")
        # If origin metadata is missing but lon/lat exists, use first route point.
        if lon0 == 0.0 and lat0 == 0.0 and abs(points[0].lon) > 1e-12 and abs(points[0].lat) > 1e-12:
            lon0 = points[0].lon
            lat0 = points[0].lat
        return cls(points, lon0=lon0, lat0=lat0)

    def interpolate(self, s: float) -> RoutePoint:
        s = clamp(s, 0.0, self.length)
        pts = self.points
        # Fast path for near integer-meter routes.
        lo, hi = 0, len(pts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if pts[mid].s < s:
                lo = mid + 1
            else:
                hi = mid
        idx = max(1, lo)
        p0 = pts[idx - 1]
        p1 = pts[idx]
        ds = p1.s - p0.s
        u = 0.0 if abs(ds) < 1e-9 else (s - p0.s) / ds
        u = clamp(u, 0.0, 1.0)
        yaw = p0.yaw + wrap_angle_rad(p1.yaw - p0.yaw) * u
        return RoutePoint(
            s=s,
            x=p0.x + (p1.x - p0.x) * u,
            y=p0.y + (p1.y - p0.y) * u,
            z=p0.z + (p1.z - p0.z) * u,
            yaw=yaw,
            slope=p0.slope + (p1.slope - p0.slope) * u,
            curvature=p0.curvature + (p1.curvature - p0.curvature) * u,
            segment_id=p0.segment_id if u < 0.5 else p1.segment_id,
            lon=p0.lon + (p1.lon - p0.lon) * u,
            lat=p0.lat + (p1.lat - p0.lat) * u,
        )

    def nearest_by_xy(self, x: float, y: float, s_hint: Optional[float] = None, radius_m: float = 100.0) -> Tuple[float, float]:
        """Return nearest route s and squared xy distance."""
        best_s = 0.0
        best_d2 = float("inf")
        if s_hint is None:
            indices = range(len(self.points))
        else:
            lo_s = max(0.0, s_hint - radius_m)
            hi_s = min(self.length, s_hint + radius_m)
            indices = [i for i, p in enumerate(self.points) if lo_s <= p.s <= hi_s]
            if not indices:
                indices = range(len(self.points))
        for i in indices:
            p = self.points[i]
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = p.s
        return best_s, best_d2

    def window_indices(self, s_center: float, radius_m: float) -> List[int]:
        lo_s = max(0.0, s_center - radius_m)
        hi_s = min(self.length, s_center + radius_m)
        return [i for i, p in enumerate(self.points) if lo_s <= p.s <= hi_s]


def compute_route_geometry(raw_points: List[Tuple[float, float, float, float, float]], step_m: float = 1.0) -> Tuple[List[RoutePoint], float, float]:
    """
    raw_points: list of (lon, lat, alt, x, y).
    Returns resampled route points and origin lon/lat.
    """
    if len(raw_points) < 2:
        raise ValueError("need at least 2 raw route points")
    lon0, lat0 = raw_points[0][0], raw_points[0][1]
    # cumulative distance on raw polyline
    cum = [0.0]
    for i in range(1, len(raw_points)):
        _, _, z0, x0, y0 = raw_points[i - 1]
        _, _, z1, x1, y1 = raw_points[i]
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        cum.append(cum[-1] + dist)
    total = cum[-1]
    if total <= 1e-6:
        raise ValueError("route length is too small")

    s_values = []
    s = 0.0
    while s < total:
        s_values.append(s)
        s += step_m
    if s_values[-1] < total:
        s_values.append(total)

    resampled = []
    j = 1
    for s in s_values:
        while j < len(cum) - 1 and cum[j] < s:
            j += 1
        s0, s1 = cum[j - 1], cum[j]
        u = 0.0 if abs(s1 - s0) < 1e-9 else (s - s0) / (s1 - s0)
        u = clamp(u, 0.0, 1.0)
        lon_a, lat_a, z_a, x_a, y_a = raw_points[j - 1]
        lon_b, lat_b, z_b, x_b, y_b = raw_points[j]
        lon = lon_a + (lon_b - lon_a) * u
        lat = lat_a + (lat_b - lat_a) * u
        x = x_a + (x_b - x_a) * u
        y = y_a + (y_b - y_a) * u
        z = z_a + (z_b - z_a) * u
        resampled.append([s, x, y, z, lon, lat])

    out: List[RoutePoint] = []
    for i, p in enumerate(resampled):
        s, x, y, z, lon, lat = p
        if i == 0:
            nx, ny, nz = resampled[min(i + 1, len(resampled) - 1)][1:4]
            px, py, pz = x, y, z
        else:
            px, py, pz = resampled[i - 1][1:4]
            nx, ny, nz = resampled[min(i + 1, len(resampled) - 1)][1:4]
        yaw = math.atan2(ny - py, nx - px)
        horiz = math.sqrt((nx - px) ** 2 + (ny - py) ** 2)
        slope = 0.0 if horiz < 1e-9 else (nz - pz) / horiz
        # approximate curvature from heading change per meter
        if 0 < i < len(resampled) - 1:
            yaw_prev = math.atan2(y - resampled[i - 1][2], x - resampled[i - 1][1])
            yaw_next = math.atan2(resampled[i + 1][2] - y, resampled[i + 1][1] - x)
            ds = max(resampled[i + 1][0] - resampled[i - 1][0], 1e-6)
            curvature = abs(wrap_angle_rad(yaw_next - yaw_prev)) / ds
        else:
            curvature = 0.0
        out.append(RoutePoint(s=s, x=x, y=y, z=z, yaw=yaw, slope=slope, curvature=curvature, segment_id=i, lon=lon, lat=lat))
    return out, lon0, lat0


def write_route_csv(path: str, points: List[RoutePoint], lon0: float, lat0: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["type", "s", "x", "y", "z", "yaw", "slope", "curvature", "segment_id", "lon", "lat"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"type": "origin", "s": "", "x": "", "y": "", "z": "", "yaw": "", "slope": "", "curvature": "", "segment_id": "", "lon": lon0, "lat": lat0})
        for p in points:
            writer.writerow({
                "type": "point",
                "s": f"{p.s:.3f}",
                "x": f"{p.x:.3f}",
                "y": f"{p.y:.3f}",
                "z": f"{p.z:.3f}",
                "yaw": f"{p.yaw:.8f}",
                "slope": f"{p.slope:.8f}",
                "curvature": f"{p.curvature:.8f}",
                "segment_id": p.segment_id,
                "lon": f"{p.lon:.9f}",
                "lat": f"{p.lat:.9f}",
            })
