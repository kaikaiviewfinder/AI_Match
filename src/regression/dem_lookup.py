"""China DEM lookup using precomputed elevation grid.

Provides fast elevation queries for geographic constraint fusion.
Uses a 0.05-degree (~5.5km) grid stored as compressed numpy array.
When real SRTM data becomes available, replace the grid file and
this module works without API changes.
"""

import os
import math
from typing import Optional, Tuple

import numpy as np


class ChinaDEM:
    """Elevation lookup for China at ~5.5km resolution.

    Loads a precomputed .npz grid covering 18-54N, 73-135E.
    Memory footprint: ~3.5 MB for the float32 grid (721x1241).
    """

    def __init__(self, grid_path: Optional[str] = None):
        if grid_path is None:
            grid_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "data", "dem", "china_elevation_005deg.npz"
            )
        data = np.load(grid_path)
        self.grid: np.ndarray = data["elevation"]
        self.lat_min: float = float(data["lat_min"])
        self.lat_max: float = float(data["lat_max"])
        self.lng_min: float = float(data["lng_min"])
        self.lng_max: float = float(data["lng_max"])
        self.res_deg: float = float(data["res_deg"])
        self._loaded = True

    def _lat_to_row(self, lat: float) -> float:
        """Convert latitude to fractional grid row index."""
        return (lat - self.lat_min) / self.res_deg

    def _lng_to_col(self, lng: float) -> float:
        """Convert longitude to fractional grid column index."""
        return (lng - self.lng_min) / self.res_deg

    def query(self, lat: float, lng: float) -> Optional[float]:
        """Bilinear-interpolated elevation at a point. Returns None if out of bounds."""
        r = self._lat_to_row(lat)
        c = self._lng_to_col(lng)

        r0 = int(math.floor(r))
        c0 = int(math.floor(c))
        r1 = r0 + 1
        c1 = c0 + 1

        n_rows, n_cols = self.grid.shape
        if r0 < 0 or r1 >= n_rows or c0 < 0 or c1 >= n_cols:
            return None

        dr = r - r0
        dc = c - c0

        # Bilinear interpolation
        top = self.grid[r0, c0] * (1 - dc) + self.grid[r0, c1] * dc
        bot = self.grid[r1, c0] * (1 - dc) + self.grid[r1, c1] * dc
        return float(top * (1 - dr) + bot * dr)

    def query_nearest(self, lat: float, lng: float) -> Optional[float]:
        """Nearest-neighbor elevation query. Faster, no interpolation."""
        r = int(round(self._lat_to_row(lat)))
        c = int(round(self._lng_to_col(lng)))
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            return float(self.grid[r, c])
        return None

    def bbox_elevation_range(self, lat_min: float, lat_max: float,
                             lng_min: float, lng_max: float) -> Tuple[float, float]:
        """Min/max elevation within a bounding box.

        Samples the grid at regular intervals (every ~2 cells = ~10km)
        plus corners and center.
        """
        r0 = max(0, int(math.floor(self._lat_to_row(lat_min))))
        r1 = min(self.grid.shape[0] - 1, int(math.ceil(self._lat_to_row(lat_max))))
        c0 = max(0, int(math.floor(self._lng_to_col(lng_min))))
        c1 = min(self.grid.shape[1] - 1, int(math.ceil(self._lng_to_col(lng_max))))

        if r0 > r1 or c0 > c1:
            return (0.0, 0.0)

        # Stride: sample every ~2 cells for performance
        r_step = max(1, (r1 - r0) // 10 or 1)
        c_step = max(1, (c1 - c0) // 10 or 1)
        sub = self.grid[r0:r1 + 1:r_step, c0:c1 + 1:c_step]
        return (float(sub.min()), float(sub.max()))

    def bbox_matches_elevation(self, lat_min: float, lat_max: float,
                                lng_min: float, lng_max: float,
                                target_elevation_m: float,
                                tolerance_pct: float = 0.35) -> bool:
        """Check if a bbox is compatible with a target elevation.

        Returns True if ANY sampled point in the bbox is within
        tolerance_pct of the target elevation.
        """
        elev_min, elev_max = self.bbox_elevation_range(lat_min, lat_max, lng_min, lng_max)
        lo = target_elevation_m * (1 - tolerance_pct)
        hi = target_elevation_m * (1 + tolerance_pct)
        # Overlap check: elevation range of bbox overlaps with acceptable range
        return elev_min <= hi and elev_max >= lo

    def find_elevation_range(
        self,
        target_elevation_m: float,
        tolerance_pct: float = 0.35,
        sample_step: int = 5,
        min_region_cells: int = 10,
    ) -> list:
        """Reverse lookup: find all grid regions matching a target elevation.

        Given a sensor elevation reading (e.g. barometer), find where in
        China this elevation could occur. This is the key "sensor-first"
        spatial pruning step.

        Algorithm:
          1. Compute acceptable range with adaptive tolerance:
             - Relative: ±tolerance_pct (e.g. 4500m ±35% = 2925-6075m)
             - Absolute floor: ±200m (10m reading → 0-210m, not 6.5-13.5m)
          2. Subsample grid at sample_step interval
          3. Create boolean mask of compatible cells
          4. 8-connected BFS flood fill for component clustering
          5. Filter clusters below min_region_cells
          6. Convert grid indices back to lat/lng BBox objects
          7. Sort by area descending

        Returns:
            List of BBox objects representing compatible elevation regions.
            May be empty if the elevation reading is very unusual for China.
        """
        from collections import deque

        # Adaptive tolerance: relative for high elevations, absolute floor for low
        abs_floor = 200.0
        elev_lo = min(target_elevation_m * (1.0 - tolerance_pct),
                       target_elevation_m - abs_floor)
        elev_hi = max(target_elevation_m * (1.0 + tolerance_pct),
                       target_elevation_m + abs_floor)

        # Subsample the grid
        lat_step = max(sample_step, 1)
        lng_step = max(sample_step, 1)
        subsampled = self.grid[::lat_step, ::lng_step]

        # Compatible cell mask
        mask = (subsampled >= elev_lo) & (subsampled <= elev_hi)

        if not mask.any():
            return []

        # 8-connected BFS clustering
        visited = set()
        regions = []
        n_rows, n_cols = mask.shape

        for r0 in range(n_rows):
            for c0 in range(n_cols):
                if not mask[r0, c0] or (r0, c0) in visited:
                    continue
                q = deque([(r0, c0)])
                visited.add((r0, c0))
                r_min, r_max = r0, r0
                c_min, c_max = c0, c0
                while q:
                    r, c = q.popleft()
                    r_min, r_max = min(r_min, r), max(r_max, r)
                    c_min, c_max = min(c_min, c), max(c_max, c)
                    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n_rows and 0 <= nc < n_cols:
                            if mask[nr, nc] and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                q.append((nr, nc))

                region_cells = (r_max - r_min + 1) * (c_max - c_min + 1)
                if region_cells < min_region_cells:
                    continue

                lat_min = self.lat_min + r_min * lat_step * self.res_deg
                lat_max = self.lat_min + (r_max + 1) * lat_step * self.res_deg
                lng_min = self.lng_min + c_min * lng_step * self.res_deg
                lng_max = self.lng_min + (c_max + 1) * lng_step * self.res_deg

                from .geo_kb import BBox
                regions.append(BBox(
                    lat_min=max(lat_min, 17.0), lat_max=min(lat_max, 55.0),
                    lng_min=max(lng_min, 72.0), lng_max=min(lng_max, 136.0),
                    label=f"elevation_{target_elevation_m:.0f}m",
                ))

        if not regions:
            # Fallback: single bbox covering all compatible cells
            rows, cols = mask.nonzero()
            if len(rows) > 0:
                r_min, r_max = rows.min(), rows.max()
                c_min, c_max = cols.min(), cols.max()
                lat_min = self.lat_min + r_min * lat_step * self.res_deg
                lat_max = self.lat_min + (r_max + 1) * lat_step * self.res_deg
                lng_min = self.lng_min + c_min * lng_step * self.res_deg
                lng_max = self.lng_min + (c_max + 1) * lng_step * self.res_deg
                from .geo_kb import BBox
                regions.append(BBox(
                    lat_min=lat_min, lat_max=lat_max,
                    lng_min=lng_min, lng_max=lng_max,
                    label=f"elevation_{target_elevation_m:.0f}m",
                ))

        # Sort by area descending
        regions.sort(key=lambda b: b.area_km2, reverse=True)
        return regions


# Singleton instance
_dem_instance: Optional[ChinaDEM] = None


def get_dem() -> Optional[ChinaDEM]:
    """Get the shared ChinaDEM instance. Returns None if grid file is missing."""
    global _dem_instance
    if _dem_instance is None:
        grid_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "data", "dem", "china_elevation_005deg.npz"
        )
        if os.path.exists(grid_path):
            _dem_instance = ChinaDEM(grid_path)
        else:
            return None
    return _dem_instance
