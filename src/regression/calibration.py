"""Conformal Prediction calibration for VLM element outputs.

Replaces hand-tuned sensor-consistency thresholds with empirically
calibrated quantiles.  Uses GPS-derived pseudo-ground-truth (DEM
elevation, climate lookup) as reference, so no manual labels needed.

Mechanism (per the plan's Step 2):
  1. On calibration set: compute nonconformity score per element
     s = 1 - match(VLM_output, GPS_derived_truth)
  2. Compute quantile q = ceil((n+1)*(1-alpha)) / n
  3. At inference: if s <= q → inside conformal set → normal weight
     if s > q → outside → mark unreliable, sensor substitute
"""

import math, json
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Nonconformity score functions
# ---------------------------------------------------------------------------

def _gaussian_nonconformity(sensor_val: float, lo: float, hi: float,
                            sigma_fraction: float = 0.35) -> float:
    """1 - gaussian consistency: 0 = perfect match, 1 = far outside range."""
    sigma = max(abs((lo + hi) / 2) * sigma_fraction, 1.0)
    mid = (lo + hi) / 2
    diff = sensor_val - mid
    consistency = math.exp(-0.5 * (diff / sigma) ** 2)
    return 1.0 - consistency


def elevation_nonconformity(vlm_range: list, dem_elev: float) -> float:
    """Nonconformity for elevation_estimate_m vs DEM ground truth."""
    if not vlm_range or len(vlm_range) != 2:
        return 1.0  # maximum nonconformity if VLM gave no estimate
    v_lo, v_hi = float(vlm_range[0]), float(vlm_range[1])
    v_mid = (v_lo + v_hi) / 2
    sigma = max(abs(dem_elev) * 0.15, 50.0)
    diff = dem_elev - v_mid
    consistency = math.exp(-0.5 * (diff / sigma) ** 2)
    return 1.0 - consistency


def climate_nonconformity(vlm_climate: str, true_temp: float,
                          true_humid: float) -> float:
    """Nonconformity for climate_zone vs GPS-derived temp/humidity."""
    from scoring import _CLIMATE_SENSOR_RANGES
    val = str(vlm_climate).lower().replace(" ", "_")
    ranges = _CLIMATE_SENSOR_RANGES.get(val)
    if not ranges:
        return 0.5  # unknown climate zone → moderate nonconformity
    t_lo, t_hi, h_lo, h_hi = ranges
    nc_t = _gaussian_nonconformity(true_temp, t_lo, t_hi)
    nc_h = _gaussian_nonconformity(true_humid, h_lo, h_hi)
    return 1.0 - (1.0 - nc_t) * (1.0 - nc_h)  # combined nonconformity


def terrain_nonconformity(vlm_terrain: str, dem_elev: float) -> float:
    """Nonconformity for terrain_type vs DEM elevation."""
    from scoring import _TERRAIN_ELEV_RANGES
    val = str(vlm_terrain).lower().replace(" ", "_")
    ranges = _TERRAIN_ELEV_RANGES.get(val)
    if not ranges:
        return 0.5
    e_lo, e_hi = ranges
    return _gaussian_nonconformity(dem_elev, e_lo, e_hi)


def vegetation_nonconformity(vlm_veg: str, dem_elev: float,
                             true_temp: float) -> float:
    """Nonconformity for vegetation_zone vs elevation + temperature."""
    from scoring import _VEGETATION_SENSOR_RANGES
    val = str(vlm_veg).lower().replace(" ", "_")
    ranges = _VEGETATION_SENSOR_RANGES.get(val)
    if not ranges:
        return 0.5
    e_lo, e_hi, t_lo, t_hi = ranges
    nc_e = _gaussian_nonconformity(dem_elev, e_lo, e_hi)
    nc_t = _gaussian_nonconformity(true_temp, t_lo, t_hi)
    return 1.0 - (1.0 - nc_e) * (1.0 - nc_t)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class ConformalCalibrator:
    """Per-element conformal prediction calibrator.

    Usage:
        cal = ConformalCalibrator(alpha=0.05)
        cal.fit(calibration_records)   # list of {vlm_output, gps_truth, dem, climate}
        cal.save("output/calibration/conformal.json")

        # At inference:
        result = cal.check(vlm_elements, dem_elev, sensor_temp, sensor_humid)
        # result = {element_key: {"in_set": bool, "nonconformity": float, "threshold": float}}
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.thresholds: dict[str, float] = {}  # element_type → nonconformity quantile
        self.calibrated_sets: dict[str, set] = {}  # element_type → set of "inside" values
        self.n_samples: dict[str, int] = {}
        self._scores: dict[str, list[float]] = defaultdict(list)

    def fit(self, records: list[dict]):
        """Fit calibration thresholds from records.

        Each record should have:
          - vlm_elements: dict of element category → VLM-predicted value
          - dem_elev: float (ground-truth elevation from GPS → DEM lookup)
          - true_temp: float (GPS-derived July mean temperature)
          - true_humid: float (GPS-derived July mean humidity)
        """
        # Accumulate nonconformity scores per element type
        for rec in records:
            vlm = rec.get("vlm_elements", {})
            dem_elev = rec.get("dem_elev")
            true_temp = rec.get("true_temp")
            true_humid = rec.get("true_humid")

            for cat, val in vlm.items():
                if val is None or val == "":
                    continue
                val_key = tuple(val) if isinstance(val, list) else str(val)
                nc = None

                if cat == "elevation_estimate_m":
                    if dem_elev is not None and isinstance(val, list):
                        nc = elevation_nonconformity(val, dem_elev)
                elif cat == "climate_zone":
                    if true_temp is not None and true_humid is not None:
                        nc = climate_nonconformity(str(val), true_temp, true_humid)
                elif cat == "terrain_type":
                    if dem_elev is not None:
                        nc = terrain_nonconformity(str(val), dem_elev)
                elif cat == "vegetation_zone":
                    if dem_elev is not None and true_temp is not None:
                        nc = vegetation_nonconformity(str(val), dem_elev, true_temp)
                else:
                    nc = 0.0  # no sensor reference → assume conforming

                if nc is not None:
                    self._scores[cat].append(nc)

        # Compute quantile thresholds
        for cat, scores in self._scores.items():
            n = len(scores)
            if n < 2:
                continue
            scores_sorted = sorted(scores)
            # Quantile: ceil((n+1)*(1-alpha)) / n
            q_idx = int(math.ceil((n + 1) * (1.0 - self.alpha))) - 1
            q_idx = max(0, min(q_idx, n - 1))
            self.thresholds[cat] = scores_sorted[q_idx]
            self.n_samples[cat] = n

        # Build calibrated sets: values whose scores <= threshold
        for cat, scores in self._scores.items():
            records_for_cat = [
                r for r in records
                if cat in r.get("vlm_elements", {})
            ]
            self.calibrated_sets[cat] = set()
            for rec, nc in zip(records_for_cat, scores):
                if nc <= self.thresholds.get(cat, 1.0):
                    val = rec["vlm_elements"][cat]
                    val_key = tuple(val) if isinstance(val, list) else str(val)
                    self.calibrated_sets[cat].add(val_key)

    def check(self, vlm_elements: dict, dem_elev: Optional[float] = None,
              sensor_temp: Optional[float] = None,
              sensor_humid: Optional[float] = None) -> dict:
        """Check each VLM element against calibrated thresholds.

        Returns dict mapping (cat, val) → {
            "in_set": True if conformal, False if outside,
            "nonconformity": float,
            "threshold": float,
            "n_calibration": int,
        }
        """
        results = {}
        for cat, val in vlm_elements.items():
            if val is None or val == "":
                continue

            val_key = tuple(val) if isinstance(val, list) else str(val)

            # Compute nonconformity
            nc = None
            if cat == "elevation_estimate_m":
                if dem_elev is not None and isinstance(val, list):
                    nc = elevation_nonconformity(val, dem_elev)
            elif cat == "climate_zone":
                if sensor_temp is not None and sensor_humid is not None:
                    nc = climate_nonconformity(str(val), sensor_temp, sensor_humid)
            elif cat == "terrain_type":
                if dem_elev is not None:
                    nc = terrain_nonconformity(str(val), dem_elev)
            elif cat == "vegetation_zone":
                if dem_elev is not None and sensor_temp is not None:
                    nc = vegetation_nonconformity(str(val), dem_elev, sensor_temp)
            else:
                nc = 0.0

            threshold = self.thresholds.get(cat, 0.5)
            in_set = (nc is not None and nc <= threshold) if nc is not None else True

            # Check against calibrated set as fallback
            if cat in self.calibrated_sets and self.calibrated_sets[cat]:
                in_set = in_set or (val_key in self.calibrated_sets[cat])

            results[(cat, val_key)] = {
                "in_set": in_set,
                "nonconformity": round(nc, 3) if nc is not None else None,
                "threshold": round(threshold, 3),
                "n_calibration": self.n_samples.get(cat, 0),
            }

        return results

    def save(self, path: str):
        """Save calibration to JSON."""
        out = {
            "alpha": self.alpha,
            "thresholds": {k: round(v, 4) for k, v in self.thresholds.items()},
            "n_samples": self.n_samples,
            "calibrated_sets": {k: list(v) for k, v in self.calibrated_sets.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrator":
        """Load calibration from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cal = cls(alpha=data["alpha"])
        cal.thresholds = data["thresholds"]
        cal.n_samples = data["n_samples"]
        cal.calibrated_sets = {k: set(v) for k, v in data.get("calibrated_sets", {}).items()}
        return cal

    @property
    def summary(self) -> str:
        lines = [f"ConformalCalibrator (α={self.alpha})"]
        for cat in sorted(self.thresholds.keys()):
            lines.append(
                f"  {cat}: threshold={self.thresholds[cat]:.3f}, "
                f"n={self.n_samples.get(cat, 0)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility: build calibration records from GPS-tagged images
# ---------------------------------------------------------------------------

def build_calibration_records(
    image_records: list[dict],
    dem,
    climate,
    vlm_predictions: dict[str, dict],
) -> list[dict]:
    """Build calibration records with GPS-derived pseudo-ground-truth.

    Args:
        image_records: list of {image_path, lat, lng} from merged_geotagged
        dem: DEM instance with .query(lat, lng) → elevation
        climate: ClimateValidator instance
        vlm_predictions: dict mapping image_path → VLM output elements dict

    Returns:
        List of calibration records suitable for ConformalCalibrator.fit()
    """
    records = []
    for rec in image_records:
        path = rec.get("image") or rec.get("image_path")
        lat = rec.get("lat") or rec.get("latitude")
        lng = rec.get("lng") or rec.get("longitude")
        if not path or lat is None or lng is None:
            continue

        dem_elev = dem.query(lat, lng) if dem else None
        t_lo, t_hi = (None, None)
        h_lo, h_hi = (None, None)
        if climate:
            elev = dem_elev if dem_elev is not None else 500
            t_lo, t_hi = climate.estimate_temperature_range(lat, lng, elev, month=7)
            h_lo, h_hi = climate.estimate_humidity_range(lat, lng, month=7)

        true_temp = (t_lo + t_hi) / 2 if t_lo is not None else None
        true_humid = (h_lo + h_hi) / 2 if h_lo is not None else None

        vlm_elements = vlm_predictions.get(path, {})

        records.append({
            "image": path,
            "lat": lat, "lng": lng,
            "dem_elev": dem_elev,
            "true_temp": true_temp,
            "true_humid": true_humid,
            "vlm_elements": vlm_elements,
        })
    return records
