"""Climate validator for sensor fusion.

Estimates expected temperature and humidity at any location in China
using physically-grounded models (latitude, elevation, continentality).
Used to validate candidate bboxes against real sensor readings.

Replace with gridded climate data (e.g. WorldClim) when available.
"""

import math
from typing import Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Climate estimation model
# ═══════════════════════════════════════════════════════════════════════════

# Base sea-level temperature (°C) by latitude band and month
# Rough monthly means for 20-50N, East Asia
_BASE_TEMP_TABLE = {
    # lat_band: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
    20: [18, 19, 22, 26, 28, 30, 31, 30, 28, 25, 22, 19],   # Hainan / S Guangdong
    25: [10, 12, 16, 21, 25, 28, 30, 29, 26, 22, 17, 12],   # Guilin / Kunming
    30: [5,  7,  12, 18, 22, 26, 29, 28, 24, 18, 13, 7],    # Shanghai / Chengdu
    35: [0,  3,  8,  15, 20, 25, 27, 26, 21, 15, 8,  2],    # Beijing / Xi'an
    40: [-8, -4, 3,  12, 18, 23, 25, 24, 18, 10, 1, -6],    # Shenyang / Hohhot
    45: [-18,-13,-3, 7,  14, 20, 23, 21, 14, 5, -7, -15],   # Harbin
    50: [-25,-20,-8, 3,  11, 18, 21, 18, 10, 0, -13,-22],   # N Heilongjiang
}

# Monthly humidity (%RH) by region type
_HUMIDITY_BY_REGION = {
    "coastal_south": [78, 80, 82, 83, 84, 85, 83, 83, 80, 76, 74, 75],
    "coastal_east":  [68, 70, 72, 74, 76, 80, 83, 82, 78, 74, 70, 68],
    "inland_south":  [72, 74, 74, 75, 76, 78, 77, 78, 76, 75, 74, 73],
    "inland_central": [55, 58, 60, 62, 65, 68, 75, 76, 72, 66, 60, 56],
    "inland_north":   [45, 48, 50, 48, 50, 55, 65, 68, 62, 56, 50, 47],
    "arid_nw":        [35, 35, 30, 28, 28, 30, 32, 30, 30, 32, 35, 35],
    "tibet_plateau":  [25, 25, 28, 30, 32, 38, 45, 45, 40, 32, 28, 25],
    "ne_china":       [58, 55, 52, 50, 52, 60, 72, 74, 68, 60, 58, 58],
}

LAPSE_RATE = 6.5  # °C per 1000m elevation
ELEV_UNCERTAINTY = 500  # m — DEM resolution uncertainty affects temp estimate


def _get_base_temp(lat: float, month: int) -> float:
    """Interpolate base sea-level temperature for a latitude and month."""
    bands = sorted(_BASE_TEMP_TABLE.keys())
    month_idx = max(0, min(11, month - 1))

    # Find bracketing latitude bands
    if lat <= bands[0]:
        return _BASE_TEMP_TABLE[bands[0]][month_idx]
    if lat >= bands[-1]:
        return _BASE_TEMP_TABLE[bands[-1]][month_idx]

    for i in range(len(bands) - 1):
        lo, hi = bands[i], bands[i + 1]
        if lo <= lat <= hi:
            t_lo = _BASE_TEMP_TABLE[lo][month_idx]
            t_hi = _BASE_TEMP_TABLE[hi][month_idx]
            frac = (lat - lo) / (hi - lo)
            return t_lo + frac * (t_hi - t_lo)
    return 15.0  # fallback


def _get_region_type(lat: float, lng: float) -> str:
    """Classify a location into a humidity region type."""
    # Coastal south: < 25N, east of 108E
    if lat < 25.0 and lng > 108.0:
        return "coastal_south"
    # Coastal east: 25-35N, east of 118E
    if 25.0 <= lat < 35.0 and lng > 118.0:
        return "coastal_east"
    # Inland south: 22-30N, 100-115E
    if 22.0 <= lat < 30.0 and 100.0 <= lng <= 115.0:
        return "inland_south"
    # Inland central: 28-38N, 105-118E
    if 28.0 <= lat < 38.0 and 105.0 <= lng <= 118.0:
        return "inland_central"
    # Inland north: 38-48N, 105-125E
    if 38.0 <= lat < 48.0 and 105.0 <= lng <= 125.0:
        return "inland_north"
    # NE China: 40-54N, 120-135E
    if lat >= 40.0 and lng > 120.0:
        return "ne_china"
    # Tibet Plateau: 26-37N, 78-100E, high elevation
    if 26.0 <= lat <= 37.0 and 78.0 <= lng <= 100.0:
        return "tibet_plateau"
    # Arid NW: rest of NW China
    if lat >= 35.0 and lng <= 108.0:
        return "arid_nw"
    return "inland_central"


def _get_humidity(lat: float, lng: float, month: int) -> float:
    """Estimate typical relative humidity (%) for a location and month."""
    region = _get_region_type(lat, lng)
    values = _HUMIDITY_BY_REGION.get(region, _HUMIDITY_BY_REGION["inland_central"])
    month_idx = max(0, min(11, month - 1))
    return values[month_idx]


# ═══════════════════════════════════════════════════════════════════════════
# Climate validator
# ═══════════════════════════════════════════════════════════════════════════

class ClimateValidator:
    """Validate bbox compatibility with temperature/humidity sensor readings.

    Uses physically-grounded models to estimate expected climate at any
    location, then checks if sensor readings are plausible for that location.
    """

    def __init__(self):
        pass

    def estimate_temperature_range(self, lat: float, lng: float,
                                    elevation_m: float, month: int) -> Tuple[float, float]:
        """Estimate typical min/max temperature range for a location.

        Returns (typical_low, typical_high) in °C. The range accounts for
        diurnal variation and weather variability, not just monthly mean.
        """
        base = _get_base_temp(lat, month)
        elev_correction = elevation_m / 1000.0 * LAPSE_RATE
        mean_temp = base - elev_correction

        # Diurnal range: larger in arid/continental climates, smaller in coastal
        region = _get_region_type(lat, lng)
        if region in ("arid_nw", "tibet_plateau"):
            diurnal_range = 15.0
        elif region in ("inland_north", "ne_china"):
            diurnal_range = 12.0
        else:
            diurnal_range = 8.0

        # Elevation uncertainty: ±ELEV_UNCERTAINTY m → ±3-4°C
        elev_uncertainty = ELEV_UNCERTAINTY / 1000.0 * LAPSE_RATE

        low = mean_temp - diurnal_range / 2 - elev_uncertainty
        high = mean_temp + diurnal_range / 2 + elev_uncertainty
        return (low, high)

    def estimate_humidity_range(self, lat: float, lng: float,
                                 month: int) -> Tuple[float, float]:
        """Estimate typical humidity range for a location and month."""
        mean_rh = _get_humidity(lat, lng, month)

        region = _get_region_type(lat, lng)
        if region in ("arid_nw",):
            variability = 15.0
        elif region in ("tibet_plateau",):
            variability = 12.0
        else:
            variability = 10.0

        low = max(5, mean_rh - variability)
        high = min(100, mean_rh + variability)
        return (low, high)

    def is_temperature_plausible(self, lat: float, lng: float,
                                  elevation_m: float, sensor_temp_c: float,
                                  month: Optional[int] = None,
                                  tolerance_c: float = 8.0) -> bool:
        """Check if sensor temperature is plausible for a location.

        If month is not given, checks against ANY month (more permissive).
        tolerance_c: extra tolerance beyond model uncertainty.
        """
        if month is not None:
            lo, hi = self.estimate_temperature_range(lat, lng, elevation_m, month)
            return (lo - tolerance_c) <= sensor_temp_c <= (hi + tolerance_c)

        # Without month info, check if temp is plausible for at least one season
        for m in [1, 4, 7, 10]:  # sample 4 months
            lo, hi = self.estimate_temperature_range(lat, lng, elevation_m, m)
            if (lo - tolerance_c) <= sensor_temp_c <= (hi + tolerance_c):
                return True
        return False

    def is_humidity_plausible(self, lat: float, lng: float,
                               sensor_humidity_pct: float,
                               month: Optional[int] = None,
                               tolerance_pct: float = 15.0) -> bool:
        """Check if sensor humidity is plausible for a location."""
        if month is not None:
            lo, hi = self.estimate_humidity_range(lat, lng, month)
            return (lo - tolerance_pct) <= sensor_humidity_pct <= (hi + tolerance_pct)

        for m in [1, 4, 7, 10]:
            lo, hi = self.estimate_humidity_range(lat, lng, m)
            if (lo - tolerance_pct) <= sensor_humidity_pct <= (hi + tolerance_pct):
                return True
        return False

    def bbox_matches_climate(self, lat_min: float, lat_max: float,
                              lng_min: float, lng_max: float,
                              sensor_temp_c: Optional[float] = None,
                              sensor_humidity_pct: Optional[float] = None,
                              dem=None, month: Optional[int] = None) -> bool:
        """Check if ANY point in a bbox is climatically compatible with sensors.

        Samples center + four corners. Returns True if any sampled point is
        plausible for BOTH temperature and humidity (if provided).
        """
        # Sample points: center + 4 corners
        sample_points = [
            ((lat_min + lat_max) / 2, (lng_min + lng_max) / 2),
            (lat_min, lng_min), (lat_min, lng_max),
            (lat_max, lng_min), (lat_max, lng_max),
        ]

        for lat, lng in sample_points:
            elev = dem.query(lat, lng) if dem else 500.0
            if elev is None:
                elev = 500.0

            temp_ok = True
            if sensor_temp_c is not None:
                temp_ok = self.is_temperature_plausible(
                    lat, lng, elev, sensor_temp_c, month)

            humid_ok = True
            if sensor_humidity_pct is not None:
                humid_ok = self.is_humidity_plausible(
                    lat, lng, sensor_humidity_pct, month)

            if temp_ok and humid_ok:
                return True

        return False


# Singleton
_climate_instance: Optional[ClimateValidator] = None


def get_climate() -> ClimateValidator:
    global _climate_instance
    if _climate_instance is None:
        _climate_instance = ClimateValidator()
    return _climate_instance


def filter_by_climate(bboxes, sensor_temp_c: Optional[float] = None,
                      sensor_humidity_pct: Optional[float] = None,
                      dem=None, month: Optional[int] = None):
    """Remove bboxes incompatible with temperature/humidity sensor readings.

    Two-pass strategy mirroring filter_by_elevation:
      1. Strict pass: both temp and humidity must match
      2. Relaxed pass: only one needs to match (if both provided)

    Returns empty list only if NO bbox is climatically plausible.
    """
    if sensor_temp_c is None and sensor_humidity_pct is None:
        return bboxes

    validator = get_climate()

    def _matches(b, require_both=True):
        result = validator.bbox_matches_climate(
            b.lat_min, b.lat_max, b.lng_min, b.lng_max,
            sensor_temp_c, sensor_humidity_pct, dem, month)
        if result or not require_both:
            return result
        # Relaxed: match either temp OR humidity
        if sensor_temp_c and sensor_humidity_pct:
            temp_only = validator.bbox_matches_climate(
                b.lat_min, b.lat_max, b.lng_min, b.lng_max,
                sensor_temp_c, None, dem, month)
            humid_only = validator.bbox_matches_climate(
                b.lat_min, b.lat_max, b.lng_min, b.lng_max,
                None, sensor_humidity_pct, dem, month)
            return temp_only or humid_only
        return result

    # Strict pass
    filtered = [b for b in bboxes if _matches(b, require_both=True)]
    if filtered:
        return filtered

    # Relaxed pass
    relaxed = [b for b in bboxes if _matches(b, require_both=False)]
    return relaxed
