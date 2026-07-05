"""Generate enhanced synthetic elevation grid for China.

Uses China's "three steps" (三级阶梯) topography as base, overlaid with
30+ major mountain peaks using Gaussian falloff kernels and ridge lines
connecting peaks along major mountain ranges.

Resolution: 0.05° (~5.5km at equator)
Coverage: 18-54N, 73-135E
Output: ~895K float32 points, ~3.5MB uncompressed, ~20KB compressed

Usage:
    python scripts/prepare_dem.py

Output:
    data/dem/china_elevation_005deg.npz
"""

import math
import os
import sys
import numpy as np

RES_DEG = 0.05
LAT_MIN, LAT_MAX = 18.0, 54.0
LNG_MIN, LNG_MAX = 73.0, 135.0


# ═══════════════════════════════════════════════════════════════════════════════
# Mountain peaks — (lat, lng, elevation_m, sigma_deg)
# sigma_deg controls the Gaussian falloff radius in degrees
# ═══════════════════════════════════════════════════════════════════════════════

MOUNTAIN_PEAKS = [
    # ── Himalaya (W→E) ──────────────────────────────────────────────────
    (35.24, 74.59, 8126, 0.22),   # Nanga Parbat
    (30.92, 79.59, 7756, 0.22),   # Kamet
    (30.28, 80.82, 7434, 0.20),   # Nanda Devi
    (28.60, 83.96, 8091, 0.22),   # Annapurna
    (27.99, 86.92, 8848, 0.28),   # Everest / Qomolangma
    (27.70, 88.15, 8586, 0.25),   # Kanchenjunga
    (28.08, 90.18, 7545, 0.18),   # Gangkhar Puensum
    (29.63, 95.05, 7782, 0.22),   # Namcha Barwa

    # ── Karakoram ────────────────────────────────────────────────────────
    (35.88, 76.51, 8611, 0.25),   # K2 / Godwin Austen
    (35.72, 76.70, 8080, 0.22),   # Gasherbrum I
    (35.64, 76.31, 7821, 0.20),   # Masherbrum

    # ── Pamir / W Kunlun ─────────────────────────────────────────────────
    (38.59, 75.30, 7649, 0.20),   # Kongur
    (38.28, 75.12, 7546, 0.20),   # Muztagh Ata

    # ── Kunlun (W→E) ─────────────────────────────────────────────────────
    (36.43, 76.70, 7280, 0.18),   # Skamri
    (36.12, 82.93, 7167, 0.18),   # Liushi Shan
    (36.41, 87.38, 6973, 0.18),   # Ulugh Muztagh
    (36.04, 90.87, 6860, 0.15),   # Bukadaban

    # ── Tianshan (W→E) ───────────────────────────────────────────────────
    (42.04, 80.13, 7439, 0.20),   # Tomur / Pobedy
    (42.21, 80.17, 7010, 0.18),   # Khan Tengri
    (43.08, 85.23, 4830, 0.12),   # Tianshan central
    (43.80, 88.33, 5445, 0.15),   # Bogda
    (43.63, 93.25, 4300, 0.10),   # Barkol / E Tianshan

    # ── Altun ─────────────────────────────────────────────────────────────
    (38.75, 89.25, 6100, 0.15),   # Altun highest

    # ── Qilian ────────────────────────────────────────────────────────────
    (38.87, 98.53, 5547, 0.12),   # Qilian main peak
    (37.68, 101.70, 5254, 0.10),  # Lenglongling
    (37.20, 102.80, 3600, 0.08),  # Wushaoling

    # ── Nyainqentanglha ──────────────────────────────────────────────────
    (30.38, 90.58, 7162, 0.18),   # Nyainqentanglha main
    (29.80, 92.35, 6500, 0.15),   # E Nyainqentanglha

    # ── Hengduan Mountains (N→S) ─────────────────────────────────────────
    (29.59, 101.88, 7556, 0.18),  # Gongga / Minya Konka
    (30.04, 101.95, 5880, 0.10),  # N Gongga range
    (28.44, 98.68, 6740, 0.18),   # Meili / Kawagebo
    (27.10, 100.18, 5596, 0.12),  # Yulong / Jade Dragon
    (27.38, 100.10, 5396, 0.10),  # Haba
    (28.58, 99.47, 5200, 0.10),   # Shaluli Shan
    (30.02, 102.35, 4450, 0.08),  # W Sichuan peaks

    # ── Tanggula ──────────────────────────────────────────────────────────
    (33.50, 91.05, 6621, 0.18),   # Geladaindong
    (32.85, 93.55, 6200, 0.15),   # E Tanggula

    # ── Qinling ───────────────────────────────────────────────────────────
    (33.96, 107.77, 3767, 0.10),  # Taibai Shan
    (34.48, 110.09, 2155, 0.06),  # Huashan

    # ── Altai ─────────────────────────────────────────────────────────────
    (49.81, 86.59, 4506, 0.15),   # Belukha
    (49.15, 87.93, 4374, 0.12),   # Khuiten

    # ── Daxinganling / Greater Khingan ───────────────────────────────────
    (50.08, 122.20, 1730, 0.10),  # N Greater Khingan
    (47.50, 121.50, 1500, 0.10),  # Central Greater Khingan

    # ── Changbai ──────────────────────────────────────────────────────────
    (41.99, 128.08, 2744, 0.12),  # Paektu / Changbai

    # ── Other notable peaks ───────────────────────────────────────────────
    (29.52, 103.34, 3099, 0.10),  # Emei Shan
    (39.08, 113.57, 3061, 0.08),  # Wutai Shan
    (30.13, 118.17, 1864, 0.06),  # Huangshan (Lotus)
    (27.91, 108.69, 2570, 0.08),  # Fanjing Shan
    (27.72, 117.68, 2158, 0.06),  # Wuyi Shan
    (29.35, 110.42, 1890, 0.06),  # Wuling (Zhangjiajie)
    (25.98, 110.38, 2142, 0.06),  # Mao'er Shan (Guangxi)
    (36.24, 117.11, 1545, 0.05),  # Taishan
    (29.57, 115.97, 1474, 0.05),  # Lushan
    (24.81, 113.27, 1902, 0.05),  # Danxia Shan
    (38.05, 106.33, 3556, 0.10),  # Helan Shan
    (39.90, 116.38, 100, 0.03),   # Beijing (marker, not a peak)
]


# ═══════════════════════════════════════════════════════════════════════════════
# Mountain ridges — (start_lat, start_lng, end_lat, end_lng, elev_lo, elev_hi)
# Sample points every ~0.08° along each ridge. Each point gets σ=0.06°.
# ═══════════════════════════════════════════════════════════════════════════════

RIDGES = [
    # Himalaya arc
    (35.24, 74.59, 27.99, 86.92, 5000, 8848),
    (27.99, 86.92, 29.63, 95.05, 4500, 8848),
    # Karakoram
    (35.88, 76.51, 35.64, 76.31, 5000, 8611),
    # Kunlun
    (38.59, 75.30, 36.12, 82.93, 4000, 7649),
    (36.12, 82.93, 36.04, 90.87, 4000, 7167),
    # Tianshan
    (42.04, 80.13, 43.80, 88.33, 2500, 7439),
    (43.80, 88.33, 43.63, 93.25, 2000, 5445),
    # Hengduan (N-S)
    (29.59, 101.88, 27.10, 100.18, 3000, 7556),
    # Nyainqentanglha
    (30.38, 90.58, 29.80, 92.35, 4000, 7162),
    # Tanggula
    (33.50, 91.05, 32.85, 93.55, 4000, 6621),
    # Qilian
    (38.87, 98.53, 37.20, 102.80, 2800, 5547),
    # Qinling
    (33.96, 107.77, 34.48, 110.09, 1500, 3767),
    # Altai
    (49.81, 86.59, 49.15, 87.93, 2000, 4506),
]


def _sample_ridge(ridge, spacing_deg=0.08):
    """Generate interpolation points along a ridge segment."""
    lat0, lng0, lat1, lng1, elev_lo, elev_hi = ridge
    dist_deg = math.sqrt((lat1 - lat0)**2 + (lng1 - lng0)**2)
    n = max(2, int(dist_deg / spacing_deg) + 1)
    points = []
    for i in range(n):
        t = i / (n - 1)
        lat = lat0 + t * (lat1 - lat0)
        lng = lng0 + t * (lng1 - lng0)
        elev = elev_lo + t * (elev_hi - elev_lo)
        points.append((lat, lng, elev, 0.06))
    return points


# ═══════════════════════════════════════════════════════════════════════════════
# Base elevation zones — "Three Steps" topography + sub-regions
# Evaluated in order; first match wins.
# ═══════════════════════════════════════════════════════════════════════════════

def base_elevation(lat, lng):
    """Base elevation (m) from geographic zone. Does NOT include peaks."""

    # ── Ocean / Sea (outside Chinese landmass) ──────────────────────────
    # Bay of Bengal
    if lat < 22.0 and lng < 92.0:
        return 0
    # South China Sea
    if lat < 20.0 and lng > 108.0:
        return 0
    if lat < 22.5 and lng > 116.0:
        return 0
    # East China Sea
    if 25.0 <= lat < 31.5 and lng > 122.5:
        return 0
    # Yellow Sea
    if 31.5 <= lat < 39.0 and lng > 123.0:
        return 0
    # Sea of Japan area (within grid)
    if 38.5 <= lat < 43.0 and lng > 131.5:
        return 0
    # Bohai Sea
    if 37.5 <= lat < 40.5 and 118.5 <= lng <= 122.0:
        # Bohai is shallow but water
        if 38.5 <= lat <= 39.0 and 119.0 <= lng <= 120.5:
            return 0

    # ── STEP 1: Tibet-Qinghai Plateau (>3000m core) ─────────────────────
    # Central Tibet (highest)
    if 30.0 <= lat <= 33.0 and 84.0 <= lng <= 91.0:
        if 31.0 <= lat <= 32.5 and 85.0 <= lng <= 90.0:
            return 4850  # Highest plateau core
        return 4600

    # ── Tibet river valleys (specific → must come before broad S Tibet) ──
    # Yarlung Tsangpo (Brahmaputra) valley
    if 28.5 <= lat <= 30.2 and 88.5 <= lng <= 93.0:
        if 29.0 <= lat <= 29.8 and 90.5 <= lng <= 92.0:
            return 3600  # Lhasa / Tsetang valley floor
        return 3900  # Wider valley

    # S Tibet (Himalaya foothills plateau)
    if 27.2 <= lat < 30.0 and 85.0 <= lng <= 92.0:
        return 4500

    # W Tibet / Ngari
    if 30.0 <= lat <= 35.0 and 78.0 <= lng < 84.0:
        if 33.0 <= lat <= 34.5 and 80.0 <= lng <= 83.0:
            return 5100  # NW Tibet high
        return 4700

    # N Tibet / Changtang
    if 33.0 <= lat <= 36.5 and 84.0 <= lng <= 91.0:
        return 4900

    # NE Tibet / Amdo
    if 33.0 <= lat <= 36.5 and 91.0 <= lng <= 96.0:
        return 4300

    # Qinghai Plateau (S)
    if 31.0 <= lat <= 35.0 and 92.0 <= lng <= 97.5:
        if 33.0 <= lat <= 34.5 and 94.0 <= lng <= 97.0:
            return 3600  # Yushu area
        return 3400

    # Qinghai Plateau (N) / Qaidam Basin
    if 35.0 <= lat <= 39.2 and 90.0 <= lng <= 99.0:
        # Qaidam Basin floor
        if 36.5 <= lat <= 38.2 and 91.0 <= lng <= 96.5:
            return 2800  # Qaidam — lower than surrounding plateau
        return 3200

    # E Qinghai / Gansu transition
    if 33.0 <= lat <= 39.5 and 99.0 <= lng <= 103.5:
        if 35.0 <= lat <= 38.0 and 100.0 <= lng <= 103.0:
            return 1900  # Lanzhou / Gansu corridor
        return 2700

    # ── STEP 2a: Xinjiang Basins ────────────────────────────────────────
    # Turpan Depression
    if 42.0 <= lat <= 43.5 and 88.5 <= lng <= 90.5:
        return -150  # Lowest point in China

    # Tarim Basin
    if 36.0 <= lat <= 42.0 and 77.0 <= lng <= 89.0:
        # Basin core — flat desert floor
        if 37.5 <= lat <= 41.0 and 80.0 <= lng <= 87.0:
            return 1000  # Taklamakan floor
        if 36.0 <= lat < 37.5 and 80.0 <= lng <= 85.0:
            return 1300  # S Tarim / Hotan area (higher)
        return 1200  # Basin edges

    # Junggar Basin
    if 44.0 <= lat <= 47.5 and 83.0 <= lng <= 90.0:
        if 44.5 <= lat <= 46.5 and 84.5 <= lng <= 88.5:
            return 400  # Gurbantunggut desert floor
        return 600

    # Hexi Corridor
    if 38.0 <= lat <= 42.0 and 95.0 <= lng <= 104.5:
        if 39.0 <= lat <= 41.0 and 97.0 <= lng <= 102.0:
            return 1400  # Corridor floor
        return 1600

    # ── STEP 2b: Inner Mongolia Plateau ─────────────────────────────────
    if 40.5 <= lat <= 47.0 and 105.0 <= lng <= 120.0:
        if 42.0 <= lat <= 44.5 and 112.0 <= lng <= 117.0:
            return 1100  # Central Inner Mongolia
        if 44.5 <= lat <= 47.0 and 115.0 <= lng <= 120.0:
            return 800  # E Inner Mongolia (lower)
        return 1000

    # ── STEP 2c: Loess Plateau ──────────────────────────────────────────
    if 34.5 <= lat <= 40.5 and 104.0 <= lng <= 114.0:
        if 35.5 <= lat <= 39.0 and 106.5 <= lng <= 112.0:
            return 1200  # Core Loess Plateau
        if 34.5 <= lat <= 35.5 and 108.0 <= lng <= 111.0:
            return 800  # Wei River valley (lower)
        return 1050

    # ── STEP 2d: Sichuan Basin ──────────────────────────────────────────
    if 28.5 <= lat <= 32.5 and 103.5 <= lng <= 108.0:
        if 29.5 <= lat <= 31.5 and 104.0 <= lng <= 107.0:
            return 420  # Chengdu-Chongqing basin floor
        return 600  # Basin rim

    # ── STEP 2e: Yunnan-Guizhou Plateau ─────────────────────────────────
    # W Yunnan / Gaoligong
    if 24.0 <= lat <= 28.5 and 97.5 <= lng <= 100.5:
        if 25.5 <= lat <= 27.5 and 98.5 <= lng <= 100.0:
            return 2200  # Dali / Lijiang area
        return 1700

    # Central Yunnan Plateau
    if 23.0 <= lat <= 27.0 and 100.5 <= lng <= 104.0:
        if 24.5 <= lat <= 26.5 and 101.5 <= lng <= 103.5:
            return 1900  # Kunming area
        return 1600

    # S Yunnan (Xishuangbanna)
    if 21.0 <= lat <= 24.0 and 99.5 <= lng <= 102.0:
        return 900

    # Guizhou Plateau
    if 25.0 <= lat <= 29.0 and 104.0 <= lng <= 109.0:
        if 26.0 <= lat <= 28.0 and 105.5 <= lng <= 108.0:
            return 1100  # Guizhou core
        return 900

    # W Guangxi / N Vietnam border
    if 22.0 <= lat <= 25.0 and 104.5 <= lng <= 108.0:
        return 700

    # ── STEP 3a: NE China Plain ─────────────────────────────────────────
    if 43.0 <= lat <= 49.0 and 122.0 <= lng <= 133.0:
        if 45.0 <= lat <= 47.5 and 125.0 <= lng <= 130.0:
            return 140  # Central NE Plain (Harbin area)
        return 180

    # Sanjiang Plain
    if 46.0 <= lat <= 48.5 and 130.0 <= lng <= 135.0:
        return 60

    # ── STEP 3b: North China Plain ──────────────────────────────────────
    if 32.5 <= lat <= 40.5 and 114.0 <= lng <= 120.5:
        if 35.0 <= lat <= 39.5 and 115.0 <= lng <= 118.5:
            return 35  # Core plain (Beijing-Shijiazhuang-Zhengzhou)
        if 32.5 <= lat <= 35.0 and 116.0 <= lng <= 118.5:
            return 30  # Huai River plain
        return 60

    # Shandong hills
    if 35.5 <= lat <= 38.0 and 117.0 <= lng <= 122.0:
        if 36.2 <= lat <= 37.0 and 117.5 <= lng <= 119.0:
            return 400  # Taishan area
        return 150  # Shandong coastal hills

    # ── STEP 3c: Middle-Lower Yangtze Plain ─────────────────────────────
    # Yangtze Delta
    if 29.5 <= lat <= 33.0 and 118.5 <= lng <= 122.5:
        if 30.5 <= lat <= 32.0 and 120.0 <= lng <= 121.8:
            return 5  # Shanghai / Suzhou (near sea level)
        return 15

    # Central Yangtze (Hubei/Hunan lakes)
    if 28.5 <= lat <= 31.5 and 111.0 <= lng <= 116.0:
        if 29.5 <= lat <= 30.8 and 112.5 <= lng <= 114.5:
            return 30  # Jianghan Plain / Wuhan area
        return 80

    # Dongting Lake area
    if 28.5 <= lat <= 29.8 and 112.0 <= lng <= 113.5:
        return 35

    # Poyang Lake area
    if 28.0 <= lat <= 29.8 and 115.5 <= lng <= 117.2:
        return 25

    # ── STEP 3d: SE China Hills ─────────────────────────────────────────
    # Zhejiang / Fujian coastal
    if 26.0 <= lat <= 30.0 and 119.5 <= lng <= 122.5:
        if 27.5 <= lat <= 29.5 and 120.5 <= lng <= 122.0:
            return 300  # Wenzhou / Ningde coast
        return 150

    # Fujian inland hills
    if 24.5 <= lat <= 28.0 and 116.5 <= lng <= 120.0:
        return 550

    # Jiangxi / S Anhui hills
    if 27.0 <= lat <= 31.0 and 115.5 <= lng <= 119.5:
        if 29.0 <= lat <= 30.5 and 117.0 <= lng <= 118.8:
            return 300  # Huangshan area
        return 400

    # Hunan hills
    if 25.5 <= lat <= 29.5 and 109.5 <= lng <= 114.0:
        return 400

    # ── STEP 3e: S China Coast ──────────────────────────────────────────
    # Pearl River Delta
    if 21.5 <= lat <= 23.8 and 112.5 <= lng <= 115.0:
        if 22.2 <= lat <= 23.2 and 113.2 <= lng <= 114.2:
            return 10  # Guangzhou / Shenzhen / Zhuhai
        return 50

    # Guangdong / Guangxi coastal hills
    if 21.0 <= lat <= 25.0 and 108.0 <= lng <= 117.0:
        if 21.5 <= lat <= 22.5 and 109.0 <= lng <= 110.5:
            return 100  # Leizhou / Beihai
        return 250

    # Hainan Island
    if 18.0 <= lat <= 20.2 and 108.5 <= lng <= 111.2:
        if 18.7 <= lat <= 19.5 and 109.5 <= lng <= 110.5:
            return 200  # Wuzhi Shan area
        return 80

    # Taiwan (within grid)
    if 21.8 <= lat <= 25.5 and 120.5 <= lng <= 122.5:
        return 800  # Taiwan central mountains

    # ── STEP 3f: NE China Periphery ─────────────────────────────────────
    # Lesser Khingan
    if 46.5 <= lat <= 50.0 and 126.0 <= lng <= 130.5:
        return 600

    # ── Outside China (approximate) ─────────────────────────────────────
    # India / Nepal / Bhutan (S of Himalaya — lowlands)
    if lat < 27.5 and lng < 89.0:
        if lat < 25.0:
            return 200  # N India plains
        return 800  # Nepal/Bhutan foothills

    # Myanmar
    if 19.0 <= lat <= 26.0 and 96.0 <= lng <= 100.0:
        return 500

    # Laos / Vietnam
    if 18.0 <= lat <= 23.0 and 100.0 <= lng <= 108.0:
        return 400

    # Korea Peninsula (within grid)
    if 37.0 <= lat <= 43.0 and 124.5 <= lng <= 131.0:
        return 300

    # Mongolia (S part within grid)
    if 42.0 <= lat <= 50.0 and 100.0 <= lng <= 115.0:
        return 1100  # Mongolian plateau

    # Russia Far East (within grid)
    if 49.0 <= lat <= 54.0 and 120.0 <= lng <= 135.0:
        return 500

    # Central Asia (Kyrgyzstan, Tajikistan, etc. within grid)
    if 37.0 <= lat <= 42.0 and 73.0 <= lng <= 77.0:
        return 2500  # Fergana / Pamir-Alai foothills

    # Afghanistan / N Pakistan (within grid)
    if 35.0 <= lat <= 37.0 and 73.0 <= lng <= 76.0:
        return 3000  # Hindu Kush foothills

    # Fallback
    return 500


# ═══════════════════════════════════════════════════════════════════════════════
# Grid generation with peak overlay
# ═══════════════════════════════════════════════════════════════════════════════

def generate_grid():
    """Generate full elevation grid: base zones + mountain peaks + ridges."""
    n_lat = int((LAT_MAX - LAT_MIN) / RES_DEG) + 1
    n_lng = int((LNG_MAX - LNG_MIN) / RES_DEG) + 1
    print(f"Grid: {n_lat} rows x {n_lng} cols = {n_lat * n_lng:,} points")

    lats = np.linspace(LAT_MIN, LAT_MAX, n_lat)
    lngs = np.linspace(LNG_MIN, LNG_MAX, n_lng)
    lng_grid, lat_grid = np.meshgrid(lngs, lats)  # (n_lat, n_lng)

    # ── Step 1: Fill base elevations ────────────────────────────────────
    print("Filling base elevations...")
    grid = np.zeros((n_lat, n_lng), dtype=np.float32)
    for i in range(n_lat):
        lat = LAT_MIN + i * RES_DEG
        for j in range(n_lng):
            lng = LNG_MIN + j * RES_DEG
            grid[i, j] = base_elevation(lat, lng)

    print(f"  Base range: {grid.min():.0f} - {grid.max():.0f}m")

    # ── Step 2: Generate ridge points ───────────────────────────────────
    ridge_points = []
    for ridge in RIDGES:
        pts = _sample_ridge(ridge)
        ridge_points.extend(pts)
    print(f"  Ridge points: {len(ridge_points)}")

    # ── Step 3: Overlay peaks + ridge points via Gaussian max ───────────
    all_points = list(MOUNTAIN_PEAKS) + ridge_points
    print(f"  Total peak features: {len(all_points)}")

    for idx, (plat, plng, pheight, psigma) in enumerate(all_points):
        if idx % 20 == 0:
            print(f"  Peak {idx}/{len(all_points)}: ({plat:.2f}, {plng:.2f}) {pheight:.0f}m")

        # Only process grid points within 3*sigma (beyond this, Gaussian < 0.011)
        sigma_deg = psigma
        lat_lo = plat - 3 * sigma_deg
        lat_hi = plat + 3 * sigma_deg
        lng_lo = plng - 3 * sigma_deg
        lng_hi = plng + 3 * sigma_deg

        i_lo = max(0, int((lat_lo - LAT_MIN) / RES_DEG))
        i_hi = min(n_lat - 1, int((lat_hi - LAT_MIN) / RES_DEG) + 1)
        j_lo = max(0, int((lng_lo - LNG_MIN) / RES_DEG))
        j_hi = min(n_lng - 1, int((lng_hi - LNG_MIN) / RES_DEG) + 1)

        if i_lo > i_hi or j_lo > j_hi:
            continue

        # Extract sub-grid and compute distances
        sub_lats = lat_grid[i_lo:i_hi+1, j_lo:j_hi+1]
        sub_lngs = lng_grid[i_lo:i_hi+1, j_lo:j_hi+1]
        dlat = sub_lats - plat
        dlng = sub_lngs - plng
        cos_mid = np.cos(np.radians((sub_lats + plat) / 2.0))
        dist_km = np.sqrt((dlat * 111.32) ** 2 + (dlng * 111.32 * cos_mid) ** 2)

        sigma_km = psigma * 111.32
        gaussian = np.exp(-(dist_km ** 2) / (2.0 * sigma_km ** 2))

        # Only apply where the peak rises above the existing grid
        contribution = pheight * gaussian
        sub_grid = grid[i_lo:i_hi+1, j_lo:j_hi+1]
        grid[i_lo:i_hi+1, j_lo:j_hi+1] = np.maximum(sub_grid, contribution)

    print(f"  Final range: {grid.min():.0f} - {grid.max():.0f}m")
    print(f"  Final mean:  {grid.mean():.0f}m")
    return grid


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "dem")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "china_elevation_005deg.npz")

    print("=" * 60)
    print("Building enhanced China elevation grid")
    print(f"  Resolution: {RES_DEG}° (~{RES_DEG * 111:.1f} km)")
    print(f"  Coverage: {LAT_MIN}-{LAT_MAX}N, {LNG_MIN}-{LNG_MAX}E")
    print(f"  Peaks: {len(MOUNTAIN_PEAKS)}, Ridges: {len(RIDGES)}")
    print("=" * 60)

    grid = generate_grid()

    np.savez_compressed(
        out_path,
        elevation=grid,
        lat_min=LAT_MIN, lat_max=LAT_MAX,
        lng_min=LNG_MIN, lng_max=LNG_MAX,
        res_deg=RES_DEG,
    )

    size_kb = os.path.getsize(out_path) / 1024
    n_unique = len(np.unique(grid))
    print(f"\nSaved: {out_path}")
    print(f"  Size: {size_kb:.0f} KB")
    print(f"  Shape: {grid.shape}")
    print(f"  Range: {grid.min():.0f} - {grid.max():.0f} m")
    print(f"  Mean: {grid.mean():.0f} m")
    print(f"  Unique elevations: {n_unique:,}")

    # Quick spot check
    print("\nSpot check:")
    checks = [
        ("Everest", 27.99, 86.92),
        ("Gongga", 29.59, 101.88),
        ("Meili", 28.44, 98.68),
        ("Lhasa", 29.65, 91.10),
        ("Chengdu", 30.57, 104.07),
        ("Kunming", 25.04, 102.72),
        ("Tarim Basin", 39.0, 83.0),
        ("Qaidam Basin", 37.0, 95.0),
        ("Turpan", 42.8, 89.3),
        ("Beijing", 39.9, 116.4),
        ("Shanghai", 31.2, 121.5),
        ("Huangshan", 30.13, 118.17),
        ("Taibai", 33.96, 107.77),
        ("Harbin", 45.75, 126.63),
        ("Guangzhou", 23.13, 113.26),
    ]
    n_lat = grid.shape[0]
    for name, lat, lng in checks:
        r = int((lat - LAT_MIN) / RES_DEG)
        c = int((lng - LNG_MIN) / RES_DEG)
        if 0 <= r < n_lat and 0 <= c < grid.shape[1]:
            elev = grid[r, c]
            print(f"  {name:15s} ({lat:.2f}, {lng:.2f}): {elev:.0f}m")


if __name__ == "__main__":
    main()
