"""End-to-end benchmark: GeoCoT -> element extraction -> fusion pipeline.

Two modes:
  --mock     Fast validation with pre-defined element sets (no GPU needed)
  --geocot   Full pipeline with Qwen3-VL-2B inference (needs GPU)

Usage:
  python scripts/test_fusion_e2e.py --mock          # Quick fusion validation
  python scripts/test_fusion_e2e.py --geocot        # Full E2E with VLM
  python scripts/test_fusion_e2e.py --geocot --n 50 # 50 images
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regression.element_fusion import (
    fuse_elements, fuse_elements_v3, run_fusion_pipeline, run_fusion_pipeline_v3,
    run_simple_visual_pipeline,
    simple_answers_to_elements, parse_simple_visual_json,
    _elements_to_simple_answers, FusionResult,
)
from regression.dem_lookup import get_dem
from regression.climate_lookup import get_climate


# ═══════════════════════════════════════════════════════════════════════════
# Mock test cases — diverse scenarios covering different regions
# ═══════════════════════════════════════════════════════════════════════════

MOCK_CASES = [
    # ── Karst landscapes ─────────────────────────────────────────────
    {
        "name": "yangshuo_karst",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen", "soil_color": "red",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 300,
        },
        "true_lat": 24.78, "true_lng": 110.49,
        "expected_region": "Guilin/Yangshuo",
        "scene_category": "karst",
    },
    {
        "name": "shilin_stone_forest",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen", "soil_color": "red",
            "language_script": "simplified_chinese", "scene_type": "mountain_trail_karst",
            "elevation_estimate_m": 1800,
        },
        "true_lat": 24.82, "true_lng": 103.32,
        "expected_region": "Yunnan/Shilin",
        "scene_category": "karst",
    },
    # ── Granite mountains ────────────────────────────────────────────
    {
        "name": "huangshan_granite",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest", "mountain_rock_type": "granite_spheroidal",
            "language_script": "simplified_chinese", "soil_color": "yellow_brown",
            "elevation_estimate_m": 1600,
        },
        "true_lat": 30.13, "true_lng": 118.16,
        "expected_region": "Huangshan/Anhui",
        "scene_category": "granite_mountain",
    },
    {
        "name": "huashan_granite",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest", "mountain_rock_type": "granite_spheroidal",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 2000,
        },
        "true_lat": 34.48, "true_lng": 110.08,
        "expected_region": "Huashan/Shaanxi",
        "scene_category": "granite_mountain",
    },
    # ── Snow mountains ───────────────────────────────────────────────
    {
        "name": "gongga_snow",
        "elements": {
            "climate_zone": "alpine", "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest", "mountain_rock_type": "snow_peaks_glaciers",
            "elevation_estimate_m": 4500,
        },
        "true_lat": 29.59, "true_lng": 101.88,
        "expected_region": "Gongga/Sichuan",
        "scene_category": "snow_mountain",
    },
    {
        "name": "meili_snow",
        "elements": {
            "climate_zone": "alpine", "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow", "mountain_rock_type": "snow_peaks_glaciers",
            "language_script": "tibetan",
            "elevation_estimate_m": 5000,
        },
        "true_lat": 28.44, "true_lng": 98.68,
        "expected_region": "Meili/Yunnan",
        "scene_category": "snow_mountain",
    },
    # ── Desert/arid ──────────────────────────────────────────────────
    {
        "name": "qaidam_desert",
        "elements": {
            "climate_zone": "arid", "terrain_type": "desert_dunes",
            "vegetation_zone": "desert_scrub", "scene_type": "desert",
            "language_script": "tibetan",
            "elevation_estimate_m": 2800,
        },
        "true_lat": 36.5, "true_lng": 94.0,
        "expected_region": "Qaidam/Qinghai",
        "scene_category": "desert",
    },
    {
        "name": "taklamakan_desert",
        "elements": {
            "climate_zone": "arid", "terrain_type": "desert_dunes",
            "vegetation_zone": "desert_scrub", "scene_type": "desert",
            "language_script": "uyghur_arabic",
            "elevation_estimate_m": 1000,
        },
        "true_lat": 39.0, "true_lng": 83.0,
        "expected_region": "Taklamakan/Xinjiang",
        "scene_category": "desert",
    },
    # ── Urban scenes ─────────────────────────────────────────────────
    {
        "name": "shenzhen_urban",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "language_script": "simplified_chinese", "architecture_style": "modern_glass",
            "tree_species": ["banyan"], "sky_quality": "grey_hazy",
            "likely_provinces": ["Guangdong"],
            "elevation_estimate_m": 10,
        },
        "true_lat": 22.54, "true_lng": 114.06,
        "expected_region": "Shenzhen/Guangdong",
        "scene_category": "urban",
    },
    {
        "name": "beijing_urban",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "urban_flat",
            "vegetation_zone": "mixed_forest", "urbanization": "metropolis",
            "language_script": "simplified_chinese", "architecture_style": "traditional_official",
            "tree_species": ["gingko"], "sky_quality": "grey_hazy",
            "likely_provinces": ["Beijing"],
            "elevation_estimate_m": 50,
        },
        "true_lat": 39.90, "true_lng": 116.40,
        "expected_region": "Beijing",
        "scene_category": "urban",
    },
    {
        "name": "shanghai_urban",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "language_script": "simplified_chinese", "architecture_style": "shikumen",
            "tree_species": ["plane_tree"], "sky_quality": "grey_hazy",
            "likely_provinces": ["Shanghai"],
            "elevation_estimate_m": 5,
        },
        "true_lat": 31.23, "true_lng": 121.47,
        "expected_region": "Shanghai",
        "scene_category": "urban",
    },
    {
        "name": "harbin_urban",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "urban_flat",
            "vegetation_zone": "mixed_forest", "urbanization": "medium_city",
            "language_script": "simplified_chinese", "soil_color": "black",
            "tree_species": ["poplar"],
            "likely_provinces": ["Heilongjiang"],
            "elevation_estimate_m": 150,
        },
        "true_lat": 45.75, "true_lng": 126.63,
        "expected_region": "Harbin/Heilongjiang",
        "scene_category": "urban",
    },
    # ── Grassland/steppe ─────────────────────────────────────────────
    {
        "name": "inner_mongolia_grassland",
        "elements": {
            "climate_zone": "arid", "terrain_type": "grassland_steppe",
            "vegetation_zone": "grassland", "urbanization": "rural",
            "language_script": "mongolian",
            "elevation_estimate_m": 1000,
        },
        "true_lat": 43.5, "true_lng": 116.0,
        "expected_region": "Inner Mongolia",
        "scene_category": "grassland",
    },
    # ── Forest scenes ────────────────────────────────────────────────
    {
        "name": "changbai_forest",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "rolling_hills",
            "vegetation_zone": "conifer_forest", "soil_color": "black",
            "language_script": "simplified_chinese", "scene_type": "wilderness_forest",
            "elevation_estimate_m": 800,
        },
        "true_lat": 42.0, "true_lng": 128.0,
        "expected_region": "Changbai/Jilin",
        "scene_category": "forest",
    },
    {
        "name": "xishuangbanna_tropical",
        "elements": {
            "climate_zone": "tropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "tropical_rainforest", "soil_color": "red",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 600,
        },
        "true_lat": 21.93, "true_lng": 101.25,
        "expected_region": "Xishuangbanna/Yunnan",
        "scene_category": "forest",
    },
    # ── Coastal scenes ───────────────────────────────────────────────
    {
        "name": "hainan_coastal",
        "elements": {
            "climate_zone": "tropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "tropical_rainforest", "scene_type": "coastal",
            "language_script": "simplified_chinese",
            "tree_species": ["coconut_palm"],
            "elevation_estimate_m": 30,
        },
        "true_lat": 18.6, "true_lng": 109.7,
        "expected_region": "Hainan",
        "scene_category": "coastal",
    },
    # ── Tibet Plateau ────────────────────────────────────────────────
    {
        "name": "lhasa_valley",
        "elements": {
            "climate_zone": "alpine", "terrain_type": "plateau",
            "vegetation_zone": "alpine_meadow", "urbanization": "medium_city",
            "language_script": "tibetan", "sky_quality": "clear_blue",
            "infrastructure_tags": ["tibetan_signs"],
            "elevation_estimate_m": 3650,
        },
        "true_lat": 29.65, "true_lng": 91.10,
        "expected_region": "Lhasa/Tibet",
        "scene_category": "plateau",
    },
    # ── Danxia/red sandstone ─────────────────────────────────────────
    {
        "name": "zhangye_danxia",
        "elements": {
            "climate_zone": "arid", "terrain_type": "sharp_mountains",
            "vegetation_zone": "desert_scrub", "mountain_rock_type": "red_sandstone_danxia",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 1800,
        },
        "true_lat": 38.93, "true_lng": 100.13,
        "expected_region": "Zhangye/Gansu",
        "scene_category": "danxia",
    },
    # ═══════════════════════════════════════════════════════════════════════
    # FAILURE MODE TEST CASES — target the 3 failure modes
    # ═══════════════════════════════════════════════════════════════════════
    # ── 改善3: City confusion (Chengdu vs Shanghai) ───────────────────
    {
        "name": "chengdu_midrise_redbrick",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "red_brick_tiles",
            "language_script": "simplified_chinese", "architecture_style": "modern_residential",
            "elevation_estimate_m": 500,
        },
        "true_lat": 30.57, "true_lng": 104.06,
        "expected_region": "Chengdu/Sichuan",
        "scene_category": "urban_confusion",
    },
    {
        "name": "shanghai_highrise_grey",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "high_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese", "architecture_style": "modern_glass",
            "tree_species": ["plane_tree"],
            "elevation_estimate_m": 5,
        },
        "true_lat": 31.23, "true_lng": 121.47,
        "expected_region": "Shanghai",
        "scene_category": "urban_confusion",
    },
    # ── 改善1: Elevation absurdity (VLM says [10,50] at 424m Chengdu) ─
    {
        "name": "chengdu_elevation_absurd",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "red_brick_tiles",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": [10, 50],  # VLM wildly wrong
        },
        "true_lat": 30.57, "true_lng": 104.06,
        "expected_region": "Chengdu/Sichuan (elevation=424m, VLM says 10-50m)",
        "scene_category": "elevation_failure",
    },
    {
        "name": "kunming_elevation_absurd",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "medium_city",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": [0, 100],  # VLM says sea level, truth ~1900m
        },
        "true_lat": 25.04, "true_lng": 102.68,
        "expected_region": "Kunming/Yunnan (elevation=1890m, VLM says 0-100m)",
        "scene_category": "elevation_failure",
    },
    # ── 改善2: Systematic bias (GeoCoT predicts Sichuan for E China) ──
    {
        "name": "huangshan_bias_geocot",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest", "mountain_rock_type": "granite_spheroidal",
            "soil_color": "yellow_brown",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 1600,
        },
        "true_lat": 30.13, "true_lng": 118.16,
        "expected_region": "Huangshan/Anhui (GeoCoT biased to 29.5,103.0)",
        "scene_category": "bias_failure",
        # Simulate GeoCoT predicting Sichuan for an Anhui image
        "geocot_prediction": (29.55, 103.00),
    },
    {
        "name": "nanjing_bias_geocot",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "high_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "tree_species": ["plane_tree"],
            "elevation_estimate_m": 20,
        },
        "true_lat": 32.06, "true_lng": 118.79,
        "expected_region": "Nanjing/Jiangsu (GeoCoT biased to 29.0,103.5)",
        "scene_category": "bias_failure",
        "geocot_prediction": (29.00, 103.50),
    },
    # ═══════════════════════════════════════════════════════════════════════
    # CITY FINGERPRINT TEST CASES — verify new city compound scenes
    # ═══════════════════════════════════════════════════════════════════════
    {
        "name": "wuhan_highrise_grey",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "high_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 30,
        },
        "true_lat": 30.59, "true_lng": 114.30,
        "expected_region": "Wuhan/Hubei",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "chongqing_highrise_redbrick",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "high_rise", "pavement_type": "red_brick_tiles",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 250,
        },
        "true_lat": 29.56, "true_lng": 106.55,
        "expected_region": "Chongqing",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "hangzhou_midrise_grey",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "tree_species": ["willow"],
            "elevation_estimate_m": 10,
        },
        "true_lat": 30.25, "true_lng": 120.16,
        "expected_region": "Hangzhou/Zhejiang",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "tianjin_highrise_grey",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "urban_flat",
            "vegetation_zone": "mixed_forest", "urbanization": "metropolis",
            "building_height": "high_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 5,
        },
        "true_lat": 39.14, "true_lng": 117.20,
        "expected_region": "Tianjin",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "xian_midrise_grey",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "urban_flat",
            "vegetation_zone": "mixed_forest", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "architecture_style": "traditional_official",
            "elevation_estimate_m": 400,
        },
        "true_lat": 34.26, "true_lng": 108.94,
        "expected_region": "Xi'an/Shaanxi",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "changsha_midrise_redbrick",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "red_brick_tiles",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 50,
            "water_visible": True,
        },
        "true_lat": 28.22, "true_lng": 112.97,
        "expected_region": "Changsha/Hunan",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "kunming_midrise_highland",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "soil_color": "red",
            "elevation_estimate_m": 1890,
        },
        "true_lat": 25.04, "true_lng": 102.68,
        "expected_region": "Kunming/Yunnan",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "fuzhou_midrise_banyan",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "tree_species": ["banyan"],
            "elevation_estimate_m": 10,
        },
        "true_lat": 26.07, "true_lng": 119.30,
        "expected_region": "Fuzhou/Fujian",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "guiyang_midrise_hills",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "red_brick_tiles",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 1100,
        },
        "true_lat": 26.64, "true_lng": 106.71,
        "expected_region": "Guiyang/Guizhou",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "lanzhou_arid_midrise",
        "elements": {
            "climate_zone": "arid", "terrain_type": "urban_flat",
            "vegetation_zone": "desert_scrub", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 1520,
        },
        "true_lat": 36.06, "true_lng": 103.80,
        "expected_region": "Lanzhou/Gansu",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "urumqi_arid_midrise",
        "elements": {
            "climate_zone": "arid", "terrain_type": "urban_flat",
            "vegetation_zone": "desert_scrub", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "uyghur_arabic",
            "elevation_estimate_m": 800,
        },
        "true_lat": 43.83, "true_lng": 87.62,
        "expected_region": "Urumqi/Xinjiang",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "dalian_coastal_temperate",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "rolling_hills",
            "vegetation_zone": "mixed_forest", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "water_visible": True,
            "elevation_estimate_m": 50,
        },
        "true_lat": 38.92, "true_lng": 121.63,
        "expected_region": "Dalian/Liaoning",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "qingdao_coastal_temperate",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "rolling_hills",
            "vegetation_zone": "mixed_forest", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 30,
        },
        "true_lat": 36.07, "true_lng": 120.38,
        "expected_region": "Qingdao/Shandong",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "shenyang_temperate_midrise",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "urban_flat",
            "vegetation_zone": "mixed_forest", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 50,
        },
        "true_lat": 41.80, "true_lng": 123.43,
        "expected_region": "Shenyang/Liaoning",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "haikou_tropical_midrise",
        "elements": {
            "climate_zone": "tropical", "terrain_type": "urban_flat",
            "vegetation_zone": "tropical_rainforest", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "tree_species": ["coconut_palm"],
            "elevation_estimate_m": 5,
        },
        "true_lat": 20.02, "true_lng": 110.34,
        "expected_region": "Haikou/Hainan",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "nanchang_subtropical_midrise",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "medium_city",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "elevation_estimate_m": 30,
        },
        "true_lat": 28.68, "true_lng": 115.88,
        "expected_region": "Nanchang/Jiangxi",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "xiamen_coastal_midrise",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "architecture_style": "arcade",
            "tree_species": ["banyan"],
            "elevation_estimate_m": 20,
        },
        "true_lat": 24.47, "true_lng": 118.08,
        "expected_region": "Xiamen/Fujian",
        "scene_category": "city_fingerprint",
    },
    {
        "name": "suzhou_willow_hui_style",
        "elements": {
            "climate_zone": "subtropical", "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen", "urbanization": "metropolis",
            "building_height": "mid_rise", "pavement_type": "grey_concrete",
            "language_script": "simplified_chinese",
            "architecture_style": "hui_style",
            "tree_species": ["willow"],
            "water_visible": True,
            "elevation_estimate_m": 5,
        },
        "true_lat": 31.30, "true_lng": 120.63,
        "expected_region": "Suzhou/Jiangsu",
        "scene_category": "city_fingerprint",
    },

    # ── Famous hiking routes ────────────────────────────────────────────
    {
        "name": "aotai_boulder_field",
        "elements": {
            "climate_zone": "temperate", "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "mountain_rock_type": "granite_spheroidal",
            "landform_detail": "granite_boulder_field",
            "elevation_estimate_m": 3400,
        },
        "true_lat": 33.95, "true_lng": 107.75,
        "expected_region": "鳌太线 Taibai Shan (Shaanxi)",
        "scene_category": "granite_mountain",
    },
    {
        "name": "yading_rock_line",
        "elements": {
            "climate_zone": "alpine", "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "mountain_rock_type": "alpine_lakes",
            "language_script": "tibetan",
            "architecture_style": "tibetan_stone",
            "elevation_estimate_m": 4200,
        },
        "true_lat": 28.40, "true_lng": 100.35,
        "expected_region": "洛克线 Yading (Sichuan)",
        "scene_category": "snow_mountain",
    },
    {
        "name": "langta_tianshan_valley",
        "elements": {
            "climate_zone": "alpine", "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
            "mountain_rock_type": "snow_peaks_glaciers",
            "landform_detail": "river_canyon",
            "elevation_estimate_m": 3000,
        },
        "true_lat": 43.40, "true_lng": 86.50,
        "expected_region": "狼塔线 Central Tianshan (Xinjiang)",
        "scene_category": "snow_mountain",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lng1, lat2, lng2):
    import math
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def evaluate_result(result: FusionResult, true_lat: float, true_lng: float,
                    elements: dict, sensor_elevation: Optional[float] = None) -> dict:
    """Compute all metrics for a single test case."""
    error_km = haversine_km(result.latitude, result.longitude, true_lat, true_lng)
    within_uncertainty = error_km <= result.uncertainty_km

    # Count elements
    n_strong = sum(1 for k in elements if k not in ("language_script", "architecture_style"))
    n_total = len(elements)

    # Check if compound scene was triggered
    compound_triggered = "COMPOUND" in result.explanation_text()

    # Check if elevation filter was active
    elevation_active = "ALTITUDE" in result.explanation_text()

    return {
        "error_km": round(error_km, 1),
        "uncertainty_km": round(result.uncertainty_km, 0),
        "confidence": round(result.confidence, 3),
        "within_uncertainty": within_uncertainty,
        "pred_lat": round(result.latitude, 4),
        "pred_lng": round(result.longitude, 4),
        "true_lat": true_lat, "true_lng": true_lng,
        "n_strong_elements": n_strong,
        "n_total_elements": n_total,
        "compound_triggered": compound_triggered,
        "elevation_active": elevation_active,
        "n_candidates": len(result.candidate_region.bboxes) if result.candidate_region else 0,
    }


def print_summary(results: list[dict]):
    """Print comprehensive benchmark summary."""
    if not results:
        print("No results to summarize.")
        return

    errors = [r["error_km"] for r in results]
    uncertainties = [r["uncertainty_km"] for r in results]
    confidences = [r["confidence"] for r in results]

    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY ({len(results)} test cases)")
    print(f"{'='*70}")

    print(f"\n  Error (km):")
    print(f"    Mean:   {np.mean(errors):.0f}")
    print(f"    Median: {np.median(errors):.0f}")
    print(f"    Min:    {np.min(errors):.0f}")
    print(f"    Max:    {np.max(errors):.0f}")
    print(f"    Std:    {np.std(errors):.0f}")

    print(f"\n  Uncertainty (km):")
    print(f"    Mean:   {np.mean(uncertainties):.0f}")
    print(f"    Median: {np.median(uncertainties):.0f}")

    print(f"\n  Confidence:")
    print(f"    Mean:   {np.mean(confidences):.2%}")
    print(f"    Median: {np.median(confidences):.2%}")

    within = sum(1 for r in results if r["within_uncertainty"])
    print(f"\n  Calibration: {within}/{len(results)} within uncertainty radius")

    compound_hit = sum(1 for r in results if r["compound_triggered"])
    print(f"  Compound scene triggered: {compound_hit}/{len(results)}")

    elevation_hit = sum(1 for r in results if r["elevation_active"])
    print(f"  Elevation filter active: {elevation_hit}/{len(results)}")

    # Per-category breakdown
    print(f"\n  By scene category:")
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.get("scene_category", "unknown")].append(r["error_km"])
    for cat in sorted(by_cat.keys()):
        cat_errors = by_cat[cat]
        print(f"    {cat:20s}: mean {np.mean(cat_errors):6.0f} km  "
              f"(n={len(cat_errors)})")

    # Error distribution
    print(f"\n  Error distribution:")
    bins = [0, 50, 100, 200, 500, 1000, 99999]
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = sum(1 for e in errors if lo <= e < hi)
        bar = "=" * (count * 2)
        label = f"<{hi} km" if hi < 99999 else f">={lo} km"
        print(f"    {lo:5d}-{str(hi):5s}: {count:2d} {bar}")


# ═══════════════════════════════════════════════════════════════════════════
# Mock mode: run fusion on pre-defined element sets
# ═══════════════════════════════════════════════════════════════════════════

def run_mock_benchmark(cases: list[dict] = None, v3: bool = False) -> list[dict]:
    """Run fusion pipeline on mock test cases with DEM sensor injection."""
    if cases is None:
        cases = MOCK_CASES

    dem = get_dem()
    climate = get_climate()
    results = []

    print(f"{'='*70}")
    print(f"MOCK BENCHMARK: {len(cases)} test cases with DEM + climate sensors")
    print(f"{'='*70}")

    for i, case in enumerate(cases):
        name = case["name"]
        elements = case["elements"]
        true_lat = case["true_lat"]
        true_lng = case["true_lng"]
        month = case.get("month", 7)  # default: July (summer)

        # Get DEM elevation at ground truth location (simulating barometric sensor)
        sensor_elev = dem.query(true_lat, true_lng) if dem else None

        # Get approximate climate at ground truth (simulating temp/humidity sensors)
        sensor_temp = None
        sensor_humid = None
        if climate and sensor_elev is not None:
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, sensor_elev, month)
            sensor_temp = round((t_lo + t_hi) / 2, 1)
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            sensor_humid = round((h_lo + h_hi) / 2, 1)

        print(f"\n[{i+1}/{len(cases)}] {name}")
        print(f"  Expected: {case.get('expected_region', 'N/A')}")
        print(f"  True: ({true_lat:.2f}, {true_lng:.2f})")
        sensor_str = f"DEM: {sensor_elev:.0f}m" if sensor_elev else "No DEM"
        if sensor_temp:
            sensor_str += f", {sensor_temp:.0f}C, {sensor_humid:.0f}%RH"
        print(f"  Sensor: {sensor_str}")
        if case.get("geocot_prediction"):
            gp = case["geocot_prediction"]
            print(f"  GeoCoT (biased): ({gp[0]:.2f}, {gp[1]:.2f})")

        # Pass GeoCoT prediction if test case has one (for bias failure testing)
        geocot_pred = case.get("geocot_prediction")

        t0 = time.time()
        if v3:
            result = fuse_elements_v3(
                elements,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=sensor_temp,
                sensor_humidity_pct=sensor_humid,
                geocot_prediction=geocot_pred,
            )
        else:
            result = fuse_elements(
                elements,
                sensor_elevation_m=sensor_elev,
                sensor_temperature_c=sensor_temp,
                sensor_humidity_pct=sensor_humid,
                geocot_prediction=geocot_pred,
            )
        elapsed = time.time() - t0

        metrics = evaluate_result(result, true_lat, true_lng, elements, sensor_elev)
        metrics["name"] = name
        metrics["expected_region"] = case.get("expected_region", "")
        metrics["scene_category"] = case.get("scene_category", "unknown")
        metrics["elapsed_ms"] = round(elapsed * 1000, 1)
        metrics["sensor_elevation_m"] = round(sensor_elev, 0) if sensor_elev else None

        print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f}) "
              f"+-{result.uncertainty_km:.0f}km")
        print(f"  Error: {metrics['error_km']:.0f} km  "
              f"Confidence: {metrics['confidence']:.1%}  "
              f"({elapsed*1000:.1f}ms)")
        if result.candidate_region and result.candidate_region.dropped_elements:
            print(f"  Dropped: {result.candidate_region.dropped_elements}")

        results.append(metrics)

    print_summary(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Simple mock mode: convert elements → simple answers → elements → fusion
# Measures information loss from switching to simplified visual prompt
# ═══════════════════════════════════════════════════════════════════════════

def run_simple_mock_benchmark(cases: list[dict] = None) -> list[dict]:
    """Run fusion with simplified visual pipeline on mock test cases.

    This simulates what happens when VLM answers simple visual questions
    instead of outputting structured geographic JSON:
    1. Perfect elements → simple letter answers (information compression)
    2. Simple answers → GeoKB elements (reverse mapping)
    3. Fusion pipeline with sensor pre-filter

    The gap between this and the original mock benchmark is the
    "information loss" from simplifying the VLM task.
    """
    if cases is None:
        cases = MOCK_CASES

    dem = get_dem()
    climate = get_climate()
    results = []

    print(f"{'='*70}")
    print(f"SIMPLE MOCK BENCHMARK: {len(cases)} test cases")
    print(f"Elements → Simple Answers → Elements → Fusion")
    print(f"{'='*70}")

    for i, case in enumerate(cases):
        name = case["name"]
        elements = case["elements"]
        true_lat = case["true_lat"]
        true_lng = case["true_lng"]
        month = case.get("month", 7)

        sensor_elev = dem.query(true_lat, true_lng) if dem else None
        sensor_temp = None
        sensor_humid = None
        if climate and sensor_elev is not None:
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, sensor_elev, month)
            sensor_temp = round((t_lo + t_hi) / 2, 1)
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            sensor_humid = round((h_lo + h_hi) / 2, 1)

        # ── Convert elements → simple answers → elements ──────────────
        simple_answers = _elements_to_simple_answers(elements)
        rebuilt_elements = simple_answers_to_elements(simple_answers)

        # Count information loss
        orig_keys = set(elements.keys())
        rebuilt_keys = set(rebuilt_elements.keys())
        lost_keys = orig_keys - rebuilt_keys
        gained_keys = rebuilt_keys - orig_keys

        print(f"\n[{i+1}/{len(cases)}] {name}")
        print(f"  Expected: {case.get('expected_region', 'N/A')}")
        print(f"  True: ({true_lat:.2f}, {true_lng:.2f})")
        print(f"  Simple answers: {json.dumps(simple_answers, ensure_ascii=False)}")
        print(f"  Original elements: {len(elements)} → Rebuilt: {len(rebuilt_elements)}")
        if lost_keys:
            print(f"  Lost: {sorted(lost_keys)}")
        if gained_keys:
            print(f"  Gained: {sorted(gained_keys)}")

        t0 = time.time()
        result = fuse_elements(
            rebuilt_elements,
            sensor_elevation_m=sensor_elev,
            sensor_temperature_c=sensor_temp,
            sensor_humidity_pct=sensor_humid,
        )
        elapsed = time.time() - t0

        metrics = evaluate_result(result, true_lat, true_lng, rebuilt_elements, sensor_elev)
        metrics["name"] = name
        metrics["expected_region"] = case.get("expected_region", "")
        metrics["scene_category"] = case.get("scene_category", "unknown")
        metrics["elapsed_ms"] = round(elapsed * 1000, 1)
        metrics["sensor_elevation_m"] = round(sensor_elev, 0) if sensor_elev else None
        metrics["n_original_elements"] = len(elements)
        metrics["n_rebuilt_elements"] = len(rebuilt_elements)
        metrics["lost_keys"] = sorted(lost_keys)

        print(f"  Result: ({result.latitude:.2f}, {result.longitude:.2f}) "
              f"+-{result.uncertainty_km:.0f}km")
        print(f"  Error: {metrics['error_km']:.0f} km  "
              f"Confidence: {metrics['confidence']:.1%}  "
              f"({elapsed*1000:.1f}ms)")

        results.append(metrics)

    print_summary(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# GeoCoT mode: full pipeline with VLM inference
# ═══════════════════════════════════════════════════════════════════════════

def _diverse_sample(test_info: dict, n: int) -> list:
    """Sample n images with maximum geographic diversity.

    Picks 1 image from each 0.5-degree cell first, then fills remaining
    quota from the most image-rich cells. This ensures we test
    generalization across China rather than fitting to one region.
    """
    # Group images by 0.5-degree cell
    cells = defaultdict(list)
    for path, (lat, lng) in test_info.items():
        cell = (round(lat * 2) / 2, round(lng * 2) / 2)
        cells[cell].append(path)

    # Sort cells by image count (most images first for fill-in)
    sorted_cells = sorted(cells.items(), key=lambda x: -len(x[1]))

    sampled = []
    # Phase 1: pick 1 image from each cell
    for cell, paths in sorted_cells:
        sampled.append(paths[0])
        if len(sampled) >= n:
            break

    # Phase 2: if we need more, pick 2nd image from each cell
    if len(sampled) < n:
        for cell, paths in sorted_cells:
            if len(paths) > 1 and paths[1] not in sampled:
                sampled.append(paths[1])
                if len(sampled) >= n:
                    break

    # Sort by latitude for a geographic tour
    sampled_info = [(p, test_info[p][0], test_info[p][1]) for p in sampled[:n]]
    sampled_info.sort(key=lambda x: x[1])  # Sort by latitude
    return [p for p, _, _ in sampled_info]


def _extract_elements_from_geocot(stage_outputs: dict) -> tuple:
    """Extract structured elements from GeoCoT outputs for analysis."""
    from regression.element_fusion import extract_all_elements, parse_geocot_json
    macro = parse_geocot_json(stage_outputs.get("macro", ""))
    regional = parse_geocot_json(stage_outputs.get("regional", ""))
    local = parse_geocot_json(stage_outputs.get("local", ""))
    elements = extract_all_elements(macro, regional, local)
    # Count element categories
    n_visual = sum(1 for k in elements
                   if k in ("climate_zone", "terrain_type", "vegetation_zone",
                            "mountain_rock_type", "soil_color", "scene_type",
                            "landform_detail", "water_type", "rock_color"))
    n_cultural = sum(1 for k in elements
                     if k in ("language_script", "architecture_style",
                              "urbanization", "infrastructure_tags"))
    n_spatial = sum(1 for k in elements
                    if k in ("likely_provinces", "tree_species", "sky_quality"))

    # Check which elements have GeoKB mappings
    from regression.geo_kb import get_bboxes_for_element
    geo_hits = 0
    geo_misses = []
    for key, val in elements.items():
        if val is None or val == "none_visible" or val == []:
            continue
        bboxes = get_bboxes_for_element(key, val)
        if bboxes:
            geo_hits += 1
        else:
            geo_misses.append(f"{key}={val}")

    return elements, n_visual, n_cultural, n_spatial, geo_hits, geo_misses


def run_geocot_benchmark(n_images: int = 50, diverse: bool = True, lora_path: str = None,
                        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
                        load_in_4bit: bool = False, v3: bool = False) -> list[dict]:
    """Run full GeoCoT -> fusion pipeline on test images.

    Args:
        n_images: number of images to test
        diverse: if True, sample across geographic clusters (tests generalization).
                 if False, take first N images sequentially.
        lora_path: optional path to LoRA adapter weights
        load_in_4bit: use 4-bit quantization for larger models (e.g. 4B)
    """
    from PIL import Image
    from Geocot.Geocot import GeoCoTPipeline, load_qwen2vl

    print(f"Loading {model_name}...")
    model, processor, model_fn = load_qwen2vl(model_name=model_name, lora_path=lora_path,
                                               load_in_4bit=load_in_4bit)
    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "src", "Geocot", "prompts")
    pipeline = GeoCoTPipeline(model_fn, {"prompts_dir": prompts_dir})

    dem = get_dem()
    climate = get_climate()

    with open("output/regression/test_images.json") as f:
        test_info = json.load(f)

    if diverse:
        test_paths = _diverse_sample(test_info, n_images)
        print(f"Diverse sampling: {len(test_paths)} images from "
              f"{len(set((round(test_info[p][0]*2)/2, round(test_info[p][1]*2)/2) for p in test_paths))} "
              f"geographic clusters")
    else:
        test_paths = list(test_info.keys())[:n_images]

    # Track generalization stats
    all_elements_extracted = []
    all_geo_hits = []
    all_geo_misses = []
    errors_by_cause = defaultdict(list)

    results = []
    for i, img_path in enumerate(test_paths):
        name = img_path.replace("\\", "/").split("/")[-1]
        true_lat, true_lng = test_info[img_path]
        sensor_elev = dem.query(true_lat, true_lng) if dem else None

        # Climate sensor simulation
        month = 7  # July
        sensor_temp = None
        sensor_humid = None
        if climate and sensor_elev is not None:
            t_lo, t_hi = climate.estimate_temperature_range(true_lat, true_lng, sensor_elev, month)
            sensor_temp = round((t_lo + t_hi) / 2, 1)
            h_lo, h_hi = climate.estimate_humidity_range(true_lat, true_lng, month)
            sensor_humid = round((h_lo + h_hi) / 2, 1)

        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(test_paths)}] {name}")
        print(f"  True: ({true_lat:.4f}, {true_lng:.4f}) "
              f"| DEM: {sensor_elev:.0f}m | {sensor_temp:.0f}C | {sensor_humid:.0f}%RH")

        try:
            image = Image.open(img_path).convert("RGB")
            t0 = time.time()
            geocot_result = pipeline.run(image,
                                          sensor_elevation_m=sensor_elev,
                                          sensor_temperature_c=sensor_temp,
                                          sensor_humidity_pct=sensor_humid)

            stage_outputs = geocot_result.stage_outputs

            # ── Analyze extracted elements (generalization metrics) ──────
            elements, n_vis, n_cult, n_spat, geo_hits, geo_misses = \
                _extract_elements_from_geocot(stage_outputs)

            all_elements_extracted.append(len(elements))
            all_geo_hits.append(geo_hits)

            print(f"  Elements: {len(elements)} extracted "
                  f"(visual:{n_vis} cultural:{n_cult} spatial:{n_spat}) "
                  f"| GeoKB hits: {geo_hits}/{len(elements)}")
            if geo_misses:
                print(f"  GeoKB misses: {geo_misses[:4]}{'...' if len(geo_misses)>4 else ''}")

            # ── Run fusion ──────────────────────────────────────────────
            # Pass GeoCoT coordinate prediction to fusion engine
            geo_raw = geocot_result.final_prediction
            geocot_coords = None
            if geo_raw.latitude is not None and geo_raw.longitude is not None:
                geocot_coords = (geo_raw.latitude, geo_raw.longitude)
            if v3:
                fusion_result = run_fusion_pipeline_v3(
                    stage_outputs.get("macro", ""),
                    stage_outputs.get("regional", ""),
                    stage_outputs.get("local", ""),
                    sensor_elevation_m=sensor_elev,
                    sensor_temperature_c=sensor_temp,
                    sensor_humidity_pct=sensor_humid,
                    geocot_prediction=geocot_coords,
                )
            else:
                fusion_result = run_fusion_pipeline(
                    stage_outputs.get("macro", ""),
                    stage_outputs.get("regional", ""),
                    stage_outputs.get("local", ""),
                    sensor_elevation_m=sensor_elev,
                    sensor_temperature_c=sensor_temp,
                    sensor_humidity_pct=sensor_humid,
                    geocot_prediction=geocot_coords,
                )
            elapsed = time.time() - t0

            error_km = haversine_km(
                fusion_result.latitude, fusion_result.longitude,
                true_lat, true_lng,
            )

            # ── Display reasoning chain ──────────────────────────────────
            if geo_raw.latitude is not None and geo_raw.longitude is not None:
                geo_raw_error = haversine_km(geo_raw.latitude, geo_raw.longitude, true_lat, true_lng)
                print(f"  GeoCoT raw : ({geo_raw.latitude:.2f}, {geo_raw.longitude:.2f}) "
                      f"→ error {geo_raw_error:.0f} km")
            else:
                geo_raw_error = None
                print(f"  GeoCoT raw : (no coordinate prediction)")
            print(f"  Fusion     : ({fusion_result.latitude:.2f}, {fusion_result.longitude:.2f}) "
                  f"±{fusion_result.uncertainty_km:.0f}km → error {error_km:.0f} km")
            print(f"  Confidence : {fusion_result.confidence:.1%} | {elapsed:.1f}s")

            if fusion_result.candidate_region:
                cr = fusion_result.candidate_region
                print(f"  Active     : {cr.active_elements}")
                if cr.dropped_elements:
                    print(f"  Dropped    : {cr.dropped_elements}")
                print(f"  Candidates : {len(cr.bboxes)} regions")

            # Show explanation trace (filtering effects)
            explanation = fusion_result.explanation_text()
            for line in explanation.split("\n"):
                line = line.strip()
                if line and line.startswith("["):
                    print(f"  {line}")

            # ── Categorize error cause ──────────────────────────────────
            if error_km < fusion_result.uncertainty_km:
                cause = "within_uncertainty"
            elif fusion_result.candidate_region and fusion_result.candidate_region.dropped_elements:
                cause = "element_conflict"
            elif geo_misses:
                cause = "geokb_gap"
            elif len(elements) < 3:
                cause = "few_elements"
            else:
                cause = "fundamental_ambiguity"
            errors_by_cause[cause].append(error_km)

            results.append({
                "name": name,
                "true_lat": true_lat, "true_lng": true_lng,
                "pred_lat": round(fusion_result.latitude, 4),
                "pred_lng": round(fusion_result.longitude, 4),
                "error_km": round(error_km, 1),
                "geocot_raw_error_km": round(geo_raw_error, 1) if geo_raw_error else None,
                "uncertainty_km": round(fusion_result.uncertainty_km, 0),
                "confidence": round(fusion_result.confidence, 3),
                "within_uncertainty": error_km <= fusion_result.uncertainty_km,
                "compound_triggered": "COMPOUND" in explanation,
                "elevation_active": "ALTITUDE" in explanation,
                "climate_active": "CLIMATE" in explanation,
                "n_elements": len(elements),
                "geo_hits": geo_hits,
                "geo_misses": geo_misses,
                "n_candidates": len(fusion_result.candidate_region.bboxes) if fusion_result.candidate_region else 0,
                "elapsed_s": round(elapsed, 1),
                "sensor_elevation_m": round(sensor_elev, 0) if sensor_elev else None,
                "error_cause": cause,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Print comprehensive summary ─────────────────────────────────────
    print_summary(results)

    # ── Generalization-specific summary ─────────────────────────────────
    if all_elements_extracted:
        print(f"\n{'='*70}")
        print(f"GENERALIZATION METRICS")
        print(f"{'='*70}")
        print(f"  Elements extracted per image: "
              f"mean {np.mean(all_elements_extracted):.1f}, "
              f"median {np.median(all_elements_extracted):.0f}, "
              f"range [{min(all_elements_extracted)}, {max(all_elements_extracted)}]")
        print(f"  GeoKB hit rate: "
              f"mean {np.mean(all_geo_hits)/max(np.mean(all_elements_extracted),1)*100:.0f}% "
              f"(avg {np.mean(all_geo_hits):.1f}/{np.mean(all_elements_extracted):.1f} elements)")
        if all_geo_misses:
            all_misses_flat = [m for misses in all_geo_misses for m in misses]
            from collections import Counter
            miss_counts = Counter(all_misses_flat)
            print(f"  Top GeoKB misses:")
            for miss, count in miss_counts.most_common(5):
                print(f"    - {miss} ({count}x)")

        # Error by cause
        print(f"\n  Error by root cause:")
        for cause in sorted(errors_by_cause.keys()):
            errs = errors_by_cause[cause]
            print(f"    {cause:25s}: mean {np.mean(errs):6.0f} km, "
                  f"n={len(errs)}")

        # GeoCoT raw vs fusion comparison (only if GeoCoT produced coordinates)
        geo_errors = [r["geocot_raw_error_km"] for r in results if r.get("geocot_raw_error_km") is not None]
        fusion_errors = [r["error_km"] for r in results]
        if geo_errors:
            improved = sum(1 for g, f in zip(geo_errors, fusion_errors) if f < g)
            print(f"\n  Fusion vs GeoCoT raw: {improved}/{len(results)} cases improved")
            print(f"    GeoCoT raw mean error: {np.mean(geo_errors):.0f} km")
            print(f"    Fusion mean error:     {np.mean(fusion_errors):.0f} km")
        else:
            print(f"\n  GeoCoT coordinate prediction: DISABLED (sensor-first mode)")
            print(f"    Fusion mean error: {np.mean(fusion_errors):.0f} km")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="E2E fusion engine benchmark")
    parser.add_argument("--mock", action="store_true",
                        help="Run mock benchmark (no GPU needed)")
    parser.add_argument("--simple", action="store_true",
                        help="Run simple visual pipeline mock (elements→answers→elements, no GPU)")
    parser.add_argument("--geocot", action="store_true",
                        help="Run full GeoCoT + fusion pipeline (needs GPU)")
    parser.add_argument("--v3", action="store_true",
                        help="Use sensor-first v2.2 pipeline (fuse_elements_v3 with grid search)")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA adapter for GeoCoT VLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--4bit", action="store_true",
                        help="Load model in 4-bit quantization (required for 4B+ on 8GB GPU)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of images for GeoCoT mode (default: 50)")
    parser.add_argument("--output", type=str,
                        default="output/regression/fusion_e2e_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if not args.mock and not args.geocot and not args.simple:
        print("ERROR: specify --mock, --simple, or --geocot")
        sys.exit(1)

    if args.simple:
        results = run_simple_mock_benchmark()
    elif args.mock:
        results = run_mock_benchmark(v3=args.v3)
    else:
        kwargs = vars(args)
        results = run_geocot_benchmark(args.n, lora_path=args.lora, model_name=args.model,
                                        load_in_4bit=kwargs.get("4bit", False), v3=args.v3)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary = {
        "mode": "mock" if args.mock else "geocot",
        "n_cases": len(results),
        "mean_error_km": round(float(np.mean([r["error_km"] for r in results])), 1) if results else None,
        "median_error_km": round(float(np.median([r["error_km"] for r in results])), 1) if results else None,
        "mean_uncertainty_km": round(float(np.mean([r["uncertainty_km"] for r in results])), 0) if results else None,
        "within_uncertainty_pct": round(
            100 * sum(1 for r in results if r["within_uncertainty"]) / len(results), 1
        ) if results else None,
        "compound_trigger_rate": round(
            100 * sum(1 for r in results if r.get("compound_triggered")) / len(results), 1
        ) if results else None,
        "elevation_active_rate": round(
            100 * sum(1 for r in results if r.get("elevation_active")) / len(results), 1
        ) if results else None,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
