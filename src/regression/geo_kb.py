"""Geographic Knowledge Base: maps geographic elements to spatial distributions in China.

Each element (climate zone, terrain type, vegetation, architecture, etc.) maps to
one or more bounding boxes representing where that element occurs in China.

For the constraint fusion engine: each element casts a "vote" for where the image
could be. Elements are multiplied (intersection) to narrow the candidate region.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BBox:
    """Bounding box in decimal degrees."""
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float
    label: str = ""  # region name

    @property
    def center_lat(self) -> float:
        return (self.lat_min + self.lat_max) / 2

    @property
    def center_lng(self) -> float:
        return (self.lng_min + self.lng_max) / 2

    @property
    def area_km2(self) -> float:
        """Approximate area in km^2 (rough, assumes mid-latitude)."""
        dlat = (self.lat_max - self.lat_min) * 111.0
        dlng = (self.lng_max - self.lng_min) * 102.0  # ~cos(30°) * 111
        return dlat * dlng


# ═══════════════════════════════════════════════════════════════════════════
# CLIMATE ZONES
# ═══════════════════════════════════════════════════════════════════════════

CLIMATE_ZONES = {
    "tropical": [
        BBox(18.0, 22.0, 108.0, 111.5, "Hainan"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
        BBox(20.0, 23.0, 109.0, 117.5, "S Guangdong / S Guangxi coast"),
    ],
    "subtropical": [
        BBox(22.0, 33.0, 104.0, 122.5, "S of Yangtze: Guangxi to Zhejiang"),
        BBox(22.0, 31.0, 97.0, 104.0, "Yunnan"),
        BBox(29.0, 32.5, 103.0, 110.5, "Sichuan Basin / Chongqing"),
    ],
    "temperate": [
        BBox(33.0, 43.0, 105.0, 126.5, "N China Plain to NE China"),
        BBox(31.5, 39.5, 105.5, 111.5, "Shaanxi / Shanxi"),
        BBox(32.0, 43.0, 92.0, 105.0, "Gansu / Ningxia"),
    ],
    "arid": [
        BBox(34.0, 49.5, 73.0, 96.5, "Xinjiang"),
        BBox(37.5, 43.0, 96.0, 109.0, "W Inner Mongolia / Gansu corridor"),
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin / Qinghai"),
    ],
    "alpine": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet Plateau"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan highlands"),
    ],
    "boreal": [
        BBox(48.0, 54.0, 121.0, 135.5, "N Heilongjiang / Greater Khingan"),
        BBox(46.0, 50.0, 80.0, 90.0, "Altai Mountains / N Xinjiang"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# TERRAIN TYPES
# ═══════════════════════════════════════════════════════════════════════════

TERRAIN_TYPES = {
    "urban_flat": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China cities"),  # fallback
    ],
    "farmland_plain": [
        BBox(30.0, 41.0, 113.5, 122.5, "North China Plain"),
        BBox(43.0, 48.5, 123.0, 135.5, "NE China Plain"),
        BBox(29.0, 33.0, 116.0, 122.0, "Yangtze Delta"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin"),
        BBox(21.0, 24.0, 109.0, 117.5, "Pearl River Delta"),
    ],
    "rolling_hills": [
        BBox(24.0, 30.5, 113.5, 122.5, "SE China hills: Fujian/Jiangxi/Hunan/Zhejiang"),
        BBox(20.5, 26.5, 104.0, 112.5, "Guangxi/Guizhou hills"),
        BBox(30.0, 33.5, 117.0, 120.0, "S Anhui / S Jiangsu hills"),
    ],
    "sharp_mountains": [
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan Mountains: W Sichuan / NW Yunnan"),
        BBox(31.5, 39.5, 105.5, 111.5, "Qinling Mountains"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan / Altai"),
        BBox(26.5, 36.5, 78.0, 95.0, "Himalaya / Transhimalaya"),
        BBox(27.0, 31.0, 115.0, 121.0, "Wuyi / Yandang / coastal SE mountains"),
    ],
    "karst_peaks": [
        BBox(24.0, 26.5, 102.5, 112.5, "Yunnan-Guizhou-Guangxi karst (Shilin → Guilin)"),
        BBox(24.5, 30.5, 108.5, 114.5, "Hunan karst (Zhangjiajie area)"),
        BBox(20.5, 24.0, 103.5, 108.0, "N Vietnam border karst"),
    ],
    "sandstone_pillars": [
        BBox(28.5, 30.5, 109.0, 111.5, "Zhangjiajie / Wulingyuan"),
    ],
    "desert_dunes": [
        BBox(37.0, 46.0, 75.0, 90.0, "Taklamakan Desert"),
        BBox(39.0, 46.0, 80.0, 96.5, "Gurbantunggut Desert"),
        BBox(37.5, 42.5, 100.0, 107.0, "Tengger / Badain Jaran"),
        BBox(38.0, 43.0, 88.0, 94.0, "Kumtag Desert"),
    ],
    "grassland_steppe": [
        BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia / Hulunbuir"),
        BBox(32.0, 39.5, 89.0, 103.5, "Qinghai / Gansu grassland"),
        BBox(27.0, 33.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine grassland (Zoige)"),
        BBox(41.0, 49.0, 80.0, 92.0, "N Xinjiang grassland (Ili / Altai)"),
    ],
    "plateau": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet Plateau"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai Plateau"),
        BBox(34.0, 40.0, 73.0, 78.0, "Pamir Plateau"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# VEGETATION ZONES
# ═══════════════════════════════════════════════════════════════════════════

VEGETATION_ZONES = {
    "tropical_rainforest": [
        BBox(18.0, 20.5, 108.0, 111.5, "Hainan"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
        BBox(20.0, 22.0, 109.0, 112.0, "SW Guangdong / S Guangxi coast"),
    ],
    "broadleaf_evergreen": [
        BBox(22.0, 33.0, 104.0, 122.5, "S China subtropical zone"),
        BBox(21.0, 29.5, 97.5, 104.0, "Yunnan"),
        BBox(29.0, 32.5, 103.0, 110.0, "Sichuan Basin"),
    ],
    "mixed_forest": [
        BBox(31.0, 37.0, 105.5, 122.5, "Qinling-Huaihe transitional zone"),
        BBox(29.0, 33.0, 108.0, 117.0, "Mid-Yangtze mixed forest"),
        BBox(36.0, 41.0, 113.5, 120.0, "N China mixed forest"),
        BBox(43.0, 48.5, 121.5, 132.0, "NE China mixed forest (Heilongjiang/Jilin lowlands)"),
    ],
    "conifer_forest": [
        BBox(48.0, 54.0, 121.0, 135.5, "N Heilongjiang boreal conifer"),
        BBox(41.0, 48.5, 120.0, 131.5, "NE China conifer (Changbai / Khingan)"),
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan conifer (W Sichuan / NW Yunnan)"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan / Altai spruce-fir"),
        BBox(26.5, 36.5, 88.0, 97.0, "SE Tibet conifer"),
        BBox(27.0, 33.0, 114.0, 121.0, "E China montane conifer (Dabie / Huangshan / Wuyi)"),
    ],
    "alpine_meadow": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet alpine meadow"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai alpine meadow"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine (Zoige / Daocheng)"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan alpine meadow"),
    ],
    "bamboo_forest": [
        BBox(27.0, 32.0, 103.0, 108.0, "Sichuan bamboo (panda habitat)"),
        BBox(27.0, 31.0, 117.0, 122.0, "Zhejiang / Fujian bamboo"),
        BBox(24.0, 29.0, 110.0, 117.0, "Hunan / Jiangxi bamboo"),
    ],
    "desert_scrub": [
        BBox(37.0, 46.0, 75.0, 90.0, "Taklamakan"),
        BBox(37.5, 42.5, 100.0, 107.0, "Tengger desert edges"),
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin"),
    ],
    "grassland": [
        BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia grassland"),
        BBox(41.0, 46.0, 80.0, 90.0, "N Xinjiang grassland (Ili valley)"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# TREE SPECIES (more specific than vegetation zone)
# ═══════════════════════════════════════════════════════════════════════════

TREE_SPECIES = {
    "banyan": [
        BBox(21.0, 25.5, 108.0, 121.0, "S China: Guangzhou/Fuzhou/Nanning/haikou"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
    ],
    "camphor": [
        BBox(27.0, 32.0, 110.0, 122.5, "Mid-lower Yangtze: Hangzhou/Changsha/Nanchang"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin: Chengdu/Chongqing"),
    ],
    "plane_tree": [
        BBox(30.0, 32.5, 118.0, 122.0, "Shanghai/Nanjing/Hangzhou"),
        BBox(30.0, 31.5, 112.0, 115.0, "Wuhan area"),
    ],
    "gingko": [
        BBox(30.0, 41.0, 105.0, 122.5, "N China cities (planted, common)"),
    ],
    "poplar": [
        BBox(36.0, 48.5, 113.5, 135.5, "N China Plain + NE China"),
    ],
    "coconut_palm": [
        BBox(18.0, 20.5, 108.0, 111.5, "Hainan"),
        BBox(20.0, 23.0, 109.0, 117.5, "S Guangdong / S Guangxi coast"),
        BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna"),
    ],
    "chinese_red_pine": [
        BBox(22.0, 33.0, 104.0, 122.5, "Widespread S China (masson pine)"),
    ],
    "bamboo": [
        BBox(27.0, 32.0, 103.0, 108.0, "Sichuan bamboo"),
        BBox(27.0, 31.0, 117.0, 122.0, "Zhejiang / Fujian bamboo"),
        BBox(24.0, 29.0, 110.0, 117.0, "Hunan / Jiangxi bamboo"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# ARCHITECTURE STYLES
# ═══════════════════════════════════════════════════════════════════════════

ARCHITECTURE_STYLES = {
    "hui_style": [
        BBox(29.0, 31.5, 116.5, 119.5, "S Anhui / N Jiangxi / W Zhejiang (Huizhou)"),
    ],
    "tibetan_stone": [
        BBox(26.5, 34.0, 88.0, 99.5, "Tibet"),
        BBox(26.0, 33.0, 97.0, 104.0, "W Sichuan (Ganzi/ABA)"),
        BBox(27.0, 29.5, 98.5, 101.0, "NW Yunnan (Diqing/Shangri-La)"),
        BBox(31.5, 39.5, 89.0, 103.5, "Qinghai / S Gansu"),
    ],
    "stilt_house": [
        BBox(27.0, 31.0, 107.0, 112.0, "W Hunan / Guizhou / Chongqing (Tujia/Miao)"),
        BBox(23.5, 26.5, 104.0, 108.0, "S Guizhou"),
    ],
    "tulou": [
        BBox(23.5, 26.0, 116.0, 118.0, "SW Fujian (Longyan/Zhangzhou)"),
    ],
    "arcade": [
        BBox(22.5, 24.5, 113.0, 115.0, "Guangzhou / Pearl River Delta"),
        BBox(23.5, 26.0, 116.0, 121.0, "S Fujian (Xiamen/Quanzhou/Zhangzhou)"),
        BBox(21.0, 23.5, 108.0, 110.5, "Guangxi coast (Beihai)"),
        BBox(19.0, 20.5, 109.5, 111.5, "Hainan"),
    ],
    "courtyard": [
        BBox(39.0, 41.0, 115.5, 117.5, "Beijing"),
        BBox(36.0, 42.5, 113.5, 120.0, "Hebei / N China"),
    ],
    "shikumen": [
        BBox(30.5, 31.5, 121.0, 122.0, "Shanghai"),
    ],
    "traditional_official": [
        BBox(18.0, 54.0, 73.0, 135.5, "Nationwide (official/temple style)"),
    ],
    "modern_glass": [
        BBox(18.0, 54.0, 73.0, 135.5, "All major Chinese cities"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# LANGUAGE SCRIPTS
# ═══════════════════════════════════════════════════════════════════════════

LANGUAGE_SCRIPTS = {
    "simplified_chinese": [
        BBox(18.0, 54.0, 73.0, 135.5, "Mainland China"),
    ],
    "traditional_chinese": [
        BBox(21.5, 25.5, 120.0, 122.5, "Taiwan"),
        BBox(22.0, 23.0, 113.5, 114.5, "Hong Kong"),
        BBox(22.0, 22.5, 113.0, 113.5, "Macau"),
    ],
    "tibetan": [
        BBox(26.5, 34.0, 88.0, 99.5, "Tibet"),
        BBox(31.5, 39.5, 89.0, 103.5, "Qinghai / S Gansu"),
        BBox(26.0, 33.0, 97.0, 104.0, "W Sichuan (Ganzi/ABA)"),
        BBox(27.0, 29.5, 98.5, 101.0, "NW Yunnan"),
    ],
    "uyghur_arabic": [
        BBox(34.0, 49.5, 73.0, 96.5, "Xinjiang"),
    ],
    "mongolian": [
        BBox(37.5, 51.0, 97.0, 126.5, "Inner Mongolia"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# MOUNTAIN ROCK TYPES
# ═══════════════════════════════════════════════════════════════════════════

MOUNTAIN_ROCK_TYPES = {
    "granite_spheroidal": [
        BBox(29.5, 31.0, 117.5, 119.0, "Huangshan / Sanqingshan"),
        BBox(34.0, 35.0, 109.5, 111.0, "Huashan"),
        BBox(27.5, 29.0, 117.5, 118.5, "Wuyishan"),
        BBox(28.5, 30.0, 115.5, 117.0, "Lushan"),
        BBox(32.0, 33.0, 110.5, 111.5, "Wudangshan"),
        # High-altitude granite massifs
        BBox(33.5, 34.5, 107.0, 108.0, "Taibai Shan / Qinling granite (3767m)"),
        BBox(26.0, 32.0, 97.0, 104.0, "Hengduan granite (Gongga / Siguniang massif)"),
        BBox(27.0, 30.0, 88.0, 93.0, "E Himalaya granite (Namcha Barwa / Kangri)"),
        BBox(41.0, 43.0, 86.0, 89.0, "Tianshan granite (Bogda / Tomur)"),
    ],
    "quartz_sandstone_pillars": [
        BBox(28.5, 30.0, 109.5, 111.0, "Zhangjiajie / Wulingyuan"),
    ],
    "limestone_karst": [
        BBox(24.0, 26.0, 109.5, 111.5, "Guilin / Yangshuo"),
        BBox(24.5, 26.5, 104.0, 106.0, "Shilin / S Guizhou karst"),
    ],
    "red_sandstone_danxia": [
        BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan (Danxia)"),
        BBox(38.5, 39.5, 99.5, 101.0, "Zhangye Danxia"),
        BBox(28.0, 29.0, 116.5, 117.5, "Longhushan"),
        BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan (Guangdong)"),
    ],
    "snow_peaks_glaciers": [
        BBox(29.0, 30.5, 101.0, 103.0, "Gongga / Siguniang / Minya Konka"),
        BBox(27.0, 29.0, 98.0, 100.5, "Meili Snow Mountain / Yulong / Haba"),
        BBox(27.5, 29.0, 85.5, 87.5, "Everest / Cho Oyu region"),
        BBox(30.0, 32.0, 90.0, 95.0, "Nyainqentanglha / Namcha Barwa"),
        BBox(41.0, 43.5, 80.0, 89.0, "Tianshan / Tomur / Bogda"),
    ],
    "alpine_lakes": [
        BBox(32.5, 34.0, 103.0, 104.5, "Jiuzhaigou / Huanglong"),
        BBox(28.0, 29.5, 99.5, 101.0, "Daocheng Yading"),
        BBox(27.5, 28.5, 100.5, 101.5, "Lugu Lake"),
        BBox(30.5, 32.0, 90.0, 91.5, "Namtso / Yamdrok"),
        BBox(48.5, 49.0, 87.0, 87.5, "Kanas Lake"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# SOIL COLORS
# ═══════════════════════════════════════════════════════════════════════════

SOIL_COLORS = {
    "red": [
        BBox(20.0, 30.0, 104.0, 122.5, "S China red soil zone"),
        BBox(21.0, 29.5, 97.5, 104.0, "Yunnan red soil"),
    ],
    "yellow": [
        BBox(24.0, 33.0, 108.5, 122.5, "Mid-China yellow-brown soil"),
        BBox(24.5, 30.5, 103.5, 108.5, "Guizhou / Chongqing yellow soil"),
    ],
    "yellow_brown": [
        BBox(24.0, 33.0, 108.5, 122.5, "Mid-China yellow-brown soil"),
        BBox(24.5, 30.5, 103.5, 108.5, "Guizhou / Chongqing yellow soil"),
    ],
    "brown": [
        BBox(36.0, 48.5, 113.5, 135.5, "NE China dark brown soil"),
        BBox(32.0, 43.0, 92.0, 113.5, "NW China / Gansu brown soil"),
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan mountain brown soil"),
    ],
    "black": [
        BBox(43.0, 48.5, 121.5, 131.5, "NE China black soil (Heilongjiang/Jilin)"),
    ],
    "grey": [
        BBox(37.0, 49.5, 75.0, 96.5, "Xinjiang grey desert soil"),
        BBox(37.5, 42.5, 96.0, 107.0, "W Inner Mongolia grey soil"),
    ],
    "loess_yellow": [
        BBox(34.5, 41.0, 105.5, 114.5, "Loess Plateau (Shaanxi/Shanxi/Gansu)"),
    ],
    "white": [
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin saline soil"),
        BBox(39.0, 42.0, 75.0, 79.0, "W Tarim saline soil"),
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# SKY / AIR QUALITY
# ═══════════════════════════════════════════════════════════════════════════

SKY_QUALITY = {
    "clear_blue": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan highlands"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai"),
        BBox(41.0, 49.0, 80.0, 96.5, "Xinjiang highlands"),
    ],
    "grey_hazy": [
        BBox(30.0, 41.0, 113.5, 122.5, "N China Plain (Beijing/Hebei/Henan)"),
        BBox(22.0, 33.0, 104.0, 122.5, "S China humid cities"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin (Chengdu/Chongqing fog)"),
    ],
    "thick_fog": [
        BBox(29.0, 33.0, 108.0, 122.5, "Yangtze River fog zone"),
        BBox(29.0, 32.5, 103.0, 108.0, "Sichuan Basin (Chongqing/Chengdu fog)"),
        BBox(29.5, 31.0, 117.5, 119.0, "Huangshan cloud sea"),
    ],
    "dusty_yellow": [
        BBox(34.0, 43.0, 92.0, 109.0, "Gansu / Inner Mongolia (spring sandstorms)"),
        BBox(34.0, 42.5, 75.0, 92.0, "Xinjiang dust"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPOUND SCENES — pre-computed intersections of terrain+vegetation+soil+climate
# These represent specific landscape types that only co-occur in narrow regions.
# Each entry has "conditions" (category→value pairs that must ALL match) and
# tight bboxes. Used in fusion engine as additional hard constraints.
# ═══════════════════════════════════════════════════════════════════════════

COMPOUND_SCENES = {
    # ── Karst landscapes ───────────────────────────────────────────────
    "guilin_yangshuo_karst": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "red",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(24.0, 26.0, 109.5, 111.5, "Guilin-Yangshuo tower karst (Li River)"),
        ],
    },
    "shilin_stone_forest": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "red",
            "scene_type": "mountain_trail_karst",
        },
        "bboxes": [
            BBox(24.5, 26.5, 103.5, 105.0, "Shilin stone forest (Yunnan)"),
        ],
    },
    "yunnan_karst_highland": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "subtropical",
            "soil_color": "red",
            "scene_type": "mountain_trail_karst",
        },
        "bboxes": [
            BBox(24.0, 26.0, 102.5, 105.0, "Yunnan high karst (Shilin / Puzhehei)"),
        ],
    },

    # ── Sandstone landscapes ───────────────────────────────────────────
    "zhangjiajie_sandstone_pillars": {
        "conditions": {
            "terrain_type": "sandstone_pillars",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(28.5, 30.0, 109.0, 111.5, "Zhangjiajie / Wulingyuan quartz sandstone"),
        ],
    },

    # ── Granite landscapes (subtropical) ───────────────────────────────
    "huangshan_granite_pines": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(29.5, 31.0, 117.5, 119.0, "Huangshan rounded granite + pines"),
        ],
    },
    "huangshan_granite_conifer": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "conifer_forest",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(29.5, 31.0, 117.5, 119.0, "Huangshan granite + conifer + yellow soil"),
        ],
    },
    "sanqingshan_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(28.5, 29.5, 117.5, 118.5, "Sanqingshan sharp granite peaks"),
        ],
    },
    "lushan_wudang_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(28.5, 33.0, 110.5, 117.0, "Lushan / Wudangshan granite"),
        ],
    },

    # ── Granite landscapes (alpine/temperate) ──────────────────────────
    "qinling_taibai_alpine_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(32.0, 35.0, 106.5, 109.0, "Qinling Taibai-Shan alpine granite + conifer"),
        ],
    },
    "hengduan_granite_massif": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(28.0, 32.0, 97.0, 104.0, "Hengduan granite massif (Gongga / Siguniang / Minya Konka)"),
        ],
    },

    # ── Granite landscapes (temperate) ──────────────────────────────────
    "huashan_qinling_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(33.5, 35.5, 109.0, 111.0, "Huashan / Qinling granite (Shaanxi)"),
        ],
    },

    # ── Snow mountain landscapes ───────────────────────────────────────
    "gongga_siguniang_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(29.0, 30.5, 101.0, 103.0, "Gongga / Siguniang / Minya Konka snow peaks"),
        ],
    },
    "meili_yulong_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(27.0, 29.5, 98.0, 101.0, "Meili Snow Mtn / Yulong / Haba"),
        ],
    },
    "everest_chooyu_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "vegetation_zone": "alpine_meadow",
            "language_script": "tibetan",
            "landform_detail": "glacial_valley",
        },
        "bboxes": [
            BBox(27.5, 29.0, 85.5, 87.5, "Everest / Cho Oyu region (Himalaya)"),
        ],
    },
    "nyainqentanglha_snow": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "vegetation_zone": "alpine_meadow",
            "language_script": "tibetan",
            "terrain_type": "sharp_mountains",
            "landform_detail": "plateau_plain",
        },
        "bboxes": [
            BBox(30.0, 32.0, 90.0, 95.0, "Nyainqentanglha / Namcha Barwa"),
        ],
    },
    "tianshan_snow_alpine": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.0, 44.0, 80.0, 89.0, "Tianshan / Tomur / Bogda snow + glacier"),
        ],
    },

    # ── Alpine lake landscapes ─────────────────────────────────────────
    "daocheng_yading_lakes": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(28.0, 29.5, 99.5, 101.0, "Daocheng Yading alpine lakes + conifer"),
        ],
    },
    "jiuzhaigou_huanglong_lakes": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(32.5, 34.0, 103.0, 105.0, "Jiuzhaigou / Huanglong travertine lakes"),
        ],
    },
    "kanas_lake": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "boreal",
        },
        "bboxes": [
            BBox(48.0, 49.0, 87.0, 88.0, "Kanas Lake (Altai)"),
        ],
    },

    # ── Desert / arid landscapes ───────────────────────────────────────
    "qaidam_basin_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "plateau",
        },
        "bboxes": [
            BBox(34.0, 39.0, 88.0, 97.0, "Qaidam Basin desert plateau"),
        ],
    },
    "taklamakan_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
        },
        "bboxes": [
            BBox(37.0, 43.0, 75.0, 90.0, "Taklamakan Desert core"),
        ],
    },

    # ── Grassland / wetland landscapes ─────────────────────────────────
    "zoige_alpine_wetland": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "terrain_type": "grassland_steppe",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(32.0, 34.5, 101.0, 104.0, "Zoige alpine wetland (NW Sichuan)"),
        ],
    },
    "inner_mongolia_grassland": {
        "conditions": {
            "vegetation_zone": "grassland",
            "terrain_type": "grassland_steppe",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia / Hulunbuir grassland"),
        ],
    },
    "east_inner_mongolia_arid_steppe": {
        "conditions": {
            "vegetation_zone": "grassland",
            "terrain_type": "grassland_steppe",
            "climate_zone": "arid",
            "language_script": "mongolian",
        },
        "bboxes": [
            BBox(41.0, 46.0, 110.0, 120.0, "E Inner Mongolia arid steppe (Xilingol)"),
        ],
    },

    # ── Inner Mongolia grassland mirror competition ───────────────────────
    # inner_mongolia_grassland covers the entire 41-51N steppe belt. These
    # mirrors create healthy competition for different steppe sub-regions:
    # Hulunbuir (meadow steppe, lush, many rivers) vs Xilingol (typical
    # steppe, drier, classic rolling grassland). All share temperate +
    # grassland + grassland_steppe. When VLM can't distinguish → DISPERSE.
    "hulunbuir_meadow_steppe": {
        "conditions": {
            "vegetation_zone": "grassland",
            "terrain_type": "grassland_steppe",
            "climate_zone": "temperate",
            "language_script": "mongolian",
        },
        "bboxes": [
            BBox(47.5, 51.0, 117.0, 126.0, "Hulunbuir meadow steppe (E Inner Mongolia)"),
        ],
    },
    "xilingol_temperate_steppe": {
        "conditions": {
            "vegetation_zone": "grassland",
            "terrain_type": "grassland_steppe",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(42.0, 46.5, 110.0, 120.0, "Xilingol typical steppe (C Inner Mongolia)"),
        ],
    },

    # ── Tropical landscapes ────────────────────────────────────────────
    "xishuangbanna_tropical": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_rainforest",
            "terrain_type": "karst_peaks",
        },
        "bboxes": [
            BBox(21.0, 22.5, 100.0, 102.0, "Xishuangbanna tropical rainforest + karst"),
        ],
    },
    "hainan_tropical": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_rainforest",
        },
        "bboxes": [
            BBox(18.0, 20.5, 108.5, 111.5, "Hainan tropical"),
        ],
    },

    # ── Danxia landscapes ──────────────────────────────────────────────
    "danxiashan_red_sandstone": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan (Guangdong)"),
        ],
    },
    "zhangye_danxia": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "arid",
        },
        "bboxes": [
            BBox(38.5, 39.5, 99.5, 101.0, "Zhangye Danxia (Gansu corridor)"),
        ],
    },
    "wuyishan_danxia": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan Danxia + tea terraces"),
        ],
    },

    # ── Volcanic / special landscapes ──────────────────────────────────
    "changbaishan_volcanic": {
        "conditions": {
            "climate_zone": "temperate",
            "vegetation_zone": "conifer_forest",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.5, 42.5, 127.0, 129.0, "Changbaishan volcanic + crater lake"),
        ],
    },

    # ── Bamboo landscapes ──────────────────────────────────────────────
    "sichuan_bamboo_forest": {
        "conditions": {
            "vegetation_zone": "bamboo_forest",
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(27.0, 32.0, 103.0, 108.0, "Sichuan bamboo (panda habitat)"),
        ],
    },
    "zhejiang_fujian_bamboo": {
        "conditions": {
            "vegetation_zone": "bamboo_forest",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(27.0, 31.0, 117.0, 122.0, "Zhejiang / Fujian bamboo mountains"),
        ],
    },

    # ── Tropical forest variants ──────────────────────────────────────
    "xishuangbanna_tropical_hills": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_rainforest",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(21.0, 23.0, 100.0, 102.0, "Xishuangbanna tropical hills"),
        ],
    },

    # ── Tibet specific ────────────────────────────────────────────────
    "lhasa_valley": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "plateau",
            "vegetation_zone": "alpine_meadow",
            "sky_quality": "clear_blue",
        },
        "bboxes": [
            BBox(29.0, 30.5, 90.0, 92.0, "Lhasa / Yarlung Tsangpo valley"),
        ],
    },
    "everest_high_desert": {
        "conditions": {
            "climate_zone": "alpine",
            "vegetation_zone": "desert_scrub",
            "terrain_type": "plateau",
        },
        "bboxes": [
            BBox(28.0, 33.0, 80.0, 88.0, "W Tibet high-altitude cold desert"),
        ],
    },

    # ── Desert variants ───────────────────────────────────────────────
    "qaidam_desert_dunes": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
            "scene_type": "desert",
        },
        "bboxes": [
            BBox(35.0, 39.5, 88.0, 97.0, "Qaidam Basin dunes"),
        ],
    },
    "qaidam_tibetan_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(35.0, 38.0, 91.0, 96.0, "Qaidam core desert + Tibetan script"),
        ],
    },
    "gobi_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "soil_color": "grey",
        },
        "bboxes": [
            BBox(40.0, 47.0, 95.0, 107.0, "Gobi Desert (S Mongolia / N Gansu)"),
        ],
    },

    # ── Junggar / Turpan deserts ─────────────────────────────────────────
    # Gurbantunggut (古尔班通古特) in Junggar Basin — China's 2nd largest
    # desert. Fixed/semi-fixed dunes, distinct from Taklamakan's mobile dunes.
    # Turpan Depression (吐鲁番盆地) — lowest point in China (-154m), flaming
    # mountain (火焰山), unique arid basin with grape valleys.
    "junggar_desert": {
        "conditions": {
            "vegetation_zone": "desert_scrub",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
        },
        "bboxes": [
            BBox(44.0, 47.5, 85.0, 91.0, "Junggar Basin / Gurbantunggut Desert"),
        ],
    },
    "turpan_depression": {
        "conditions": {
            "vegetation_zone": "sparse",
            "climate_zone": "arid",
            "terrain_type": "desert_dunes",
        },
        "bboxes": [
            BBox(42.0, 43.5, 88.5, 91.0, "Turpan Depression / Flaming Mountain"),
        ],
    },

    # ── NE China forest ───────────────────────────────────────────────
    "ne_china_conifer_forest": {
        "conditions": {
            "vegetation_zone": "conifer_forest",
            "soil_color": "black",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(41.0, 48.0, 121.0, 132.0, "NE China mixed-conifer forest belt"),
        ],
    },
    "changbaishan_broadleaf": {
        "conditions": {
            "climate_zone": "temperate",
            "vegetation_zone": "mixed_forest",
            "terrain_type": "sharp_mountains",
            "soil_color": "black",
        },
        "bboxes": [
            BBox(41.0, 43.0, 126.0, 129.0, "Changbaishan mixed forest"),
        ],
    },

    # ── Canyon landscapes ─────────────────────────────────────────────
    "tiger_leaping_gorge": {
        "conditions": {
            "landform_detail": "river_canyon",
            "climate_zone": "alpine",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(26.5, 28.0, 99.5, 101.0, "Tiger Leaping Gorge / Nu River canyon"),
        ],
    },
    "three_gorges": {
        "conditions": {
            "landform_detail": "river_canyon",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(30.0, 31.5, 108.0, 111.5, "Three Gorges (Yangtze) canyon"),
        ],
    },

    # ── Urban compound scenes ─────────────────────────────────────────
    "guangzhou_arcade_banyan": {
        "conditions": {
            "architecture_style": "arcade",
            "tree_species": "banyan",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(23.0, 24.5, 112.5, 114.5, "Guangzhou / Foshan arcade + banyan"),
        ],
    },
    "shanghai_shikumen_planetree": {
        "conditions": {
            "architecture_style": "shikumen",
            "tree_species": "plane_tree",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(31.0, 31.5, 121.2, 121.7, "Shanghai shikumen + plane tree lanes"),
        ],
    },
    "beijing_courtyard_traditional": {
        "conditions": {
            "architecture_style": "traditional_official",
            "climate_zone": "temperate",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(39.7, 40.2, 116.1, 116.7, "Beijing courtyard / traditional"),
        ],
    },
    "shenzhen_modern_metro": {
        "conditions": {
            "urbanization": "metropolis",
            "architecture_style": "modern_glass",
            "climate_zone": "subtropical",
            "sky_quality": "grey_hazy",
        },
        "bboxes": [
            BBox(22.3, 23.5, 113.0, 114.5, "Shenzhen / Guangzhou / Dongguan"),
        ],
    },
    "harbin_ne_urban": {
        "conditions": {
            "soil_color": "black",
            "urbanization": "medium_city",
            "climate_zone": "temperate",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(45.0, 47.0, 125.5, 128.0, "Harbin / Daqing NE China black soil urban"),
        ],
    },
    "tibetan_town_highland": {
        "conditions": {
            "urbanization": "medium_city",
            "language_script": "tibetan",
            "sky_quality": "clear_blue",
        },
        "bboxes": [
            BBox(28.5, 30.5, 90.5, 92.0, "Lhasa / Shigatse area"),
            BBox(31.0, 33.0, 96.0, 97.5, "Yushu / Nangqen (S Qinghai)"),
            BBox(27.0, 29.0, 98.5, 100.0, "Shangri-La / Diqing (NW Yunnan)"),
        ],
    },

    # ── Architecture-anchored: tulou, stilt house, soviet ─────────────────
    # VLM can identify these architecture styles (regional.txt architecture_style
    # field) but GeoKB lacked compound scenes to leverage them.
    "fujian_tulou_earthen": {
        "conditions": {
            "architecture_style": "tulou",
            "climate_zone": "subtropical",
            "urbanization": "rural",
        },
        "bboxes": [
            BBox(24.0, 25.5, 116.5, 117.5, "Fujian tulou (Nanjing/Yongding)"),
        ],
    },
    "sw_china_stilt_house": {
        "conditions": {
            "architecture_style": "stilt_house",
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(25.5, 29.0, 106.0, 111.5, "SW stilt house: Guizhou/W Hunan/N Guangxi"),
        ],
    },
    "ne_china_soviet_industrial": {
        "conditions": {
            "architecture_style": "soviet_industrial",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
        },
        "bboxes": [
            BBox(41.5, 46.0, 122.0, 127.0, "NE soviet-industrial: Changchun/Harbin/Shenyang"),
            BBox(43.0, 44.5, 87.0, 88.5, "Urumqi soviet-industrial"),
        ],
    },

    # ── Coastal scenes ────────────────────────────────────────────────
    "south_china_coastal": {
        "conditions": {
            "scene_type": "coastal",
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(21.0, 27.0, 108.0, 118.0, "S China subtropical coast"),
        ],
    },
    "tropical_coastal": {
        "conditions": {
            "scene_type": "coastal",
            "climate_zone": "tropical",
            "tree_species": "coconut_palm",
        },
        "bboxes": [
            BBox(18.0, 20.5, 108.5, 111.5, "Hainan tropical coast"),
        ],
    },

    # ── Loess Plateau ─────────────────────────────────────────────────
    "loess_plateau_farmland": {
        "conditions": {
            "soil_color": "yellow",
            "climate_zone": "temperate",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(34.0, 39.0, 104.0, 112.0, "Loess Plateau (Shaanxi / Gansu / Shanxi)"),
        ],
    },

    # ── Sichuan Basin ─────────────────────────────────────────────────
    "chengdu_basin_urban": {
        "conditions": {
            "urbanization": "metropolis",
            "climate_zone": "subtropical",
            "terrain_type": "urban_flat",
            "tree_species": "camphor",
        },
        "bboxes": [
            BBox(30.3, 30.9, 103.8, 104.3, "Chengdu basin metro"),
        ],
    },
    "sichuan_basin_city": {
        # Low-threshold variant: VLM often calls Chengdu/Chongqing "medium_city"
        # instead of "metropolis" from single street photos. This catches the
        # same element combo that currently matches Nanning/Nanchang at 100%.
        "conditions": {
            "urbanization": "medium_city",
            "climate_zone": "subtropical",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(28.5, 32.5, 103.0, 110.0, "Sichuan Basin / Chongqing"),
        ],
    },
    "sichuan_basin_overcast": {
        # Sichuan Basin is famous for persistent cloud/fog trapped by
        # surrounding mountains. Subtropical + overcast urban = strong signal.
        "conditions": {
            "climate_zone": "subtropical",
            "sky_quality": "overcast_grey",
            "urbanization": "medium_city",
        },
        "bboxes": [
            BBox(28.5, 32.5, 103.0, 110.0, "Sichuan Basin overcast"),
        ],
    },
    "sichuan_red_brick_city": {
        # Red brick pavement is highly characteristic of Chengdu/Chongqing.
        # Lower threshold: accepts medium_city, doesn't require building_height.
        "conditions": {
            "pavement_type": "red_brick_tiles",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
        },
        "bboxes": [
            BBox(28.5, 32.5, 103.0, 110.0, "Sichuan Basin red-brick zone"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # GEOGRAPHIC COMPETITION MIRRORS — prevent single-region lock-in
    # ═══════════════════════════════════════════════════════════════════════
    # Problem: generic compound scenes (e.g. sichuan_basin_city, zhengzhou_
    # temperate_mid) use conditions that are equally true in distant cities.
    # Without competing scenes, East China photos lock to Sichuan, Beijing
    # photos lock to Zhengzhou/Taiyuan — producing catastrophic errors.
    #
    # Solution: mirror scenes with identical conditions but different bboxes.
    # When both match → COMPOUND-DISPERSE → neither locks → grid search decides.
    # When only one matches → correctly restricts to that region.
    #
    # Key design rule: any compound scene with ≥3 generic conditions that
    # span multiple distant cities needs a geographic competitor.
    "east_china_subtropical_city": {
        "conditions": {
            "urbanization": "medium_city",
            "climate_zone": "subtropical",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(30.0, 32.5, 118.0, 122.0, "Shanghai/Nanjing/Hangzhou delta"),
            BBox(28.0, 30.5, 112.0, 118.0, "Changsha/Nanchang/Wuhan"),
        ],
    },
    "east_china_overcast": {
        "conditions": {
            "climate_zone": "subtropical",
            "sky_quality": "overcast_grey",
            "urbanization": "medium_city",
        },
        "bboxes": [
            BBox(30.0, 32.5, 118.0, 122.0, "East China overcast urban"),
            BBox(28.0, 30.5, 112.0, 118.0, "Mid-Yangtze overcast urban"),
        ],
    },

    # ── North China Plain cities — split by geography ──────────────────
    # Formerly one huge scene (600km span) caused DISPERSE where Beijing
    # and Zhengzhou clusters were indistinguishable. Now split into 3
    # sub-regions (~200km each) so DBSCAN forms distinct clusters.
    # Each also has a tree_species variant for city-level discrimination
    # when VLM reliably extracts tree species.
    "beijing_heb_temperate_medium": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(37.5, 42.0, 114.0, 118.0, "Beijing/Tianjin/Hebei"),
        ],
    },
    "beijing_plane_gingko_temperate": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
            "tree_species": "plane_tree",
        },
        "bboxes": [
            BBox(39.7, 40.2, 116.1, 116.7, "Beijing plane-tree temperate"),
        ],
    },
    "shandong_temperate_medium": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(35.0, 38.0, 116.0, 122.5, "Shandong temperate cities"),
        ],
    },
    "liaoning_temperate_medium": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(41.0, 43.0, 122.0, 126.0, "Liaoning/Jilin temperate cities"),
        ],
    },
    "north_china_plain_temperate_deciduous": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_deciduous",
        },
        "bboxes": [
            BBox(37.5, 42.0, 114.0, 118.0, "Beijing/Tianjin/Hebei deciduous"),
            BBox(35.0, 38.0, 116.0, 122.5, "Shandong deciduous"),
            BBox(41.0, 43.0, 122.0, 126.0, "Liaoning/Jilin deciduous"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # CITY FINGERPRINTS — building_height + pavement + climate + architecture
    # ═══════════════════════════════════════════════════════════════════════
    # ── North China Plain mirror competition — Shanxi & Guanzhong ──────────
    # beijing_heb / shandong / liaoning all share identical conditions
    # (mid_rise + temperate + medium_city + urban_flat). Without mirrors,
    # Taiyuan/Xi'an photos also match only those 3 eastern scenes → bias east.
    # These mirrors create healthy competition: all 5 scenes match together
    # → DISPERSE → multi-hypothesis resolves via sensor/elevation/other elements.
    "shanxi_temperate_midrise": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(37.0, 40.5, 112.0, 114.5, "Shanxi basin: Taiyuan/Datong/Yangquan"),
        ],
    },
    "guanzhong_temperate_midrise": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(34.0, 35.5, 107.5, 110.5, "Guanzhong plain: Xi'an/Xianyang/Baoji"),
        ],
    },

    # Each fingerprint uses the most reliable VLM elements to distinguish
    # Chinese cities that would otherwise share identical broad features.
    # ═══════════════════════════════════════════════════════════════════════

    # ── Tier-1 / New Tier-1 cities ──────────────────────────────────────
    "chengdu_red_brick_mid": {
        "conditions": {
            "pavement_type": "red_brick_tiles",
            "building_height": "mid_rise",
            "urbanization": "metropolis",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(30.3, 31.0, 103.8, 104.3, "Chengdu core"),
            BBox(29.3, 29.8, 106.2, 106.8, "Chongqing core"),
        ],
    },
    "shanghai_grey_high": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "high_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(31.0, 31.5, 121.1, 121.7, "Shanghai grey-pavement high-rise"),
        ],
    },
    "beijing_grey_temperate": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(39.7, 40.2, 116.1, 116.7, "Beijing grey mid-rise temperate"),
        ],
    },
    "nanjing_planetree_grey": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "high_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "tree_species": "plane_tree",
        },
        "bboxes": [
            BBox(31.9, 32.2, 118.6, 118.9, "Nanjing downtown plane-tree + grey + high-rise"),
        ],
    },
    "wuhan_river_grey": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "high_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(30.4, 30.8, 114.1, 114.5, "Wuhan high-rise river city"),
        ],
    },
    "chongqing_red_brick_mountain": {
        "conditions": {
            "pavement_type": "red_brick_tiles",
            "building_height": "high_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(29.3, 29.8, 106.2, 106.8, "Chongqing red-brick + high-rise mountain city"),
        ],
    },
    "hangzhou_mid_grey_water": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(30.1, 30.4, 120.0, 120.4, "Hangzhou mid-rise grey"),
        ],
    },
    "tianjin_grey_temperate_high": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "high_rise",
            "climate_zone": "temperate",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(39.0, 39.3, 117.1, 117.8, "Tianjin high-rise temperate coastal"),
        ],
    },
    "suzhou_hui_grey_water": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "architecture_style": "hui_style",
        },
        "bboxes": [
            BBox(31.1, 31.4, 120.4, 120.8, "Suzhou hui-style + grey + water town"),
        ],
    },
    "xian_grey_temperate_mid": {
        "conditions": {
            "pavement_type": "grey_concrete",
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(34.1, 34.4, 108.7, 109.1, "Xi'an temperate mid-rise"),
        ],
    },
    "changsha_red_mid_river": {
        "conditions": {
            "pavement_type": "red_brick_tiles",
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "water_visible": True,
        },
        "bboxes": [
            BBox(28.0, 28.4, 112.8, 113.2, "Changsha red-brick mid-rise"),
        ],
    },

    # ── Tier-2 provincial capitals ──────────────────────────────────────
    "kunming_mid_highland": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(24.8, 25.2, 102.5, 103.0, "Kunming mid-rise highland"),
        ],
    },
    "fuzhou_mid_coastal": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
            "tree_species": "banyan",
        },
        "bboxes": [
            BBox(25.9, 26.2, 119.1, 119.5, "Fuzhou banyan subtropical coastal"),
        ],
    },
    "guiyang_red_mid_hills": {
        "conditions": {
            "pavement_type": "red_brick_tiles",
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(26.4, 26.8, 106.5, 106.9, "Guiyang red-brick hills"),
        ],
    },
    "nanning_subtropical_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(22.6, 23.0, 108.1, 108.5, "Nanning subtropical mid-rise"),
        ],
    },
    "zhengzhou_temperate_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
            "tree_species": "poplar",
        },
        "bboxes": [
            BBox(34.5, 34.9, 113.4, 113.9, "Zhengzhou temperate mid-rise + poplar"),
        ],
    },
    "hefei_subtropical_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(31.7, 32.0, 117.1, 117.4, "Hefei subtropical mid-rise"),
        ],
    },
    "nanchang_subtropical_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(28.5, 28.9, 115.7, 116.1, "Nanchang subtropical mid-rise"),
        ],
    },

    # ── NE China cities (temperate + medium_city) ───────────────────────
    "shenyang_temperate_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "vegetation_zone": "mixed_forest",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(41.6, 42.0, 123.2, 123.6, "Shenyang temperate mid-rise"),
        ],
    },
    "changchun_temperate_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "soil_color": "black",
        },
        "bboxes": [
            BBox(43.7, 44.1, 125.1, 125.5, "Changchun temperate black-soil"),
        ],
    },
    "dalian_coastal_temperate": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "water_visible": True,
        },
        "bboxes": [
            BBox(38.8, 39.1, 121.4, 121.8, "Dalian coastal temperate"),
        ],
    },
    "qingdao_coastal_temperate": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(35.9, 36.3, 120.2, 120.5, "Qingdao coastal hills temperate"),
        ],
    },

    # ── NW / Arid region cities ─────────────────────────────────────────
    "lanzhou_arid_valley": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "arid",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
            "vegetation_zone": "desert_scrub",
            "pavement_type": "grey_concrete",
        },
        "bboxes": [
            BBox(36.0, 36.2, 103.6, 104.0, "Lanzhou arid valley city"),
        ],
    },
    "urumqi_arid_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "arid",
            "urbanization": "medium_city",
            "language_script": "uyghur_arabic",
            "pavement_type": "grey_concrete",
            "vegetation_zone": "desert_scrub",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(43.6, 44.0, 87.4, 87.8, "Urumqi arid mid-rise"),
        ],
    },
    "hohhot_arid_steppe": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "arid",
            "urbanization": "medium_city",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(40.7, 41.0, 111.5, 111.9, "Hohhot arid steppe city"),
        ],
    },
    "taiyuan_temperate_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(37.7, 38.1, 112.3, 112.7, "Taiyuan temperate mid-rise"),
        ],
    },
    "shijiazhuang_temperate_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_deciduous",
        },
        "bboxes": [
            BBox(37.9, 38.2, 114.3, 114.7, "Shijiazhuang temperate mid-rise"),
        ],
    },
    "yinchuan_arid_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "arid",
            "urbanization": "medium_city",
            "pavement_type": "grey_concrete",
            "vegetation_zone": "desert_scrub",
        },
        "bboxes": [
            BBox(38.3, 38.7, 106.1, 106.4, "Yinchuan arid mid-rise"),
        ],
    },
    "xining_arid_highland": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "arid",
            "urbanization": "medium_city",
            "terrain_type": "plateau",
        },
        "bboxes": [
            BBox(36.5, 36.8, 101.6, 101.9, "Xining arid plateau city"),
        ],
    },

    # ── Tropical / far south ────────────────────────────────────────────
    "haikou_tropical_coastal": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "tropical",
            "urbanization": "medium_city",
            "vegetation_zone": "tropical_rainforest",
        },
        "bboxes": [
            BBox(19.9, 20.2, 110.1, 110.5, "Haikou tropical coastal"),
        ],
    },

    # ── S China coastal belt (subtropical + coastal + metropolis/med_city) ─
    "xiamen_coastal_mid": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(24.3, 24.6, 117.9, 118.3, "Xiamen coastal hills"),
        ],
    },

    # ── Architecture-anchored city fingerprints ─────────────────────────
    "guangdong_arcade_banyan_subtropical": {
        "conditions": {
            "architecture_style": "arcade",
            "tree_species": "banyan",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(22.5, 23.8, 112.8, 114.5, "Guangzhou / Foshan arcade + banyan"),
            BBox(21.5, 22.5, 108.0, 109.5, "Guangxi coast arcade"),
        ],
    },
    "xiamen_arcade_coastal": {
        "conditions": {
            "architecture_style": "arcade",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(24.3, 26.0, 117.5, 120.0, "S Fujian arcade cities (Xiamen/Quanzhou/Zhangzhou)"),
        ],
    },
    "harbin_russian_ne": {
        "conditions": {
            "climate_zone": "temperate",
            "soil_color": "black",
            "urbanization": "medium_city",
        },
        "bboxes": [
            BBox(45.5, 46.0, 126.4, 127.0, "Harbin black soil temperate"),
            BBox(43.5, 44.5, 124.5, 126.0, "Changchun / Jilin black soil"),
        ],
    },

    # ── Tree-species-anchored city fingerprints ─────────────────────────
    "beijing_gingko_courtyard": {
        "conditions": {
            "tree_species": "gingko",
            "climate_zone": "temperate",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(39.7, 40.2, 116.1, 116.7, "Beijing gingko + temperate metro"),
        ],
    },
    "south_china_camphor_metro": {
        "conditions": {
            "tree_species": "camphor",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(28.0, 32.5, 103.0, 122.0, "S China camphor metro cities"),
        ],
    },
    "ne_china_poplar_black_soil": {
        "conditions": {
            "tree_species": "poplar",
            "soil_color": "black",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(41.0, 48.5, 121.0, 135.5, "NE China poplar + black soil"),
        ],
    },
    "north_china_poplar_temperate": {
        "conditions": {
            "tree_species": "poplar",
            "climate_zone": "temperate",
            "urbanization": "medium_city",
        },
        "bboxes": [
            BBox(33.0, 42.0, 105.0, 122.5, "N China poplar temperate cities"),
        ],
    },
    "hangzhou_willow_water": {
        "conditions": {
            "tree_species": "willow",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "water_visible": True,
        },
        "bboxes": [
            BBox(30.1, 30.4, 120.0, 120.4, "Hangzhou willow + water (West Lake)"),
            BBox(31.1, 31.4, 120.4, 120.8, "Suzhou willow canals"),
        ],
    },
    "south_china_banyan_coastal": {
        "conditions": {
            "tree_species": "banyan",
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(22.5, 26.5, 109.0, 120.0, "S China banyan coastal hills"),
        ],
    },
    "nw_china_poplar_arid": {
        "conditions": {
            "tree_species": "poplar",
            "climate_zone": "arid",
        },
        "bboxes": [
            BBox(36.0, 44.5, 87.0, 107.0, "NW China poplar arid cities"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FAMOUS HIKING ROUTES — distinctive visual features + tight bboxes
    # ═══════════════════════════════════════════════════════════════════════
    # Each route has a "full" variant (high precision, strict conditions)
    # and a "low" variant (high recall, 2-3 conditions) so VLM doesn't
    # need to output all 6 fields perfectly to trigger a match.
    # ═══════════════════════════════════════════════════════════════════════

    # ── 鳌太线 (Aotai Line): Qinling granite boulder field above treeline ──
    "aotai_boulder_field": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(33.8, 34.3, 107.4, 107.9, "鳌太线 Taibai Shan boulder field (elev 3000-3767m)"),
        ],
    },
    "aotai_boulder_field_low": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "terrain_type": "sharp_mountains",
            "climate_zone": "temperate",
        },
        "bboxes": [
            BBox(33.5, 35.5, 106.5, 109.0, "秦岭太白山花岗岩山区 (鳌太线/华山区域)"),
        ],
    },

    # ── 洛克线 (Rock Line): Yading Three Holy Mountains + alpine lakes ──
    "yading_rock_line": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "language_script": "tibetan",
            "architecture_style": "tibetan_stone",
        },
        "bboxes": [
            BBox(28.2, 28.6, 100.2, 100.5, "洛克线 Yading Three Holy Mountains (Sichuan)"),
        ],
    },
    "yading_rock_line_low": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(27.5, 30.0, 99.5, 101.5, "稻城亚丁/洛克线 高山湖区 (川西)"),
        ],
    },

    # ── 狼塔线 (Langta Line): Central Tianshan glacial valley traverse ──
    "langta_tianshan_valley": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(43.0, 43.8, 85.5, 87.5, "狼塔线 Central Tianshan glacial valley (Xinjiang)"),
        ],
    },
    "langta_valley_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.0, 44.5, 80.0, 89.0, "天山冰川峡谷区域 (狼塔线/博格达)"),
        ],
    },

    # ── 武功山 (Wugongshan): Subtropical alpine meadow ridgeline ──
    # Unique in China: golden grassland above treeline at only 1500-1900m
    # in subtropical E China. No snow, no granite pillars — just endless
    # rolling grass ridges. Boardwalk trails visible in many photos.
    "wugongshan_alpine_meadow": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "climate_zone": "subtropical",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(27.3, 27.7, 113.8, 114.3, "武功山 subtropical alpine meadow (江西)"),
        ],
    },

    # ── 雨崩 (Yubeng): Meili Snow Mountain pilgrimage trek ──
    # Tibetan village nestled below 6740m Kawagarbo, prayer flags, rhododendron
    # forests, dramatic elevation change from 3000m village to 4000m passes.
    "yubeng_meili_trek": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "language_script": "tibetan",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(28.3, 28.6, 98.6, 99.0, "雨崩/梅里雪山徒步 (Yubeng / Meili Snow Mtn)"),
        ],
    },
    "yubeng_meili_trek_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(27.0, 30.0, 97.5, 101.5, "藏区雪山区域 (梅里/贡嘎/雅拉)"),
        ],
    },

    # ── 扎尕那 (Zhagana): Tibetan limestone spires + villages ──
    # Dramatic sharp limestone pinnacles rising above Tibetan villages
    # at 3000-4000m in Gansu/Sichuan border. Unlike Guilin (subtropical,
    # low-elevation karst towers), Zhagana is alpine karst with Tibetan culture.
    "zhagana_limestone_tibetan": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "alpine",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(33.8, 34.5, 102.5, 103.5, "扎尕那 alpine karst + Tibetan (Gansu/Sichuan)"),
        ],
    },

    # ── 虎跳峡 (Tiger Leaping Gorge): Deep canyon between Jade Dragon & Haba ──
    # World's deepest river gorge: 2000m+ cliffs on both sides, Jinsha River
    # below, Jade Dragon Snow Mtn (5596m) above. Distinct from Three Gorges
    # (subtropical, lower relief) — this is alpine canyon with snow peaks visible.
    "tiger_leaping_gorge_trek": {
        "conditions": {
            "landform_detail": "river_canyon",
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(27.0, 27.5, 100.0, 100.5, "虎跳峡 Tiger Leaping Gorge (Yunnan)"),
        ],
    },

    # ── 喀拉峻 (Kalajun): Tianshan alpine meadow plateau ──
    # Vast rolling alpine meadows at 2000-3000m with Tianshan snow peaks
    # on the horizon. Kazakh yurts in summer. Unlike Inner Mongolia grassland
    # (temperate, lower elevation, no snow peaks).
    "kalajun_tianshan_meadow": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "climate_zone": "alpine",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(42.5, 43.5, 81.0, 83.0, "喀拉峻 Tianshan alpine meadow (Xinjiang)"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # P0-1: Tropical <25N — 8 scenes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Hainan inland monsoon forest ────────────────────────────────────────
    "hainan_inland_monsoon_forest": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_monsoon_forest",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(18.5, 19.5, 109.0, 110.5, "Hainan inland monsoon forest (五指山区域)"),
        ],
    },

    # ── Hainan coastal urban ────────────────────────────────────────────────
    "hainan_coastal_urban": {
        "conditions": {
            "climate_zone": "tropical",
            "tree_species": "coconut",
            "urbanization": "city",
        },
        "bboxes": [
            BBox(18.2, 20.0, 109.0, 111.0, "Hainan coastal urban (海口/三亚/琼海)"),
        ],
    },

    # ── Xishuangbanna rubber plantation ─────────────────────────────────────
    "xishuangbanna_rubber_plantation": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_monsoon_forest",
            "terrain_type": "rolling_hills",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(21.5, 22.5, 100.0, 101.5, "Xishuangbanna rubber plantation + tropical hills"),
        ],
    },

    # ── Dehong tropical border ──────────────────────────────────────────────
    "dehong_tropical_border": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_monsoon_forest",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(23.8, 25.0, 97.5, 99.0, "Dehong tropical border (德宏/缅甸边界)"),
        ],
    },

    # ── Guangxi tropical karst ──────────────────────────────────────────────
    "guangxi_tropical_karst": {
        "conditions": {
            "climate_zone": "tropical",
            "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(21.5, 23.0, 106.5, 108.5, "Guangxi tropical karst (桂西南)"),
        ],
    },

    # ── Pearl River Delta clear-sky subtropical ───────────────────────────
    # PRD skies tend toward clear_blue (coastal maritime air). Sichuan Basin
    # is famous for persistent overcast. When VLM sees clear skies with
    # subtropical urban elements, PRD is much more likely than Sichuan.
    "prd_clear_sky_subtropical": {
        "conditions": {
            "sky_quality": "clear_blue",
            "climate_zone": "subtropical",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(22.0, 24.0, 113.0, 115.0, "PRD clear-sky subtropical (Shenzhen/Guangzhou)"),
        ],
    },
    # ── Pearl River Delta scooter subtropical ─────────────────────────────
    # Electric scooters are ubiquitous in PRD cities (shared e-scooter fleets,
    # narrow streets favor 2-wheelers). Sichuan cities have far fewer.
    "prd_scooter_subtropical": {
        "conditions": {
            "infrastructure_tags": "electric_scooters",
            "climate_zone": "subtropical",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(22.0, 23.8, 113.0, 114.5, "PRD scooter subtropical"),
        ],
    },

    # ── Pearl River Delta tropical fringe ───────────────────────────────────
    "pearl_river_delta_tropical_fringe": {
        "conditions": {
            "climate_zone": "tropical",
            "urbanization": "metropolis",
            "vegetation_zone": "broadleaf_evergreen",
            "tree_species": "banyan",
            "water_visible": "yes",
        },
        "bboxes": [
            BBox(22.0, 23.5, 113.0, 114.5, "Pearl River Delta tropical fringe (珠三角)"),
        ],
    },

    # ── Leizhou Peninsula tropical ──────────────────────────────────────────
    "leizhou_peninsula_tropical": {
        "conditions": {
            "climate_zone": "tropical",
            "terrain_type": "urban_flat",
            "vegetation_zone": "tropical_monsoon_forest",
        },
        "bboxes": [
            BBox(20.2, 21.5, 109.5, 110.5, "Leizhou Peninsula tropical (雷州半岛)"),
        ],
    },

    # ── S China tropical hills generic (competition mirror) ─────────────────
    "s_china_tropical_hills_generic": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "broadleaf_evergreen",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(18.0, 24.0, 97.0, 111.0, "S China tropical hills generic"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # P0-2: Mountain >1000m — 14 scenes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Qinling mid-elevation mixed ─────────────────────────────────────────
    "qinling_mid_elevation_mixed": {
        "conditions": {
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
            "mountain_rock_type": "granite_spheroidal",
        },
        "bboxes": [
            BBox(33.0, 34.5, 106.0, 110.0, "Qinling mid-elevation mixed forest (1500-2800m)"),
        ],
    },

    # ── Qinling treeline conifer ────────────────────────────────────────────
    "qinling_treeline_conifer": {
        "conditions": {
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "landform_detail": "above_treeline",
        },
        "bboxes": [
            BBox(33.5, 34.3, 107.0, 108.5, "Qinling treeline+ conifer (2500-3767m)"),
        ],
    },

    # ── Dabieshan mid-mountain ──────────────────────────────────────────────
    "dabieshan_mid_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
            "mountain_rock_type": "granite_dome",
        },
        "bboxes": [
            BBox(30.5, 32.0, 115.0, 117.5, "Dabieshan mid-mountain (500-1777m)"),
        ],
    },

    # ── Wuyishan tea terraces ──────────────────────────────────────────────
    "wuyishan_tea_terraces": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
            "mountain_rock_type": "red_sandstone_danxia",
        },
        "bboxes": [
            BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan tea terraces (武夷山岩茶)"),
        ],
    },

    # ── Wuyishan low (low-threshold variant) ───────────────────────────────
    "wuyishan_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(27.0, 29.0, 117.0, 119.0, "Wuyishan mountain area (broad)"),
        ],
    },

    # ── Nanling subtropical mountain ────────────────────────────────────────
    "nanling_subtropical_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "red",
        },
        "bboxes": [
            BBox(24.5, 26.5, 111.0, 114.5, "Nanling subtropical mountain (南岭)"),
        ],
    },

    # ── Luoxiao Jinggangshan ────────────────────────────────────────────────
    "luoxiao_jinggangshan": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
            "soil_color": "red",
            "mountain_rock_type": "granite_dome",
        },
        "bboxes": [
            BBox(26.0, 27.5, 113.5, 115.5, "Luoxiao Mt / Jinggangshan (罗霄山/井冈山)"),
        ],
    },

    # ── Daloushan limestone ─────────────────────────────────────────────────
    "daloushan_limestone": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen",
            "mountain_rock_type": "limestone_karst",
        },
        "bboxes": [
            BBox(27.5, 29.5, 106.0, 108.5, "Daloushan limestone (大娄山/遵义)"),
        ],
    },

    # ── Wumengshan highland ─────────────────────────────────────────────────
    "wumengshan_highland": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "soil_color": "red",
        },
        "bboxes": [
            BBox(25.5, 28.0, 103.0, 106.0, "Wumengshan highland (乌蒙山 2000-4000m)"),
        ],
    },

    # ── Qilian arid mountain ────────────────────────────────────────────────
    "qilian_arid_mountain": {
        "conditions": {
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "grassland_steppe",
            "landform_detail": "above_treeline",
        },
        "bboxes": [
            BBox(37.0, 40.0, 96.0, 103.0, "Qilian arid mountain (祁连山 >3500m)"),
        ],
    },

    # ── Kunlun alpine desert ────────────────────────────────────────────────
    "kunlun_alpine_desert": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "desert",
            "landform_detail": "above_treeline",
        },
        "bboxes": [
            BBox(35.0, 39.0, 76.0, 90.0, "Kunlun alpine desert (昆仑山)"),
        ],
    },

    # ── Altai boreal larch ──────────────────────────────────────────────────
    "altai_boreal_larch": {
        "conditions": {
            "climate_zone": "boreal",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "language_script": "mongolian_cyrillic",
        },
        "bboxes": [
            BBox(47.5, 49.5, 86.0, 90.0, "Altai boreal larch forest (阿尔泰泰加林)"),
        ],
    },

    # ── Yulong + Haba conifer snow ──────────────────────────────────────────
    "yulong_haba_conifer_snow": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "mountain_rock_type": "snow_peaks_glaciers",
        },
        "bboxes": [
            BBox(26.8, 27.5, 100.0, 100.5, "Yulong+Haba snow mt conifer (玉龙/哈巴雪山)"),
        ],
    },

    # ── SE China mountain generic (competition mirror) ──────────────────────
    "se_china_mountain_generic": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(22.0, 29.0, 110.0, 120.0, "SE China mountain generic"),
            BBox(29.0, 32.0, 115.0, 122.0, "E China mountain generic"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # P0-3: Subtropical 25-30N — 11 scenes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Yunnan plateau conifer + red soil ───────────────────────────────────
    "yunnan_plateau_conifer_red": {
        "conditions": {
            "climate_zone": "subtropical",
            "vegetation_zone": "conifer_forest",
            "soil_color": "red",
            "terrain_type": "plateau",
        },
        "bboxes": [
            BBox(24.0, 27.0, 100.0, 104.0, "Yunnan plateau conifer + red soil (滇中高原)"),
        ],
    },

    # ── Yunnan karst plateau ────────────────────────────────────────────────
    "yunnan_karst_plateau": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "karst_peaks",
            "soil_color": "red",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(23.5, 26.5, 102.5, 106.0, "Yunnan karst plateau (滇东喀斯特高原)"),
        ],
    },

    # ── Hunan red soil hills ────────────────────────────────────────────────
    "hunan_red_soil_hills": {
        "conditions": {
            "climate_zone": "subtropical",
            "soil_color": "red",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(27.0, 30.0, 110.0, 114.0, "Hunan red soil hills (湘中红壤丘陵)"),
        ],
    },

    # ── Hunan red soil low (low-threshold variant) ──────────────────────────
    "hunan_red_soil_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "soil_color": "red",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(25.0, 30.0, 109.0, 115.0, "Hunan/Jiangxi red soil zone"),
        ],
    },

    # ── Jiangxi mixed forest hills ──────────────────────────────────────────
    "jiangxi_mixed_forest_hills": {
        "conditions": {
            "climate_zone": "subtropical",
            "vegetation_zone": "mixed_forest",
            "soil_color": "red",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(27.0, 30.0, 115.0, 118.5, "Jiangxi mixed forest red hills"),
        ],
    },

    # ── Fujian coastal tea hills ────────────────────────────────────────────
    "fujian_coastal_tea_hills": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen",
            "tree_species": "banyan",
        },
        "bboxes": [
            BBox(24.5, 27.0, 117.0, 120.5, "Fujian coastal banyan + tea hills"),
        ],
    },

    # ── Zhejiang bamboo subtropical ─────────────────────────────────────────
    "zhejiang_bamboo_subtropical": {
        "conditions": {
            "climate_zone": "subtropical",
            "vegetation_zone": "broadleaf_evergreen",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(29.0, 31.0, 119.0, 122.0, "Zhejiang bamboo + evergreen hills"),
        ],
    },

    # ── Hubei yellow-brown hills ────────────────────────────────────────────
    "hubei_yellow_brown_hills": {
        "conditions": {
            "climate_zone": "subtropical",
            "soil_color": "yellow_brown",
            "vegetation_zone": "mixed_forest",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(30.0, 32.5, 110.0, 114.5, "Hubei yellow-brown soil hills"),
        ],
    },

    # ── Guangdong karst subtropical ─────────────────────────────────────────
    "guangdong_karst_subtropical": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(23.5, 25.5, 111.0, 114.5, "N Guangdong karst subtropical"),
        ],
    },

    # ── Subtropical red soil generic (competition mirror) ───────────────────
    "subtropical_red_soil_generic": {
        "conditions": {
            "climate_zone": "subtropical",
            "soil_color": "red",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(24.0, 30.0, 102.0, 120.0, "S China red soil hills (broad)"),
        ],
    },

    # ── Subtropical yellow-brown hills (competition mirror) ─────────────────
    "subtropical_yellow_brown_hills": {
        "conditions": {
            "climate_zone": "subtropical",
            "soil_color": "yellow_brown",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(28.0, 33.0, 106.0, 116.0, "Mid-China yellow-brown hills"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # P2: City Fingerprints — 19 scenes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Chengdu camphor + flat basin core ───────────────────────────────────
    "chengdu_camphor_flat": {
        "conditions": {
            "tree_species": "camphor",
            "pavement_type": "red_brick_tiles",
            "terrain_type": "urban_flat",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(30.4, 30.8, 103.9, 104.3, "Chengdu camphor + flat basin core"),
        ],
    },

    # ── Chengdu flat low ────────────────────────────────────────────────────
    "chengdu_flat_low": {
        "conditions": {
            "terrain_type": "urban_flat",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "pavement_type": "red_brick_tiles",
        },
        "bboxes": [
            BBox(30.3, 31.0, 103.5, 104.5, "Chengdu plain wider area"),
        ],
    },

    # ── Chongqing mountain-river ────────────────────────────────────────────
    "chongqing_mountain_river": {
        "conditions": {
            "terrain_type": "sharp_mountains",
            "building_height": "high_rise",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "water_visible": "yes",
        },
        "bboxes": [
            BBox(29.3, 29.7, 106.3, 106.8, "Chongqing mountain+river+Yangtze"),
        ],
    },

    # ── Chongqing rolling low ───────────────────────────────────────────────
    "chongqing_rolling_low": {
        "conditions": {
            "terrain_type": "rolling_hills",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(29.0, 30.5, 105.5, 107.5, "Chongqing wider area (rolling terrain)"),
        ],
    },

    # ── Sichuan basin twin cities (competition mirror for Chengdu/Chongqing)
    "sichuan_basin_twin_cities": {
        "conditions": {
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "pavement_type": "red_brick_tiles",
        },
        "bboxes": [
            BBox(29.0, 31.0, 103.5, 107.0, "Sichuan Basin twin cities (Chengdu/Chongqing rivalry)"),
        ],
    },

    # ── Beijing wide street temperate ───────────────────────────────────────
    "beijing_wide_street_temperate": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "metropolis",
            "building_height": "mid_rise",
            "tree_species": "gingko",
        },
        "bboxes": [
            BBox(39.7, 40.2, 116.0, 116.8, "Beijing wide street + gingko"),
        ],
    },

    # ── Shenyang heavy industry ─────────────────────────────────────────────
    "shenyang_heavy_industry": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "metropolis",
            "vegetation_zone": "broadleaf_deciduous",
            "building_height": "mid_rise",
            "soil_color": "black",
        },
        "bboxes": [
            BBox(41.5, 42.2, 123.0, 124.0, "Shenyang heavy industry + black soil"),
        ],
    },

    # ── Changchun black soil wide ───────────────────────────────────────────
    "changchun_black_soil_wide": {
        "conditions": {
            "climate_zone": "temperate",
            "soil_color": "black",
            "urbanization": "metropolis",
            "building_height": "mid_rise",
        },
        "bboxes": [
            BBox(43.5, 44.2, 125.0, 125.8, "Changchun black soil + wide streets"),
        ],
    },

    # ── Changsha red river camphor ──────────────────────────────────────────
    "changsha_red_river_camphor": {
        "conditions": {
            "pavement_type": "red_brick_tiles",
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "tree_species": "camphor",
            "water_visible": "yes",
        },
        "bboxes": [
            BBox(28.0, 28.4, 112.8, 113.2, "Changsha Xiang River + red brick + camphor"),
        ],
    },

    # ── Nanchang lake plain ─────────────────────────────────────────────────
    "nanchang_lake_plain": {
        "conditions": {
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "terrain_type": "urban_flat",
            "water_visible": "yes",
        },
        "bboxes": [
            BBox(28.4, 28.9, 115.6, 116.2, "Nanchang Poyang Lake plain"),
        ],
    },

    # ── Wuhan Yangtze + Han River ───────────────────────────────────────────
    "wuhan_yangtze_han_river": {
        "conditions": {
            "climate_zone": "subtropical",
            "urbanization": "metropolis",
            "building_height": "high_rise",
            "water_visible": "yes",
            "pavement_type": "grey_concrete",
        },
        "bboxes": [
            BBox(30.3, 30.8, 114.0, 114.6, "Wuhan Yangtze-Han River confluence"),
        ],
    },

    # ── Xi'an ancient grey temperate ────────────────────────────────────────
    "xian_ancient_grey_temperate": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "metropolis",
            "pavement_type": "grey_concrete",
            "architecture_style": "ancient_chinese",
        },
        "bboxes": [
            BBox(34.0, 34.5, 108.7, 109.2, "Xi'an ancient city wall + grey"),
        ],
    },

    # ── Zhengzhou Yellow River plain ────────────────────────────────────────
    "zhengzhou_yellow_river_plain": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "urban_flat",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(34.5, 35.0, 113.3, 114.0, "Zhengzhou Yellow River alluvial plain"),
        ],
    },

    # ── Taiyuan loess basin ─────────────────────────────────────────────────
    "taiyuan_loess_basin": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "plateau",
            "soil_color": "yellow_brown",
            "vegetation_zone": "grassland_steppe",
        },
        "bboxes": [
            BBox(37.5, 38.2, 112.2, 113.0, "Taiyuan loess basin (太原盆地)"),
        ],
    },

    # ── Shijiazhuang deciduous flat ─────────────────────────────────────────
    "shijiazhuang_deciduous_flat": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_deciduous",
            "pavement_type": "grey_concrete",
        },
        "bboxes": [
            BBox(37.8, 38.3, 114.2, 114.8, "Shijiazhuang N China Plain deciduous"),
        ],
    },

    # ── North China temperate midrise generic (competition mirror) ──────────
    "north_china_temperate_midrise_generic": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(34.0, 42.0, 112.0, 120.0, "N China temperate midrise generic"),
            BBox(34.0, 42.0, 120.0, 126.0, "NE China temperate midrise generic"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Gap Fill: Low-threshold variants for existing hiking/mountain scenes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 虎跳峡低门槛变体 ──────────────────────────────────────────────────
    "tiger_leaping_gorge_trek_low": {
        "conditions": {
            "landform_detail": "river_canyon",
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(26.5, 28.5, 99.5, 101.0, "虎跳峡/金沙江峡谷区域"),
        ],
    },

    # ── 武功山低门槛变体 ──────────────────────────────────────────────────
    "wugongshan_alpine_meadow_low": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(27.0, 29.0, 113.0, 115.5, "武功山/华东高山草甸区域"),
        ],
    },

    # ── 贡嘎/四姑娘山低门槛变体 ───────────────────────────────────────────
    "gongga_siguniang_snow_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "terrain_type": "sharp_mountains",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(29.0, 31.5, 101.0, 103.5, "贡嘎/四姑娘山区域 (川西)"),
        ],
    },

    # ── 稻城亚丁低门槛变体 ────────────────────────────────────────────────
    "daocheng_yading_lakes_low": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(27.5, 30.5, 99.5, 101.0, "稻城亚丁高原湖区"),
        ],
    },

    # ── 丹霞山低门槛变体 ──────────────────────────────────────────────────
    "danxiashan_red_sandstone_low": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(24.0, 27.0, 112.0, 115.0, "粤北丹霞地貌区域"),
        ],
    },

    # ── 云南喀斯特高原低门槛变体 ──────────────────────────────────────────
    "yunnan_karst_highland_low": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(23.0, 27.0, 102.0, 106.5, "云南/贵州喀斯特高原"),
        ],
    },

    # ── 黄山/三清山花岗岩低门槛变体 ──────────────────────────────────────
    "huangshan_granite_low": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "subtropical",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(29.5, 31.5, 116.0, 119.5, "皖南/赣北花岗岩山区 (黄山/三清山)"),
        ],
    },

    # ── 三清山花岗岩低门槛变体 ──────────────────────────────────────────
    "sanqingshan_granite_low": {
        "conditions": {
            "mountain_rock_type": "granite_dome",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(28.0, 30.0, 117.0, 119.0, "赣东北花岗岩山区 (三清山)"),
        ],
    },

    # ── 张家界砂岩低门槛变体 ─────────────────────────────────────────────
    "zhangjiajie_sandstone_low": {
        "conditions": {
            "terrain_type": "sandstone_pillars",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(28.5, 30.0, 109.5, 111.5, "张家界/湘西石英砂岩区域"),
        ],
    },

    # ── 九寨沟/黄龙低门槛变体 ─────────────────────────────────────────────
    "jiuzhaigou_huanglong_low": {
        "conditions": {
            "mountain_rock_type": "alpine_lakes",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(32.0, 34.0, 103.0, 105.0, "九寨沟/黄龙高海拔湖区 (川北)"),
        ],
    },

    # ── 梅里/玉龙低门槛变体 ──────────────────────────────────────────────
    "meili_yulong_snow_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(27.0, 29.5, 98.0, 101.0, "滇西北雪山区域 (梅里/玉龙)"),
        ],
    },

    # ── 珠峰高海拔荒漠低门槛变体 ──────────────────────────────────────────
    "everest_high_desert_low": {
        "conditions": {
            "climate_zone": "alpine",
            "vegetation_zone": "alpine_meadow",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(27.5, 30.0, 85.0, 88.0, "喜马拉雅高海拔荒漠区域"),
        ],
    },

    # ── 秦岭林线低门槛变体 ───────────────────────────────────────────────
    "qinling_treeline_low": {
        "conditions": {
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(33.0, 35.0, 106.0, 109.5, "秦岭高海拔针叶林区域"),
        ],
    },

    # ── 乌蒙山低门槛变体 ──────────────────────────────────────────────────
    "wumengshan_highland_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(25.5, 28.5, 103.0, 106.5, "乌蒙山高地区域 (滇东北/黔西北)"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Gap Fill: High-risk & popular hiking routes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 墨脱徒步线 (Medog Trek): 派镇→墨脱, 中国最危险的徒步线 ──────────
    "medog_trek": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(29.0, 29.8, 94.5, 95.8, "墨脱徒步线 (派镇→墨脱, 雅鲁藏布大峡谷)"),
        ],
    },
    "medog_trek_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(28.5, 30.5, 94.0, 96.5, "藏东南峡谷徒步区域 (墨脱/察隅)"),
        ],
    },

    # ── 夏特古道 (Xiate Ancient Trail): 昭苏→阿克苏, 天山冰川峡谷 ──────
    "xiate_ancient_trail": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(42.0, 42.8, 80.5, 81.5, "夏特古道 (昭苏→阿克苏, 木扎尔特冰川)"),
        ],
    },
    "xiate_ancient_trail_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(41.5, 43.5, 80.0, 82.0, "西天山冰川峡谷区域 (夏特/木扎尔特)"),
        ],
    },

    # ── 乌孙古道 (Wusun Ancient Trail): 特克斯→库车, 天山南脉 ──────────
    "wusun_ancient_trail": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(42.5, 43.5, 82.0, 84.5, "乌孙古道 (特克斯→库车, 天山南脉)"),
        ],
    },
    "wusun_ancient_trail_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(42.0, 44.0, 81.0, 85.0, "中天山冰川区域 (乌孙/巴音布鲁克)"),
        ],
    },

    # ── 博格达大环线 (Bogda Circuit): 东天山, 博格达峰5445m ────────────
    "bogda_circuit": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(43.6, 44.2, 88.0, 89.0, "博格达大环线 (东天山博格达峰5445m)"),
        ],
    },
    "bogda_circuit_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(43.0, 45.0, 87.0, 90.0, "东天山博格达/天山天池区域"),
        ],
    },

    # ── 太白山南北穿越 (Taibai Traverse): 秦岭最高峰 ────────────────────
    "taibai_traverse": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(33.8, 34.2, 107.6, 107.9, "太白山南北穿越 (厚畛子→汤峪, 拔仙台3767m)"),
        ],
    },
    "taibai_traverse_low": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(33.5, 34.5, 107.0, 108.5, "秦岭太白山区域 (鳌太/南北穿越)"),
        ],
    },

    # ── 船底顶 (Chuandiding): "广东户外毕业线" ──────────────────────────
    "chuandiding_trek": {
        "conditions": {
            "mountain_rock_type": "granite_dome",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(24.3, 24.6, 113.1, 113.6, "船底顶 (粤北, 广东户外毕业线1586m)"),
        ],
    },
    "chuandiding_trek_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(24.0, 25.5, 112.5, 114.5, "粤北南岭山区 (船底顶/大东山)"),
        ],
    },

    # ── 韭菜岭 (Jiucailing): 湘桂边境, 十大非著名山峰之首 ──────────────
    "jiucailing_trek": {
        "conditions": {
            "mountain_rock_type": "granite_dome",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(25.3, 25.6, 111.0, 111.5, "韭菜岭 (湘桂边境都庞岭, 2009m)"),
        ],
    },
    "jiucailing_trek_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(25.0, 26.5, 110.0, 112.0, "湘桂边境都庞岭/海洋山脉区域"),
        ],
    },

    # ── 喀纳斯徒步 (Kanas Trek): 阿尔泰山, 中国最美徒步线之一 ──────────
    "kanas_trek": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "boreal",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(48.5, 49.2, 86.8, 87.8, "喀纳斯徒步 (喀纳斯湖→禾木→小黑湖, 阿尔泰山)"),
        ],
    },
    "kanas_trek_low": {
        "conditions": {
            "climate_zone": "boreal",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(47.5, 49.5, 86.0, 89.0, "阿尔泰山泰加林区域 (喀纳斯/禾木)"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Gap Fill: Eastern / SE coastal mountains
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 深圳沿海山脉 (梧桐山/七娘山/马峦山) ──────────────────────────────
    "shenzhen_coastal_mountain": {
        "conditions": {
            "mountain_rock_type": "granite_dome",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(22.4, 22.8, 113.8, 114.5, "深圳沿海山脉 (梧桐山943m/七娘山)"),
        ],
    },
    "shenzhen_coastal_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(22.0, 23.5, 113.5, 115.0, "珠三角沿海山脉区域"),
        ],
    },

    # ── 台湾阿里山山脉 (中低海拔亚热带) ─────────────────────────────────
    "taiwan_subtropical_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(23.0, 24.5, 120.5, 121.5, "台湾阿里山山脉 (中低海拔亚热带)"),
        ],
    },

    # ── 台湾玉山/雪山 (>3000m高山针叶林) ───────────────────────────────
    "taiwan_alpine_conifer": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(23.2, 24.5, 120.8, 121.8, "台湾玉山/雪山 (>3000m高山针叶林)"),
        ],
    },

    # ── 台湾山地通用低门槛 ────────────────────────────────────────────────
    "taiwan_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(22.5, 25.0, 120.5, 122.0, "台湾山地通用场景"),
        ],
    },

    # ── 海南五指山/鹦哥岭 热带山地雨林 ──────────────────────────────────
    "hainan_tropical_mountain": {
        "conditions": {
            "climate_zone": "tropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "tropical_monsoon_forest",
        },
        "bboxes": [
            BBox(18.7, 19.2, 109.3, 110.0, "海南五指山/鹦哥岭 (热带山地雨林)"),
        ],
    },
    "hainan_mountain_low": {
        "conditions": {
            "climate_zone": "tropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(18.5, 20.0, 109.0, 110.5, "海南岛山地通用场景"),
        ],
    },

    # ── 香港都市山脉 (太平山/狮子山) ────────────────────────────────────
    "hk_coastal_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "urbanization": "metropolis",
        },
        "bboxes": [
            BBox(22.2, 22.5, 114.1, 114.3, "香港沿海城市山脉 (太平山/狮子山)"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Gap Fill: Competition mirrors
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 西南喀斯特通用镜像 (广西/贵州/云南 karst 竞争) ──────────────────
    "southwest_karst_generic": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(22.0, 28.0, 102.0, 111.0, "西南喀斯特通用 (广西/贵州/云南)"),
        ],
    },

    # ── 华东花岗岩山区通用镜像 (黄山/三清山/天柱山 竞争) ──────────────
    "east_china_granite_mountain_generic": {
        "conditions": {
            "mountain_rock_type": "granite_dome",
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(28.0, 32.0, 115.0, 120.0, "华东花岗岩山区通用 (黄山/三清山/天柱山)"),
        ],
    },

    # ── 热带岛屿山地通用镜像 (海南/台湾 竞争) ──────────────────────────
    "tropical_island_mountain_generic": {
        "conditions": {
            "climate_zone": "tropical",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(18.0, 25.0, 109.0, 122.0, "中国热带海岛山地通用 (海南/台湾)"),
        ],
    },

    # ── 西部高山峡谷通用镜像 (天山/昆仑/横断 竞争) ────────────────────
    "western_alpine_valley_generic": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
            "landform_detail": "river_canyon",
        },
        "bboxes": [
            BBox(28.0, 44.0, 80.0, 100.0, "西部高山峡谷通用 (天山/昆仑/横断)"),
        ],
    },

    # ── 中部丘陵通用镜像 (湖北/湖南/江西/安徽 竞争) ──────────────────
    "central_china_rolling_hills_generic": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(28.0, 33.0, 110.0, 118.0, "中部丘陵通用 (湖北/湖南/江西/安徽)"),
        ],
    },

    # ── 热带季雨林通用镜像 (西双版纳/德宏/海南 竞争) ──────────────────
    "tropical_monsoon_forest_generic": {
        "conditions": {
            "climate_zone": "tropical",
            "vegetation_zone": "tropical_monsoon_forest",
        },
        "bboxes": [
            BBox(21.0, 25.0, 97.0, 111.0, "热带季雨林通用 (西双版纳/德宏/海南)"),
        ],
    },

    # ── 东南沿海丘陵通用镜像 (福建/浙江 竞争) ──────────────────────────
    "coastal_subtropical_hills_generic": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(23.0, 28.0, 117.0, 122.0, "东南沿海丘陵通用 (福建/浙江)"),
        ],
    },

    # ── 北方温带山地通用镜像 (秦岭/太行/长白山 竞争) ──────────────────
    "northern_temperate_mountain_generic": {
        "conditions": {
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(33.0, 42.0, 106.0, 120.0, "北方温带山地通用 (秦岭/太行/长白山)"),
        ],
    },


    # ═══════════════════════════════════════════════════════════════════════════
    # Final Gap Fill: Missing major mountain ranges + missing low variants
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 大巴山 (Daba Shan): 川陕鄂交界, 汉江与嘉陵江分水岭 ──────────────
    "dabashan_mid_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
            "mountain_rock_type": "limestone_karst",
        },
        "bboxes": [
            BBox(31.5, 33.0, 107.0, 110.5, "大巴山 (川陕鄂交界, 2000-3100m)"),
        ],
    },
    "dabashan_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(31.0, 33.5, 106.5, 111.0, "大巴山/米仓山区域 (川陕鄂交界)"),
        ],
    },

    # ── 太行山 (Taihang Shan): 华北平原西缘, 中国最重要的山脉之一 ──────
    "taihangshan_mountain": {
        "conditions": {
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_deciduous",
            "mountain_rock_type": "limestone_karst",
        },
        "bboxes": [
            BBox(35.0, 40.0, 112.0, 115.5, "太行山 (华北平原西缘, 1500-3000m)"),
        ],
    },
    "taihangshan_mountain_low": {
        "conditions": {
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_deciduous",
        },
        "bboxes": [
            BBox(34.0, 40.5, 111.0, 116.0, "太行山/吕梁山区域 (华北西缘山地)"),
        ],
    },

    # ── 雪峰山 (Xuefeng Shan): 湘西, 沅江与资水分水岭 ──────────────────
    "xuefengshan_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
            "soil_color": "red",
        },
        "bboxes": [
            BBox(26.5, 28.5, 109.5, 111.5, "雪峰山 (湘西, 沅江/资水分水岭, 1500-2000m)"),
        ],
    },
    "xuefengshan_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "soil_color": "red",
        },
        "bboxes": [
            BBox(26.0, 29.0, 109.0, 112.0, "湘西山地 (雪峰山/武陵山)"),
        ],
    },

    # ── 雁荡山 (Yandang Shan): 浙东南, 流纹岩地貌, 世界地质公园 ────────
    "yandangshan_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
            "mountain_rock_type": "granite_dome",
        },
        "bboxes": [
            BBox(28.0, 28.8, 120.5, 121.5, "雁荡山 (浙东南, 流纹岩地貌, 1100m)"),
        ],
    },
    "yandangshan_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(27.5, 29.0, 120.0, 122.0, "浙东南沿海山脉 (雁荡山/括苍山)"),
        ],
    },

    # ── 戴云山 (Daiyun Shan): 闽中, 福建南部最高峰 ────────────────────
    "daiyunshan_mountain": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
            "mountain_rock_type": "granite_dome",
        },
        "bboxes": [
            BBox(25.3, 26.2, 117.5, 119.0, "戴云山 (闽中, 福建南部最高峰1856m)"),
        ],
    },
    "daiyunshan_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(25.0, 27.0, 117.0, 119.5, "闽中闽南山地 (戴云山/博平岭)"),
        ],
    },

    # ── 阴山 (Yin Shan): Inner Mongolia, arid temperate mountain barrier ──
    "yinshan_arid_mountain": {
        "conditions": {
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "sparse",
        },
        "bboxes": [
            BBox(40.0, 42.5, 106.0, 114.0, "阴山 (Inner Mongolia arid mountain barrier)"),
        ],
    },
    "yinshan_mountain_low": {
        "conditions": {
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(39.5, 43.0, 105.0, 115.0, "阴山/狼山区域 (内蒙古高原南缘)"),
        ],
    },

    # ── 阿尔金山 (Altun Shan): Gansu/Qinghai/Xinjiang border, arid alpine ──
    "altun_arid_mountain": {
        "conditions": {
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "desert_scrub",
        },
        "bboxes": [
            BBox(37.5, 39.5, 86.0, 92.0, "Altun Shan (Gansu/Qinghai/Xinjiang border)"),
        ],
    },
    "altun_mountain_low": {
        "conditions": {
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(37.0, 40.0, 85.0, 93.0, "阿尔金山/东昆仑区域"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Missing low-threshold variants for existing mountain scenes
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 大别山低门槛 ──────────────────────────────────────────────────────
    "dabieshan_mid_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(30.0, 32.5, 114.5, 117.5, "大别山区域 (鄂豫皖交界)"),
        ],
    },

    # ── 天山雪峰冰川低门槛 ──────────────────────────────────────────────
    "tianshan_snow_alpine_low": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(41.0, 45.0, 80.0, 90.0, "天山雪峰冰川区域 (西天山/中天山/东天山)"),
        ],
    },

    # ── 秦岭太白山低门槛 ────────────────────────────────────────────────
    "qinling_taibai_alpine_granite_low": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(33.0, 35.0, 106.5, 109.0, "秦岭太白山/鳌山花岗岩区域"),
        ],
    },

    # ── 长白山火山低门槛 ────────────────────────────────────────────────
    "changbaishan_volcanic_low": {
        "conditions": {
            "climate_zone": "boreal",
            "vegetation_zone": "conifer_forest",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.0, 43.0, 127.0, 129.0, "长白山火山区域 (吉林/朝鲜边境)"),
        ],
    },

    # ── 喀拉峻天山草甸低门槛 ────────────────────────────────────────────
    "kalajun_tianshan_meadow_low": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(42.0, 44.5, 80.0, 83.5, "天山高山草甸区域 (喀拉峻/那拉提/巴音布鲁克)"),
        ],
    },

    # ── 长白山阔叶林低门槛 ──────────────────────────────────────────────
    "changbaishan_broadleaf_low": {
        "conditions": {
            "climate_zone": "temperate",
            "vegetation_zone": "broadleaf_deciduous",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(41.0, 44.0, 124.0, 129.0, "长白山阔叶林区域 (吉林/辽宁)"),
        ],
    },

    # ── 华山/秦岭花岗岩低门槛 ──────────────────────────────────────────
    "huashan_qinling_granite_low": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(33.5, 35.0, 108.5, 111.0, "华山/秦岭东段花岗岩山区"),
        ],
    },

    # ── 昆仑高寒荒漠低门槛 ──────────────────────────────────────────────
    "kunlun_alpine_desert_low": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "desert",
        },
        "bboxes": [
            BBox(34.0, 40.0, 75.0, 92.0, "昆仑山/喀喇昆仑高寒荒漠区域"),
        ],
    },

    # ── 大娄山石灰岩低门槛 ──────────────────────────────────────────────
    "daloushan_limestone_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "karst_peaks",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(27.0, 30.0, 105.0, 109.0, "大娄山/黔北石灰岩区域 (遵义/毕节)"),
        ],
    },

    # ── 武夷山丹霞低门槛 ─────────────────────────────────────────────────
    "wuyishan_danxia_low": {
        "conditions": {
            "mountain_rock_type": "red_sandstone_danxia",
            "climate_zone": "subtropical",
        },
        "bboxes": [
            BBox(27.0, 29.0, 117.0, 119.0, "武夷山/闽北丹霞区域"),
        ],
    },

    # ── 云南高原针叶林红壤低门槛 ──────────────────────────────────────
    "yunnan_plateau_conifer_red_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "vegetation_zone": "conifer_forest",
            "soil_color": "red",
        },
        "bboxes": [
            BBox(23.5, 27.5, 99.0, 104.5, "云南高原针叶林红壤区域 (滇中高原)"),
        ],
    },

    # ── 若尔盖高寒湿地低门槛 ────────────────────────────────────────────
    "zoige_alpine_wetland_low": {
        "conditions": {
            "vegetation_zone": "alpine_meadow",
            "terrain_type": "plateau",
            "climate_zone": "alpine",
        },
        "bboxes": [
            BBox(32.0, 35.0, 100.5, 104.0, "若尔盖/川西北高寒湿地草原区域"),
        ],
    },

    # ── 罗霄山/井冈山低门槛 ────────────────────────────────────────────
    "luoxiao_jinggangshan_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(25.5, 28.0, 113.0, 116.0, "罗霄山脉区域 (井冈山/武功山南段)"),
        ],
    },

    # ── 南岭低门槛 ──────────────────────────────────────────────────────
    "nanling_subtropical_mountain_low": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(24.0, 26.5, 110.0, 115.0, "南岭山脉区域 (湘赣粤桂交界)"),
        ],
    },

    # ── 虎跳峡低门槛 ──────────────────────────────────────────────────
    "tiger_leaping_gorge_low": {
        "conditions": {
            "landform_detail": "river_canyon",
            "climate_zone": "subtropical",
            "mountain_rock_type": "snow_peaks_glaciers",
        },
        "bboxes": [
            BBox(26.5, 28.0, 99.5, 101.0, "虎跳峡/金沙江峡谷区域"),
        ],
    },

    # ── 扎尕那石灰岩低门槛 ────────────────────────────────────────────
    "zhagana_limestone_low": {
        "conditions": {
            "terrain_type": "karst_peaks",
            "climate_zone": "alpine",
            "language_script": "tibetan",
        },
        "bboxes": [
            BBox(33.0, 35.5, 101.0, 104.5, "甘南/川西北石灰岩藏区 (扎尕那/郎木寺)"),
        ],
    },




    # ═══════════════════════════════════════════════════════════════════════════
    # >40N City Fingerprints — 9 missing cities + low variants + competition mirror
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 齐齐哈尔 (Qiqihar): 嫩江平原, 重工业+黑土 ───────────────────────
    "qiqihar_black_soil_industrial": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
            "building_height": "mid_rise",
            "vegetation_zone": "broadleaf_deciduous",
        },
        "bboxes": [
            BBox(47.0, 47.6, 123.5, 124.5, "齐齐哈尔 (嫩江平原, 重工业+黑土)"),
        ],
    },
    "qiqihar_city_low": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
        },
        "bboxes": [
            BBox(46.0, 48.5, 122.0, 126.0, "黑龙江西部城市 (齐齐哈尔/大庆)"),
        ],
    },

    # ── 牡丹江 (Mudanjiang): 长白山北麓, 朝鲜族文化 ────────────────────
    "mudanjiang_border_city": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "mixed_forest",
        },
        "bboxes": [
            BBox(44.3, 44.8, 129.3, 130.0, "牡丹江 (长白山北麓, 中俄朝边境)"),
        ],
    },
    "mudanjiang_city_low": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(43.5, 45.5, 128.5, 131.0, "黑龙江南部城市 (牡丹江/绥芬河)"),
        ],
    },

    # ── 吉林市 (Jilin City): 松花江畔, 雾凇之都 ───────────────────────
    "jilin_city_river_industrial": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
            "building_height": "mid_rise",
            "water_visible": "yes",
        },
        "bboxes": [
            BBox(43.5, 44.0, 126.2, 127.0, "吉林市 (松花江畔, 化工+雾凇)"),
        ],
    },
    "jilin_city_low": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
        },
        "bboxes": [
            BBox(43.0, 44.5, 125.5, 127.5, "吉林省中部城市 (吉林市)"),
        ],
    },

    # ── 包头 (Baotou): 草原钢城, 稀土之都 ──────────────────────────────
    "baotou_steppe_industrial": {
        "conditions": {
            "climate_zone": "arid",
            "urbanization": "city",
            "terrain_type": "grassland_steppe",
            "vegetation_zone": "grassland_steppe",
            "building_height": "mid_rise",
        },
        "bboxes": [
            BBox(40.3, 40.8, 109.5, 110.5, "包头 (草原钢城, 阴山南麓)"),
        ],
    },
    "baotou_city_low": {
        "conditions": {
            "climate_zone": "arid",
            "urbanization": "city",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(40.0, 41.5, 109.0, 111.0, "内蒙古中部城市 (包头/鄂尔多斯)"),
        ],
    },

    # ── 赤峰 (Chifeng): 辽西, 农牧交错带 ────────────────────────────────
    "chifeng_farming_pastoral": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "rolling_hills",
            "vegetation_zone": "grassland_steppe",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(42.0, 42.5, 118.5, 119.5, "赤峰 (辽西, 农牧交错带)"),
        ],
    },
    "chifeng_city_low": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "vegetation_zone": "grassland_steppe",
        },
        "bboxes": [
            BBox(41.5, 43.0, 117.5, 120.5, "辽西/内蒙古东南部城市 (赤峰/通辽南)"),
        ],
    },

    # ── 通辽 (Tongliao): 科尔沁草原腹地 ────────────────────────────────
    "tongliao_grassland_city": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "grassland_steppe",
            "vegetation_zone": "grassland_steppe",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(43.3, 43.9, 122.0, 123.0, "通辽 (科尔沁草原腹地)"),
        ],
    },
    "tongliao_city_low": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(43.0, 44.5, 121.5, 123.5, "科尔沁地区城市 (通辽)"),
        ],
    },

    # ── 海拉尔 (Hailar): 呼伦贝尔草原, 中俄蒙边境 ─────────────────────
    "hailar_steppe_border": {
        "conditions": {
            "climate_zone": "boreal",
            "urbanization": "town",
            "terrain_type": "grassland_steppe",
            "vegetation_zone": "grassland_steppe",
            "language_script": "mongolian_cyrillic",
        },
        "bboxes": [
            BBox(49.0, 49.5, 119.5, 120.5, "海拉尔 (呼伦贝尔草原, 中俄蒙边境)"),
        ],
    },
    "hailar_city_low": {
        "conditions": {
            "climate_zone": "boreal",
            "urbanization": "town",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(48.0, 50.0, 118.0, 121.5, "呼伦贝尔城市区域 (海拉尔/满洲里)"),
        ],
    },

    # ── 佳木斯 (Jiamusi): 三江平原, 中国东极 ───────────────────────────
    "jiamusi_river_plain": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
            "terrain_type": "urban_flat",
            "vegetation_zone": "broadleaf_deciduous",
        },
        "bboxes": [
            BBox(46.5, 47.0, 130.0, 130.8, "佳木斯 (三江平原, 松花江畔)"),
        ],
    },
    "jiamusi_city_low": {
        "conditions": {
            "climate_zone": "boreal",
            "soil_color": "black",
            "urbanization": "city",
            "terrain_type": "urban_flat",
        },
        "bboxes": [
            BBox(46.0, 47.5, 129.5, 132.0, "三江平原城市 (佳木斯/双鸭山/鹤岗)"),
        ],
    },

    # ── 大同 (Datong): 煤都, 云冈石窟, 雁北重镇 ────────────────────────
    "datong_loess_coal": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "plateau",
            "soil_color": "yellow_brown",
            "vegetation_zone": "grassland_steppe",
        },
        "bboxes": [
            BBox(39.8, 40.3, 113.0, 113.8, "大同 (煤都/云冈石窟, 雁北盆地)"),
        ],
    },
    "datong_city_low": {
        "conditions": {
            "climate_zone": "temperate",
            "urbanization": "city",
            "terrain_type": "plateau",
            "soil_color": "yellow_brown",
        },
        "bboxes": [
            BBox(39.5, 41.0, 112.0, 114.5, "晋北城市 (大同/朔州)"),
        ],
    },

    # ── >40N 城市通用竞争镜像 ───────────────────────────────────────────
    "northeast_china_city_generic": {
        "conditions": {
            "climate_zone": "boreal",
            "urbanization": "city",
            "soil_color": "black",
        },
        "bboxes": [
            BBox(41.0, 49.0, 122.0, 132.0, "东北城市通用 (黑吉辽)"),
        ],
    },
    "inner_mongolia_city_generic": {
        "conditions": {
            "climate_zone": "arid",
            "urbanization": "city",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(39.0, 50.0, 106.0, 121.0, "内蒙古城市通用 (呼和浩特/包头/海拉尔)"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Missing low variants for existing >40N cities
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 哈尔滨低门槛 ────────────────────────────────────────────────────
    "harbin_ne_urban_low": {
        "conditions": {
            "soil_color": "black",
            "urbanization": "city",
            "climate_zone": "boreal",
        },
        "bboxes": [
            BBox(44.5, 48.0, 125.0, 128.0, "哈尔滨/黑龙江中部城市区域"),
        ],
    },

    # ── 沈阳低门槛 ──────────────────────────────────────────────────────
    "shenyang_temperate_mid_low": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "city",
        },
        "bboxes": [
            BBox(41.0, 43.0, 122.5, 125.0, "辽宁中部城市区域 (沈阳/抚顺/鞍山)"),
        ],
    },

    # ── 长春低门槛 ──────────────────────────────────────────────────────
    "changchun_temperate_mid_low": {
        "conditions": {
            "building_height": "mid_rise",
            "climate_zone": "temperate",
            "urbanization": "city",
            "soil_color": "black",
        },
        "bboxes": [
            BBox(43.0, 44.5, 124.5, 126.5, "长春/吉林省中部城市区域"),
        ],
    },

    # ── 呼和浩特低门槛 ──────────────────────────────────────────────────
    "hohhot_arid_steppe_low": {
        "conditions": {
            "climate_zone": "arid",
            "urbanization": "city",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(40.0, 41.5, 110.5, 113.0, "呼和浩特/包头/鄂尔多斯区域"),
        ],
    },

    # ── 乌鲁木齐低门槛 ──────────────────────────────────────────────────
    "urumqi_arid_mid_low": {
        "conditions": {
            "climate_zone": "arid",
            "urbanization": "city",
            "language_script": "arabic",
        },
        "bboxes": [
            BBox(43.0, 45.0, 87.0, 89.0, "北疆城市区域 (乌鲁木齐/昌吉)"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════
    # P1: Targeted fixes for 500-test failure patterns
    # ═══════════════════════════════════════════════════════════════════

    # ── P1-1: Yunnan-Guizhou Plateau (temperate mislabel) ─────────────

    # NE China temperate has deciduous or mixed forest, NOT broadleaf_evergreen.
    # red soil only occurs in S/SW China — combined with temperate = Yunnan Plateau.
    "temperate_red_rolling_hills": {
        "conditions": {
            "climate_zone": "temperate",
            "soil_color": "red",
            "terrain_type": "rolling_hills",
        },
        "bboxes": [
            BBox(23.0, 28.0, 100.0, 106.0, "Yunnan-Guizhou Plateau temperate red soil hills"),
        ],
    },

    # ── P1-2: Huangshan / SE China granite mountains ──────────────────
    # VLM uses "temperate" instead of "subtropical" for Huangshan (~30N, 1800m).
    # Granite + broadleaf_evergreen is the Huangshan fingerprint.
    "huangshan_temperate_granite": {
        "conditions": {
            "mountain_rock_type": "granite_spheroidal",
            "climate_zone": "temperate",
            "vegetation_zone": "broadleaf_evergreen",
        },
        "bboxes": [
            BBox(29.5, 31.0, 117.5, 119.0, "Huangshan temperate granite + evergreen"),
            BBox(27.0, 29.0, 117.0, 119.0, "Wuyi Mountains granite (N Fujian)"),
        ],
    },

    # ── P1-3: Latitude-aware alpine (Tianshan vs Tibet) ───────────────
    # VLM cannot distinguish Tianshan (43N) from Tibet (30N) — both get
    # alpine + sharp_mountains + alpine_meadow. Granite_spheroidal is a
    # Tianshan signal (Tibet peaks are mostly sedimentary/metamorphic).
    "tianshan_alpine_granite": {
        "conditions": {
            "climate_zone": "alpine",
            "mountain_rock_type": "granite_spheroidal",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(41.0, 45.0, 80.0, 90.0, "Tianshan alpine granite + meadow (high latitude)"),
            BBox(45.0, 49.0, 86.0, 92.0, "Altai alpine granite (high latitude)"),
        ],
    },

    # Snow peaks with meadow at high latitude → Tianshan, not Tibet.
    # Tibet snow peaks typically lack alpine_meadow (too arid, desert scrub instead).
    "tianshan_snow_meadow": {
        "conditions": {
            "climate_zone": "alpine",
            "mountain_rock_type": "snow_peaks_glaciers",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(41.0, 44.0, 80.0, 89.0, "Tianshan snow peaks + meadow (high latitude)"),
        ],
    },

    # Alpine meadow + grassland_steppe → Tianshan/Kalajun. Distinct from
    # Tibet which has alpine_meadow + desert_scrub at similar elevations.
    "tianshan_meadow_steppe": {
        "conditions": {
            "climate_zone": "alpine",
            "vegetation_zone": "alpine_meadow",
            "terrain_type": "grassland_steppe",
        },
        "bboxes": [
            BBox(41.5, 44.5, 80.0, 87.0, "Tianshan alpine meadow-steppe (Kalajun/Narat)"),
        ],
    },

    # ── Tianshan north/south slope asymmetry ────────────────────────────────
    # North slope (facing Junggar Basin): captures Atlantic moisture →
    # boreal conifer forest + alpine meadow. South slope (facing Tarim Basin):
    # rain-shadow → arid steppe → desert transition. This N/S asymmetry
    # is the defining feature of Tianshan geography.
    "tianshan_north_slope_boreal": {
        "conditions": {
            "mountain_rock_type": "snow_peaks_glaciers",
            "climate_zone": "boreal",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(42.5, 45.5, 80.0, 91.0, "Tianshan N slope: boreal conifer (Ili/Narat)"),
        ],
    },
    "tianshan_south_slope_arid": {
        "conditions": {
            "climate_zone": "arid",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "sparse",
        },
        "bboxes": [
            BBox(40.5, 43.0, 78.0, 89.0, "Tianshan S slope: arid rain-shadow"),
        ],
    },

    # ── Shangri-La / NW Yunnan high plateau (elevation >2500m) ──────────
    # VLM often labels these as subtropical + sharp_mountains, which would
    # normally map to SE China (~30N) or Sichuan. The conifer_forest or
    # alpine_meadow vegetation at subtropical latitude is the key signal
    # for NW Yunnan (26.5-28N, 99-101E). Bbox is tight because these
    # conditions are very specific to the Hengduan foothills.
    "shangrila_high_subtropical_conifer": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "conifer_forest",
        },
        "bboxes": [
            BBox(25.5, 28.5, 98.0, 102.0, "Shangri-La/NW Yunnan high subtropical conifer"),
        ],
    },
    "shangrila_high_subtropical_meadow": {
        "conditions": {
            "climate_zone": "subtropical",
            "terrain_type": "sharp_mountains",
            "vegetation_zone": "alpine_meadow",
        },
        "bboxes": [
            BBox(25.5, 28.5, 98.0, 101.5, "Shangri-La/NW Yunnan high subtropical meadow"),
        ],
    },

    # ── Low-threshold Tianshan (no rock_type needed) ────────────────────
    # Existing Tianshan scenes require mountain_rock_type or snow_peaks
    # which VLM rarely outputs. This low-threshold variant catches any
    # alpine + sharp_mountains combination and maps it to >40N (Tianshan/
    # Altai), distinguishing from Tibet which is at 28-35N.
    "tianshan_alpine_low": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(41.0, 49.0, 80.0, 92.0, "Tianshan/Altai alpine mountains (>40N)"),
        ],
    },
    # Tibet counterpart: alpine + sharp_mountains at 28-35N
    "tibet_alpine_low": {
        "conditions": {
            "climate_zone": "alpine",
            "terrain_type": "sharp_mountains",
        },
        "bboxes": [
            BBox(28.0, 35.0, 78.0, 98.0, "Tibet/Qinghai alpine mountains (28-35N)"),
        ],
    },

}


# ── Fuzzy matching rules for VLM common confusions ──────────────────────
# Maps (category, expected_value) → set of acceptable VLM output values

_FUZZY_RULES = {
    # soil_color: "yellow_brown" never matches because prompt only has
    # "yellow" and "brown" separately — accept either
    ("soil_color", "yellow_brown"): {"yellow", "brown"},

    # vegetation_zone: broadleaf family confusion
    ("vegetation_zone", "broadleaf_evergreen"): {"broadleaf_deciduous"},

    # vegetation_zone: conifer/mixed overlap
    ("vegetation_zone", "conifer_forest"): {"mixed_forest"},

    # terrain_type: karst visually similar to generic sharp mountains
    ("terrain_type", "karst_peaks"): {"sharp_mountains"},

    # terrain_type: sandstone pillars confused with karst
    ("terrain_type", "sandstone_pillars"): {"karst_peaks", "sharp_mountains"},

    # mountain_rock_type: granite prefix match
    ("mountain_rock_type", "granite_spheroidal"): {"granite_dome", "granite_tors"},

    # mountain_rock_type: red sandstone confused with quartz sandstone
    ("mountain_rock_type", "red_sandstone_danxia"): {"quartz_sandstone_pillars"},
    ("mountain_rock_type", "quartz_sandstone_pillars"): {"red_sandstone_danxia"},

    # scene_type: mountain trail sub-type confusion
    ("scene_type", "mountain_trail_karst"): {"mountain_trail_granite", "mountain_trail"},

    # climate_zone: alpine / boreal confusion (high elevation vs high latitude)
    ("climate_zone", "alpine"): {"boreal"},
    ("climate_zone", "boreal"): {"alpine"},

    # tree_species: confusable pairs
    ("tree_species", "camphor"): {"banyan"},           # both subtropical broadleaf
    ("tree_species", "chinese_red_pine"): {"conifer"},  # generic conifer

    # mountain_rock_type: alpine lakes ↔ snow peaks (Yading has both)
    ("mountain_rock_type", "alpine_lakes"): {"snow_peaks_glaciers"},
    ("mountain_rock_type", "snow_peaks_glaciers"): {"alpine_lakes"},

    # landform_detail: granite boulder field ↔ rounded granite (鳌太线石海)
    ("landform_detail", "granite_boulder_field"): {"granite_spheroidal_rounded"},
    ("landform_detail", "granite_spheroidal_rounded"): {"granite_boulder_field"},
}


def _fuzzy_match(expected_val: str, actual_val, cat: str) -> bool:
    """Check if actual_val matches expected_val, with fuzzy rules for common VLM confusions."""
    if actual_val is None:
        return False
    # Handle dict values (e.g. nested objects) — can't match
    if isinstance(actual_val, dict):
        return False
    # Handle list values (e.g. tree_species, elevation_estimate_m)
    if isinstance(actual_val, list):
        # For lists, check if expected_val is in the list
        if expected_val in actual_val:
            return True
        # Also check fuzzy rules for each item
        acceptable = _FUZZY_RULES.get((cat, expected_val), set())
        for item in actual_val:
            if item in acceptable:
                return True
        return False
    if expected_val == actual_val:
        return True
    # Check fuzzy rules for this (category, expected) pair
    acceptable = _FUZZY_RULES.get((cat, expected_val), set())
    return actual_val in acceptable


def match_compound_scenes(elements: dict) -> list[tuple[str, list[BBox]]]:
    """Detect compound scenes matching the given geographic elements.

    Each compound scene requires ALL its conditions to match. If matched,
    its tight bboxes are returned as additional constraints.

    Returns:
        List of (scene_name, bboxes) tuples.
    """
    matches = []
    for name, scene in COMPOUND_SCENES.items():
        conditions = scene["conditions"]
        if all(
            elements.get(cat) == val
            for cat, val in conditions.items()
        ):
            matches.append((name, scene["bboxes"]))
    return matches


def match_compound_scenes_soft(
    elements: dict, min_ratio: float = 0.6, min_matches: int = 1,
) -> list[tuple[str, list[BBox], float]]:
    """Fuzzy compound scene matching with partial credit.

    Unlike match_compound_scenes() which requires exact string match on ALL
    conditions, this version:
      1. Uses fuzzy rules (_FUZZY_RULES) to accept VLM-common near-misses
      2. Returns a match_ratio = matched / total for partial matches
      3. Only returns scenes with match_ratio >= min_ratio AND matched >= min_matches

    min_matches=2 prevents 1-condition matches on 2-3 condition scenes
    from triggering false positives (e.g. Shanghai matching harbin_ne_urban
    with only temperate+medium_city=2/4).

    Returns:
        List of (scene_name, bboxes, match_ratio) tuples, sorted by ratio desc.
    """
    matches = []
    for name, scene in COMPOUND_SCENES.items():
        conditions = scene["conditions"]
        if not conditions:
            continue
        matched = 0
        for cat, expected_val in conditions.items():
            actual_val = elements.get(cat)
            if _fuzzy_match(expected_val, actual_val, cat):
                matched += 1
        ratio = matched / len(conditions)
        if ratio >= min_ratio and matched >= min_matches:
            matches.append((name, scene["bboxes"], ratio))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches


# ═══════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE TAGS — visible infrastructure clues from GeoCoT regional stage
# ═══════════════════════════════════════════════════════════════════════════

INFRASTRUCTURE_TAGS = {
    "electric_scooters": [
        BBox(22.0, 33.0, 104.0, 122.5, "S China e-scooter cities"),
    ],
    "tibetan_signs": [
        BBox(26.5, 34.0, 88.0, 99.5, "Tibet proper"),
        BBox(31.5, 39.5, 89.0, 103.5, "Qinghai / S Gansu"),
        BBox(26.0, 33.0, 97.0, 104.0, "W Sichuan (Ganzi/ABA)"),
        BBox(27.0, 29.5, 98.5, 101.0, "NW Yunnan (Diqing)"),
    ],
    "cable_cars": [
        BBox(18.0, 45.0, 97.0, 122.5, "Major Chinese mountain scenic areas"),
    ],
    "green_taxis": [
        BBox(22.5, 23.5, 113.0, 115.0, "Guangzhou / Pearl River Delta"),
        BBox(23.5, 26.0, 116.0, 121.0, "S Fujian (Xiamen area)"),
        BBox(30.5, 31.5, 121.0, 122.0, "Shanghai"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# LANDFORM DETAILS — micro-geomorphology from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

LANDFORM_DETAILS = {
    "limestone_tower_karst_scattered": [
        BBox(24.0, 26.0, 109.5, 111.5, "Guilin-Yangshuo scattered tower karst"),
    ],
    "limestone_tower_karst_dense": [
        BBox(24.5, 26.5, 104.5, 109.0, "S Guizhou / N Guangxi dense karst pinnacles"),
    ],
    "granite_spheroidal_rounded": [
        BBox(29.5, 31.0, 117.5, 119.0, "Huangshan / Sanqingshan rounded granite"),
        BBox(28.5, 30.0, 115.5, 117.0, "Lushan / Wudangshan rounded granite"),
    ],
    "granite_spheroidal_sharp": [
        BBox(28.5, 29.5, 117.5, 118.5, "Sanqingshan sharp granite"),
        BBox(34.0, 35.0, 109.5, 111.0, "Huashan sharp granite"),
        BBox(26.0, 32.0, 97.0, 104.0, "Hengduan sharp granite massif"),
    ],
    "quartz_sandstone_pillars": [
        BBox(28.5, 30.0, 109.0, 111.5, "Zhangjiajie / Wulingyuan quartz pillars"),
    ],
    "red_sandstone_danxia": [
        BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan Danxia"),
        BBox(38.5, 39.5, 99.5, 101.0, "Zhangye Danxia"),
        BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan (Guangdong)"),
    ],
    "snow_peaks_glaciers": [
        BBox(29.0, 30.5, 101.0, 103.0, "Gongga / Siguniang"),
        BBox(27.0, 29.0, 98.0, 100.5, "Meili / Yulong / Haba"),
        BBox(27.5, 29.0, 85.5, 87.5, "Everest / Cho Oyu"),
        BBox(30.0, 32.0, 90.0, 95.0, "Nyainqentanglha / Namcha Barwa"),
        BBox(41.0, 43.5, 80.0, 89.0, "Tianshan glaciers"),
    ],
    "alpine_meadow": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet alpine meadow"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai alpine meadow"),
        BBox(26.0, 32.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan alpine meadow"),
    ],
    "plateau_plain": [
        BBox(26.5, 36.5, 78.0, 99.5, "Tibet Plateau"),
        BBox(31.0, 39.5, 89.0, 103.5, "Qinghai Plateau"),
    ],
    "river_canyon": [
        BBox(26.0, 30.0, 98.5, 101.0, "Tiger Leaping Gorge / Nu River"),
        BBox(30.5, 31.5, 109.5, 111.5, "Three Gorges (Yangtze)"),
        BBox(29.0, 30.5, 94.0, 96.5, "Yarlung Tsangpo Grand Canyon"),
        BBox(43.0, 44.0, 85.5, 87.5, "Tianshan glacial valleys (Langta)"),
    ],
    "granite_boulder_field": [
        BBox(33.8, 34.3, 107.4, 107.9, "鳌太线 Taibai boulder field (石海) 3000-3767m"),
    ],
    "coastal_shore": [
        BBox(18.0, 42.0, 108.0, 123.0, "China coastline"),
    ],
    "urban_flat": [
        BBox(18.0, 48.5, 73.0, 135.5, "All China urban lowlands"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# WATER TYPES — visible water bodies from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

WATER_TYPES = {
    "ocean": [
        BBox(18.0, 42.0, 108.0, 123.0, "China east / south coastline"),
    ],
    "lake": [
        BBox(27.5, 33.0, 87.0, 91.5, "Tibet lakes (Namtso / Yamdrok)"),
        BBox(41.0, 49.0, 80.0, 92.0, "Xinjiang lakes (Bosten / Kanas)"),
        BBox(28.0, 34.0, 103.0, 111.0, "SW China lakes (Jiuzhaigou / Lugu)"),
        BBox(28.0, 33.0, 116.0, 121.0, "Yangtze Delta lakes (Taihu / Poyang)"),
        BBox(37.0, 39.0, 96.0, 101.0, "Qinghai Lake area"),
    ],
    "glacier": [
        BBox(27.5, 36.5, 78.0, 104.0, "Tibet / W Sichuan glaciers"),
        BBox(41.0, 44.0, 80.0, 89.0, "Tianshan glaciers"),
    ],
    "waterfall": [
        BBox(23.5, 30.5, 104.0, 112.0, "Guizhou / Guangxi waterfall region"),
        BBox(28.5, 30.0, 109.5, 111.5, "Zhangjiajie waterfalls"),
        BBox(24.5, 28.0, 100.5, 101.5, "Yunnan waterfalls"),
    ],
    "river": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China (rivers are ubiquitous)"),
    ],
    "stream": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China (streams are ubiquitous)"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# ROCK COLORS — visible rock surface color from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

ROCK_COLORS = {
    "grey_white": [
        BBox(26.0, 35.0, 97.0, 119.0, "Granite / limestone mountains (wide)"),
    ],
    "red_brown": [
        BBox(27.5, 28.5, 117.5, 118.5, "Wuyishan red sandstone"),
        BBox(38.5, 39.5, 99.5, 101.0, "Zhangye red beds"),
        BBox(24.5, 26.0, 113.0, 114.5, "Danxiashan red sandstone"),
        BBox(28.0, 29.0, 116.5, 117.5, "Longhushan red sandstone"),
        BBox(34.0, 41.0, 105.5, 114.5, "Loess Plateau red-brown earth"),
    ],
    "dark_grey": [
        BBox(26.5, 36.5, 78.0, 104.0, "Tibet / W Sichuan dark metamorphic rock"),
        BBox(26.0, 32.0, 97.0, 104.0, "Hengduan dark granite"),
        BBox(41.0, 44.0, 80.0, 89.0, "Tianshan dark metamorphic"),
    ],
    "yellow_brown": [
        BBox(34.5, 41.0, 105.5, 114.5, "Loess Plateau (Shaanxi / Shanxi / Gansu)"),
        BBox(24.0, 33.0, 108.5, 122.5, "Mid-China yellow-brown earth"),
    ],
    "black": [
        BBox(41.5, 42.5, 127.0, 129.0, "Changbaishan black basalt"),
        BBox(19.0, 20.5, 109.5, 111.5, "Hainan basalt"),
    ],
    "white": [
        BBox(34.0, 42.5, 88.0, 96.5, "Qaidam Basin white saline soil"),
        BBox(27.5, 36.5, 78.0, 88.0, "W Tibet white evaporite"),
    ],
    "mixed": [
        BBox(18.0, 54.0, 73.0, 135.5, "All China (non-specific)"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# SCENE TYPES — high-level scene category from GeoCoT local stage
# ═══════════════════════════════════════════════════════════════════════════

SCENE_TYPE_REGIONS = {
    "desert": [
        BBox(34.0, 49.5, 73.0, 107.0, "NW China deserts (Taklamakan / Gobi / Tengger)"),
    ],
    "grassland": [
        BBox(41.0, 51.0, 110.0, 126.5, "Inner Mongolia grassland"),
        BBox(32.0, 39.5, 89.0, 103.5, "Qinghai / Gansu alpine grassland"),
        BBox(27.0, 33.0, 97.0, 104.0, "W Sichuan / NW Yunnan alpine grassland"),
        BBox(41.0, 49.0, 80.0, 92.0, "N Xinjiang grassland (Ili / Altai)"),
    ],
    "coastal": [
        BBox(18.0, 42.0, 108.0, 125.0, "China east / south coastline"),
    ],
    "river_valley": [
        BBox(24.0, 35.0, 96.0, 118.0, "Hengduan river canyons + Three Gorges"),
    ],
    "city_street": [
        BBox(18.0, 48.5, 73.0, 135.5, "All China urban"),
    ],
    "mountain_trail_karst": [
        BBox(23.5, 30.5, 103.5, 115.0, "S China karst belt"),
    ],
    "mountain_trail_granite": [
        BBox(26.0, 35.0, 97.0, 119.0, "China granite mountain belt"),
        BBox(29.0, 31.0, 117.0, 119.0, "Huangshan / Sanqingshan granite"),
        BBox(33.5, 35.0, 107.0, 111.0, "Huashan / Qinling granite"),
    ],
    "mountain_trail_snow": [
        BBox(26.5, 36.5, 78.0, 104.0, "Tibet / W Sichuan snow mountains"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan snow mountains"),
    ],
    "rural_village": [
        BBox(22.0, 45.0, 104.0, 126.5, "Agricultural China (dense rural)"),
    ],
    "tourist_scenic": [
        BBox(18.0, 45.0, 73.0, 135.5, "All China scenic areas"),
    ],
    "wilderness_forest": [
        BBox(18.0, 54.0, 108.0, 135.5, "E China forest belt"),
        BBox(26.0, 34.5, 97.0, 104.0, "Hengduan forest"),
        BBox(48.0, 54.0, 121.0, 135.5, "NE China boreal forest"),
        BBox(41.0, 45.0, 80.0, 96.5, "Tianshan spruce forest"),
    ],
}


PROVINCE_BBOXES = {
    # Province-level bounding boxes for provincial anchoring
    "Beijing":       [BBox(39.4, 41.0, 115.4, 117.5, "Beijing")],
    "Tianjin":       [BBox(38.5, 40.2, 116.6, 118.0, "Tianjin")],
    "Shanghai":      [BBox(30.6, 31.9, 120.8, 122.0, "Shanghai")],
    "Chongqing":     [BBox(28.1, 32.2, 105.3, 110.2, "Chongqing")],
    "Hebei":         [BBox(36.0, 42.6, 113.5, 119.9, "Hebei")],
    "Shanxi":        [BBox(34.5, 40.8, 110.2, 114.5, "Shanxi")],
    "Liaoning":      [BBox(38.7, 43.5, 118.8, 125.8, "Liaoning")],
    "Jilin":         [BBox(40.8, 46.3, 121.6, 131.3, "Jilin")],
    "Heilongjiang":  [BBox(43.4, 53.6, 121.2, 135.1, "Heilongjiang")],
    "Jiangsu":       [BBox(30.7, 35.1, 116.3, 122.0, "Jiangsu")],
    "Zhejiang":      [BBox(27.0, 31.2, 118.0, 123.0, "Zhejiang")],
    "Anhui":         [BBox(29.4, 34.7, 114.9, 119.6, "Anhui")],
    "Fujian":        [BBox(23.5, 28.7, 115.8, 120.7, "Fujian")],
    "Jiangxi":       [BBox(24.5, 30.1, 113.6, 118.5, "Jiangxi")],
    "Shandong":      [BBox(34.3, 38.3, 114.8, 122.7, "Shandong")],
    "Henan":         [BBox(31.4, 36.4, 110.3, 116.7, "Henan")],
    "Hubei":         [BBox(29.0, 33.3, 108.3, 116.2, "Hubei")],
    "Hunan":         [BBox(24.6, 30.1, 108.8, 114.2, "Hunan")],
    "Guangdong":     [BBox(20.2, 25.5, 109.6, 117.3, "Guangdong")],
    "Guangxi":       [BBox(21.4, 26.3, 104.5, 112.0, "Guangxi")],
    "Hainan":        [BBox(18.1, 20.3, 108.6, 111.0, "Hainan")],
    "Sichuan":       [BBox(26.0, 34.3, 97.3, 108.5, "Sichuan")],
    "Guizhou":       [BBox(24.6, 29.2, 103.6, 109.6, "Guizhou")],
    "Yunnan":        [BBox(21.1, 29.3, 97.5, 106.2, "Yunnan")],
    "Tibet":         [BBox(26.8, 36.5, 78.4, 99.1, "Tibet (Xizang)")],
    "Shaanxi":       [BBox(31.7, 39.6, 105.5, 111.3, "Shaanxi")],
    "Gansu":         [BBox(32.5, 42.8, 92.7, 108.8, "Gansu")],
    "Qinghai":       [BBox(31.5, 39.3, 89.4, 103.1, "Qinghai")],
    "Xinjiang":      [BBox(34.3, 49.2, 73.4, 96.4, "Xinjiang")],
    "Inner Mongolia":[BBox(37.4, 53.4, 97.2, 126.1, "Inner Mongolia")],
    "Ningxia":       [BBox(35.2, 39.3, 104.3, 107.6, "Ningxia")],
    "Hong Kong":     [BBox(22.1, 22.6, 113.8, 114.4, "Hong Kong")],
    "Macau":         [BBox(22.0, 22.3, 113.4, 113.7, "Macau")],
    "Taiwan":        [BBox(21.9, 25.3, 120.0, 122.0, "Taiwan")],
}

URBANIZATION_LEVELS = {
    "metropolis": [
        # Tier-1 megacities (>10M urban pop) — tight ~0.5° bboxes
        BBox(39.6, 40.2, 116.0, 116.8, "Beijing"),
        BBox(30.9, 31.6, 121.1, 121.8, "Shanghai"),
        BBox(22.4, 23.3, 113.0, 114.4, "Guangzhou / Shenzhen / Dongguan"),
        BBox(30.3, 30.9, 103.8, 104.3, "Chengdu"),
        BBox(29.2, 29.8, 106.2, 106.8, "Chongqing"),
        BBox(30.2, 30.9, 113.9, 114.7, "Wuhan"),
        BBox(31.7, 32.5, 118.5, 119.0, "Nanjing"),
        BBox(30.0, 30.5, 119.9, 120.5, "Hangzhou"),
        BBox(38.8, 39.4, 117.0, 117.7, "Tianjin"),
        BBox(34.0, 34.5, 108.6, 109.3, "Xi'an"),
        BBox(28.0, 28.4, 112.7, 113.3, "Changsha"),
        BBox(24.2, 24.8, 117.8, 118.5, "Xiamen"),
        BBox(22.5, 23.3, 112.8, 113.7, "Foshan / Zhongshan / Zhuhai"),
    ],
    "medium_city": [
        # Provincial capitals and prefecture cities (~50-100km bboxes)
        BBox(45.3, 46.1, 126.3, 127.2, "Harbin"),
        BBox(43.5, 44.3, 125.0, 126.0, "Changchun / Jilin"),
        BBox(41.5, 42.1, 123.0, 124.0, "Shenyang"),
        BBox(36.3, 36.9, 116.8, 117.4, "Jinan"),
        BBox(34.3, 35.0, 113.2, 114.0, "Zhengzhou"),
        BBox(37.5, 38.2, 112.2, 113.0, "Taiyuan"),
        BBox(31.5, 32.4, 116.3, 117.5, "Hefei"),
        BBox(28.3, 29.0, 115.5, 116.2, "Nanchang"),
        BBox(25.8, 26.4, 119.0, 119.7, "Fuzhou"),
        BBox(22.5, 23.2, 107.8, 108.8, "Nanning"),
        BBox(19.7, 20.3, 109.8, 110.7, "Haikou"),
        BBox(24.8, 25.6, 102.3, 103.1, "Kunming"),
        BBox(26.2, 26.9, 106.3, 107.1, "Guiyang"),
        BBox(36.3, 37.0, 103.3, 104.3, "Lanzhou"),
        BBox(36.2, 37.0, 101.3, 102.3, "Xining"),
        BBox(43.3, 44.2, 87.2, 88.3, "Urumqi"),
        BBox(38.3, 39.0, 106.0, 106.7, "Yinchuan"),
        BBox(29.3, 29.9, 90.7, 91.5, "Lhasa"),
        BBox(48.0, 48.5, 88.5, 89.5, "Altay"),
        BBox(49.0, 49.5, 119.4, 120.0, "Hailar / Hulunbuir"),
        BBox(31.0, 31.8, 108.0, 108.8, "Dazhou / Wanzhou"),
        BBox(25.8, 26.5, 113.0, 113.6, "Chenzhou / Hengyang"),
        BBox(32.0, 32.8, 119.0, 120.0, "Yangzhou / Zhenjiang"),
        BBox(27.3, 28.0, 119.3, 120.5, "Wenzhou / Ningde"),
    ],
    "small_town": [
        # County towns — most populated areas of China
        BBox(18.0, 48.0, 97.0, 135.5, "E/SE China populated belt"),
    ],
    "village": [
        # Rural villages — similar to small_town but more widespread
        BBox(18.0, 48.0, 97.0, 135.5, "Rural settled China"),
    ],
    "rural": [
        # Agricultural countryside — China's populated plains and valleys
        BBox(18.0, 48.0, 97.0, 135.5, "Populated China"),
        # But exclude true wilderness — rural doesn't mean uninhabited desert
    ],
    "wilderness": [
        # True uninhabited / very sparsely populated areas
        BBox(26.5, 36.5, 78.0, 95.5, "NW Tibet / W Qinghai uninhabited"),
        BBox(36.5, 42.0, 78.0, 88.0, "Taklamakan Desert"),
        BBox(40.0, 47.0, 88.0, 99.0, "Gobi / E Xinjiang"),
        BBox(27.0, 32.0, 97.0, 100.5, "Hengduan canyons wilderness"),
        BBox(44.0, 49.0, 85.0, 90.0, "Altai wilderness"),
        BBox(33.0, 37.0, 88.0, 95.0, "Qaidam Basin / Hoh Xil"),
    ],
}

# ── Building height (city-scale skyline distinction) ──────────────────
BUILDING_HEIGHT = {
    "super_tall": [
        # CBDs with iconic >300m skyscrapers
        BBox(31.20, 31.30, 121.44, 121.55, "Shanghai Lujiazui"),
        BBox(22.51, 22.56, 113.92, 114.02, "Shenzhen Futian CBD"),
        BBox(22.49, 22.54, 114.03, 114.10, "Shenzhen Nanshan"),
        BBox(23.10, 23.15, 113.30, 113.37, "Guangzhou Tianhe CBD"),
        BBox(39.88, 39.96, 116.42, 116.50, "Beijing CBD Guomao"),
        BBox(31.97, 32.05, 118.72, 118.80, "Nanjing Hexi CBD"),
        BBox(39.05, 39.13, 117.35, 117.45, "Tianjin Binhai"),
        BBox(30.55, 30.63, 114.28, 114.38, "Wuhan Jiang'an CBD"),
    ],
    "high_rise": [
        # 20-40 floor dense commercial/residential — 1st/2nd tier city cores
        BBox(31.15, 31.40, 121.30, 121.65, "Shanghai inner ring"),
        BBox(22.48, 22.65, 113.85, 114.20, "Shenzhen urban core"),
        BBox(23.05, 23.20, 113.22, 113.45, "Guangzhou urban core"),
        BBox(39.85, 40.05, 116.30, 116.55, "Beijing urban core"),
        BBox(30.55, 30.75, 103.95, 104.20, "Chengdu south / Tianfu"),
        BBox(29.50, 29.65, 106.45, 106.65, "Chongqing Yuzhong / Jiangbei"),
        BBox(30.18, 30.38, 119.95, 120.30, "Hangzhou downtown / Binjiang"),
        BBox(31.95, 32.12, 118.65, 118.85, "Nanjing downtown"),
        BBox(34.20, 34.35, 108.85, 109.05, "Xi'an Gaoxin"),
        BBox(30.55, 30.70, 114.20, 114.40, "Wuhan downtown"),
        BBox(36.60, 36.75, 117.00, 117.15, "Jinan CBD"),
        BBox(28.10, 28.25, 112.90, 113.05, "Changsha downtown"),
        BBox(22.80, 22.95, 113.55, 113.75, "Dongguan downtown"),
        BBox(24.45, 24.60, 118.05, 118.20, "Xiamen downtown"),
        BBox(38.00, 38.15, 114.40, 114.60, "Shijiazhuang downtown"),
    ],
    "mid_rise": [
        # 6-15 floors — most Chinese urban areas
        BBox(18.0, 48.0, 97.0, 135.5, "Urban China (generic mid-rise)"),
    ],
    "low_rise": [
        # 1-3 floors — old towns, small cities, rural towns
        BBox(18.0, 48.0, 97.0, 135.5, "Low-rise China (broad)"),
    ],
    "mixed": [
        BBox(18.0, 48.0, 97.0, 135.5, "Mixed urban China (broad)"),
    ],
    "no_buildings": [
        # Wilderness / rural without visible buildings
        BBox(18.0, 55.0, 73.0, 136.0, "All China — no buildings constraint"),
    ],
}

# ── Pavement type (sidewalk material ≡ city fingerprint) ────────────
PAVEMENT_TYPE = {
    "red_brick_tiles": [
        # Strong Sichuan Basin signal — Chengdu/Chongqing sidewalks
        BBox(28.5, 32.5, 103.0, 110.0, "Sichuan Basin red-brick zone"),
        BBox(25.5, 27.5, 105.5, 108.5, "Guizhou red-brick zone"),
        BBox(23.0, 25.5, 107.0, 110.5, "Guangxi red-brick (partial)"),
    ],
    "grey_concrete": [
        # Dominant in East/North China cities
        BBox(18.0, 45.0, 110.0, 135.5, "E/NE China grey concrete zone"),
        BBox(18.0, 45.0, 97.0, 110.0, "Central China grey concrete"),
    ],
    "asphalt": [
        # Roads everywhere — minimal spatial constraint
        BBox(18.0, 55.0, 73.0, 136.0, "All China asphalt"),
    ],
    "natural": [
        BBox(18.0, 55.0, 73.0, 136.0, "All China natural ground"),
    ],
    "not_visible": [
        BBox(18.0, 55.0, 73.0, 136.0, "All China — no constraint"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Lookup helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_bboxes_for_element(category: str, value) -> list[BBox]:
    """Get bounding boxes for a geographic element value.

    Args:
        category: one of 'climate_zone', 'terrain_type', 'vegetation_zone',
                  'tree_species', 'architecture_style', 'language_script',
                  'mountain_rock_type', 'soil_color', 'sky_quality'
        value: the element value (str or list of str)

    Returns:
        List of BBox objects, empty if not found.
    """
    kb_map = {
        "climate_zone": CLIMATE_ZONES,
        "terrain_type": TERRAIN_TYPES,
        "vegetation_zone": VEGETATION_ZONES,
        "tree_species": TREE_SPECIES,
        "architecture_style": ARCHITECTURE_STYLES,
        "language_script": LANGUAGE_SCRIPTS,
        "mountain_rock_type": MOUNTAIN_ROCK_TYPES,
        "soil_color": SOIL_COLORS,
        "sky_quality": SKY_QUALITY,
        "scene_type": SCENE_TYPE_REGIONS,
        "compound_scene": {},  # dynamic, handled via match_compound_scenes()
        "landform_detail": LANDFORM_DETAILS,
        "water_type": WATER_TYPES,
        "infrastructure_tags": INFRASTRUCTURE_TAGS,
        "rock_color": ROCK_COLORS,
        "urbanization": URBANIZATION_LEVELS,
        "building_height": BUILDING_HEIGHT,
        "pavement_type": PAVEMENT_TYPE,
        "likely_provinces": PROVINCE_BBOXES,
    }

    kb = kb_map.get(category, {})
    if kb is None or value is None:
        return []

    # Handle list values
    if isinstance(value, list):
        result = []
        for v in value:
            result.extend(kb.get(v, []))
        return result

    # Handle dict values (e.g. water_type with nested fields)
    if isinstance(value, dict):
        return []

    return kb.get(value, [])


def get_all_categories() -> list[str]:
    """Return all supported geographic element categories."""
    return [
        "climate_zone", "terrain_type", "vegetation_zone",
        "tree_species", "architecture_style", "language_script",
        "mountain_rock_type", "soil_color", "sky_quality",
        "building_height", "pavement_type", "urbanization",
    ]
