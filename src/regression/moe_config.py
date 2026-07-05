"""MoE configuration: scene taxonomy, landmark coordinates, training hyperparams."""

from enum import IntEnum


class SceneType(IntEnum):
    URBAN = 0
    KARST_GRANITE = 1
    ALPINE = 2
    OTHER = 3


SCENE_TYPE_NAMES = {
    0: "urban",
    1: "karst_granite",
    2: "alpine_snow",
    3: "other",
}

SCENE_TYPE_LABELS_CN = {
    0: "城市街景",
    1: "喀斯特/花岗岩峰林",
    2: "雪山高原",
    3: "其他地貌",
}

NUM_EXPERTS = 4

# ── Key landmark coordinates per scene type ──────────────────────────
# Each entry: (lat, lng, radius_km)
# Images within radius_km of any landmark get auto-labeled with that scene type.
# Radius is in lat/lng degrees (1° ≈ 111 km at equator, ~102 km at 30°N).

SCENE_LANDMARKS = {
    SceneType.URBAN: [
        # All Chinese city centers are tagged via _CITY_COORD_DB at runtime.
        # This list only contains extra urban anchor points.
    ],
    SceneType.KARST_GRANITE: [
        (30.1296, 118.1690, 0.35),   # Huangshan scenic area
        (29.3355, 110.4807, 0.35),   # Zhangjiajie / Wulingyuan
        (25.2736, 110.2900, 0.40),   # Guilin
        (24.7785, 110.4965, 0.30),   # Yangshuo
        (34.4830, 110.0890, 0.30),   # Huashan
        (28.9150, 118.0820, 0.35),   # Sanqingshan
        (27.7180, 117.6820, 0.35),   # Wuyishan
        (29.5730, 115.9740, 0.30),   # Lushan
        (32.4200, 111.0200, 0.30),   # Wudangshan
        (27.2560, 112.7100, 0.30),   # Mount Heng (Hunan)
        (34.5020, 112.9350, 0.30),   # Mount Song / Shaolin
        (24.5900, 117.0600, 0.30),   # Fujian Tulou area
        (27.9482, 109.5992, 0.20),   # Fenghuang ancient town (granite region)
    ],
    SceneType.ALPINE: [
        (31.1000, 102.9000, 0.45),   # Siguniangshan
        (29.5950, 101.8790, 0.55),   # Gongga Mountain / Minya Konka
        (28.4380, 98.6850, 0.55),    # Meili Snow Mountain / Kawagebo
        (28.4200, 100.3500, 0.45),   # Daocheng Yading
        (29.6500, 91.1000, 0.60),    # Lhasa / Tibet plateau
        (33.2630, 103.9190, 0.35),   # Jiuzhaigou
        (32.7500, 103.8300, 0.30),   # Huanglong
        (28.1417, 86.8550, 0.80),    # Everest Base Camp (Tibet)
        (31.7400, 110.6800, 0.40),   # Shennongjia (high altitude forest)
        (27.0980, 100.1750, 0.40),   # Yulong Snow Mountain
        (27.2900, 100.1300, 0.35),   # Haba Snow Mountain
    ],
    SceneType.OTHER: [
        (27.1980, 100.1183, 0.35),   # Tiger Leaping Gorge
        (30.8300, 111.0000, 0.45),   # Three Gorges area
        (26.8721, 100.2299, 0.25),   # Lijiang Old Town
        (25.6065, 100.2676, 0.25),   # Dali Old Town
        (27.7030, 100.7910, 0.35),   # Lugu Lake
        (33.9300, 108.9700, 0.35),   # Zhongnan Mountain
        (29.5700, 103.7700, 0.30),   # Leshan / Emei foothills
        (36.5500, 109.4900, 0.35),   # Yan'an / Loess Plateau
        (41.5500, 115.9000, 0.50),   # Bashang Grassland
        (48.8200, 87.0400, 0.50),    # Kanas Lake (Xinjiang)
        (38.9258, 100.4498, 0.50),   # Zhangye / Rainbow Mountains
        (40.0370, 94.8040, 0.40),    # Dunhuang / Mogao Caves
        (39.4677, 75.9897, 0.60),    # Kashgar
        (29.2570, 117.8620, 0.25),   # Wuyuan
        (28.3730, 121.0620, 0.30),   # Yandangshan
    ],
}

# ── Urban proximity threshold ────────────────────────────────────────
URBAN_PROXIMITY_KM = 30.0  # images within this distance of any city → URBAN

# ── Scene labeling ───────────────────────────────────────────────────
# Min confidence for auto-labeling via proximity (0.0-1.0)
PROXIMITY_CONFIDENCE_THRESHOLD = 0.8

# ── Sparse expert training (for experts with <100 samples) ───────────
SPARSE_EXPERT_CONFIG = {
    "hidden_dims": [256, 128, 64, 16],   # reduced from [512,256,64,16]
    "dropout": 0.3,                      # increased from 0.1
    "lr": 1e-4,                          # reduced from 3e-4
}

# Default expert config (for experts with >=100 samples)
DEFAULT_EXPERT_CONFIG = {
    "hidden_dims": [512, 256, 64, 16],
    "dropout": 0.1,
    "lr": 3e-4,
}

SPARSE_EXPERT_THRESHOLD = 100   # switch to sparse config below this
MIN_EXPERT_SAMPLES = 20         # merge into Expert 3 below this


# ── Router training ──────────────────────────────────────────────────
ROUTER_HIDDEN_DIMS = [256, 64]
ROUTER_TEMPERATURE = 1.0       # initial softmax temperature
ROUTER_DROPOUT = 0.1

# Router loss weights
ROUTER_ENTROPY_WEIGHT = 0.3       # alpha: prevents collapse to single expert (was 0.05)
ROUTER_LOAD_BALANCE_WEIGHT = 0.05  # beta: encourages even expert usage (was 0.01)
SCENE_CLASS_WEIGHT = 0.3           # gamma: scene classification auxiliary loss (was 0.5)

# ── Training hyperparams ─────────────────────────────────────────────
DEFAULT_EPOCHS = 50
ROUTER_EPOCHS = 50
PATIENCE = 15
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

# For optional joint fine-tuning
JOINT_FINETUNE_LR = 1e-5
JOINT_FINETUNE_EPOCHS = 10

# ── Data augmentation for sparse experts ─────────────────────────────
AUGMENTATION_ENABLED = True
# Applied at feature extraction time (re-extract from augmented images):
# - horizontal flip
# - ±5 degree rotation
# - brightness/contrast ±10%
