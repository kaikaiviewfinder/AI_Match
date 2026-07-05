"""Three-stage distillation training for GeoVLM.

Stage 1 (Representation Alignment): Freeze ViT, train Q-Former + GNN + Heads
  - 200 easy images, ~30 minutes
  - L_task + L_distill

Stage 2 (Full Distillation): Unfreeze ViT last 4 layers, progressive difficulty
  - 500 images, ~3 hours
  - L_task + L_distill + L_feature + L_constraint

Stage 3 (Sensor Grounding): Full model, sensor noise augmentation
  - 200 images, ~1 hour
  - L_task + L_distill + L_sensor (x2 weight)

Loss weights from design spec:
  L_total = 1.0*L_task + 0.3*L_distill + 0.1*L_feature + 0.3*L_sensor + 0.2*L_constraint + 0.01*L_reg
"""
import sys, json, time, os, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from geovlm import GeoVLM, GeoVLMConfig
from geovlm.vision_encoder import prepare_multi_scale_images

LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"
OUTPUT_DIR = "D:/Geocomp/output/geovlm_checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights
W_TASK = 1.0
W_DISTILL = 0.3
W_FEATURE = 0.1
W_SENSOR = 0.3
W_CONSTRAINT = 0.2
W_REG = 0.01


class TeacherLabelDataset(Dataset):
    """Loads teacher-labeled images with hard + soft labels."""

    FIELD_KEYS = [
        "climate_zone_pred", "terrain_type_pred", "vegetation_zone_pred",
        "urbanization_pred", "architecture_style_pred", "pavement_type_pred",
        "language_script_pred",
        # New spatial-narrowing fields
        "soil_color_pred", "sky_quality_pred", "mountain_rock_type_pred",
        "tree_species_pred", "building_height_pred", "water_type_pred",
        "landform_detail_pred", "scene_type_pred",
    ]
    FIELD_VOCABS = {
        "climate_zone_pred": ["tropical", "subtropical", "temperate", "arid", "alpine", "boreal"],
        "terrain_type_pred": ["urban_flat", "farmland_plain", "rolling_hills", "sharp_mountains",
                              "karst_peaks", "sandstone_pillars", "desert_dunes",
                              "grassland_steppe", "plateau"],
        "vegetation_zone_pred": ["tropical_rainforest", "broadleaf_evergreen",
                                 "broadleaf_deciduous", "conifer_forest", "mixed_forest",
                                 "alpine_meadow", "desert_scrub", "grassland",
                                 "bamboo_forest", "cropland", "sparse"],
        "urbanization_pred": ["metropolis", "medium_city", "small_town",
                              "village", "rural", "wilderness"],
        "architecture_style_pred": ["modern_glass", "modern_residential", "old_residential",
                                    "hui_style", "tibetan_stone", "courtyard", "arcade",
                                    "stilt_house", "tulou", "shikumen", "traditional_official",
                                    "soviet_industrial", "none_visible"],
        "pavement_type_pred": ["red_brick_tiles", "grey_concrete", "asphalt",
                               "natural", "not_visible"],
        "language_script_pred": ["simplified_chinese", "traditional_chinese", "tibetan",
                                 "uyghur_arabic", "mongolian", "bilingual_cn_en", "none_visible"],
        "soil_color_pred": ["red", "yellow", "brown", "black", "grey", "loess_yellow", "white"],
        "sky_quality_pred": ["clear_blue", "grey_hazy", "thick_fog", "dusty_yellow"],
        "mountain_rock_type_pred": ["granite_spheroidal", "quartz_sandstone_pillars",
                                    "limestone_karst", "red_sandstone_danxia",
                                    "snow_peaks_glaciers", "alpine_lakes", "none_visible"],
        "tree_species_pred": ["banyan", "camphor", "plane_tree", "gingko", "poplar",
                              "coconut_palm", "chinese_red_pine", "bamboo", "none_visible"],
        "building_height_pred": ["super_tall", "high_rise", "mid_rise", "low_rise", "none_visible"],
        "water_type_pred": ["ocean", "lake", "glacier", "waterfall", "river", "stream", "none_visible"],
        "landform_detail_pred": ["limestone_tower_karst_scattered", "limestone_tower_karst_dense",
                                 "granite_spheroidal_rounded", "granite_spheroidal_sharp",
                                 "quartz_sandstone_pillars", "red_sandstone_danxia",
                                 "snow_peaks_glaciers", "alpine_meadow", "plateau_plain",
                                 "river_canyon", "granite_boulder_field", "coastal_shore",
                                 "urban_flat", "none_visible"],
        "scene_type_pred": ["desert", "grassland", "coastal", "river_valley", "city_street",
                            "mountain_trail_karst", "mountain_trail_granite", "mountain_trail_snow",
                            "rural_village", "tourist_scenic", "wilderness_forest", "none_visible"],
    }

    def __init__(self, labels_file: str, difficulty: str = "all"):
        self.samples = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "hard_labels" in rec:
                        self.samples.append(rec)

        # Difficulty filter for progressive curriculum
        # With 16 fields and 3 versions, most samples have 4-7 split fields
        if difficulty == "easy":
            self.samples = [s for s in self.samples
                          if sum(1 for v in s.get("vote_quality", {}).values() if v == "split") <= 4]
        elif difficulty == "hard":
            self.samples = [s for s in self.samples
                          if sum(1 for v in s.get("vote_quality", {}).values() if v == "split") >= 6]

        print(f"Dataset ({difficulty}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img = Image.open(rec["image_path"]).convert("RGB")
        scales = prepare_multi_scale_images(img)

        sensor = torch.tensor([
            rec.get("sensor_elevation", 500) or 500,
            rec.get("sensor_temperature", 20) or 20,
            rec.get("sensor_humidity", 60) or 60,
        ], dtype=torch.float32)

        # Hard labels -> class indices; UNKNOWN fields get mask=0
        hard_indices = {}
        field_mask = {}  # 1.0 = valid, 0.0 = UNKNOWN (skip in loss)
        for field in self.FIELD_KEYS:
            value = rec["hard_labels"].get(field, "UNKNOWN")
            vocab = self.FIELD_VOCABS[field]
            if value in vocab:
                hard_indices[field] = vocab.index(value)
                field_mask[field] = 1.0
            else:
                hard_indices[field] = 0  # placeholder, masked out
                field_mask[field] = 0.0

        # Soft labels -> probability distributions
        soft_probs = {}
        for field in self.FIELD_KEYS:
            vocab = self.FIELD_VOCABS[field]
            dist = torch.zeros(len(vocab))
            soft_dict = rec.get("soft_labels", {}).get(field, {})
            for val, prob in soft_dict.items():
                if val in vocab:
                    dist[vocab.index(val)] = prob
            if dist.sum() < 1e-8:
                hard_val = rec["hard_labels"].get(field, vocab[0])
                if hard_val in vocab:
                    dist[vocab.index(hard_val)] = 1.0
                else:
                    dist[0] = 1.0
            soft_probs[field] = dist / dist.sum()

        # Elevation target (normalized to [0, 1])
        elev_est = rec["hard_labels"].get("elevation_estimate_m", 500)
        if isinstance(elev_est, list):
            elev_target = torch.tensor(elev_est, dtype=torch.float32) / 5000.0
        else:
            mid = (elev_est if isinstance(elev_est, (int, float)) else 500) / 5000.0
            elev_target = torch.tensor([mid * 0.85, mid * 1.15], dtype=torch.float32)

        # Training weight (lower for split-quality labels)
        quality = rec.get("vote_quality", {})
        n_split = sum(1 for v in quality.values() if v == "split")
        train_weight = 0.7 if n_split > 0 else 1.0

        return {
            "scales": scales,
            "sensor": sensor,
            "hard_indices": hard_indices,
            "soft_probs": soft_probs,
            "field_mask": field_mask,
            "elev_target": elev_target,
            "train_weight": train_weight,
        }


def collate_fn(batch):
    """Custom collate: scales are lists of PIL images, can't stack as tensors."""
    scales_batch = [[s[i] for s in [b["scales"] for b in batch]] for i in range(3)]
    return {
        "scales": scales_batch,  # 3 lists of B PIL Images
        "sensor": torch.stack([b["sensor"] for b in batch]),
        "hard_indices": {k: torch.tensor([b["hard_indices"][k] for b in batch])
                        for k in batch[0]["hard_indices"]},
        "soft_probs": {k: torch.stack([b["soft_probs"][k] for b in batch])
                      for k in batch[0]["soft_probs"]},
        "field_mask": {k: torch.tensor([b["field_mask"][k] for b in batch])
                      for k in batch[0]["field_mask"]},
        "elev_target": torch.stack([b["elev_target"] for b in batch]),
        "train_weight": torch.tensor([b["train_weight"] for b in batch]),
    }


def compute_loss(model, batch, stage: int) -> tuple:
    """Compute total loss for a batch. Stage 1/2/3 controls which losses apply."""
    sensor = batch["sensor"].to(DEVICE)
    output = model.forward(batch["scales"], sensor)
    losses = {}

    field_map = {
        "climate_zone": "climate_zone_pred",
        "terrain_type": "terrain_type_pred",
        "vegetation_zone": "vegetation_zone_pred",
        "urbanization": "urbanization_pred",
        "architecture_style": "architecture_style_pred",
        "pavement_type": "pavement_type_pred",
        "language_script": "language_script_pred",
        "soil_color": "soil_color_pred",
        "sky_quality": "sky_quality_pred",
        "mountain_rock_type": "mountain_rock_type_pred",
        "tree_species": "tree_species_pred",
        "building_height": "building_height_pred",
        "water_type": "water_type_pred",
        "landform_detail": "landform_detail_pred",
        "scene_type": "scene_type_pred",
    }

    # L_task: Cross-entropy per field (hard labels), mask UNKNOWN
    task_loss = 0.0
    for geo_name, ds_name in field_map.items():
        logits = output["raw"]["logits"][geo_name]
        target = batch["hard_indices"][ds_name].to(DEVICE)
        mask = batch["field_mask"][ds_name].to(DEVICE)
        weight = batch["train_weight"].to(DEVICE)
        ce = F.cross_entropy(logits, target, reduction="none")
        task_loss += (ce * mask * weight).sum() / (mask.sum() + 1e-8)
    losses["task"] = task_loss

    # L_distill: KL divergence (soft labels from teacher), mask UNKNOWN
    distill_loss = 0.0
    for geo_name, ds_name in field_map.items():
        log_probs = F.log_softmax(output["raw"]["logits"][geo_name], dim=-1)
        soft_target = batch["soft_probs"][ds_name].to(DEVICE)
        mask = batch["field_mask"][ds_name].to(DEVICE)
        kl = F.kl_div(log_probs, soft_target, reduction="none").sum(dim=-1)
        distill_loss += (kl * mask).sum() / (mask.sum() + 1e-8)
    losses["distill"] = distill_loss

    # L_sensor: elevation from sensor data directly (no model head)
    # Elevation is sourced from sensor input, not model prediction
    sensor_loss = torch.tensor(0.0, device=DEVICE)
    losses["sensor"] = sensor_loss

    # L_constraint: field consistency penalty
    consistency = output["field_consistency"]  # [B, 1]
    constraint_loss = (1.0 - consistency).mean()
    losses["constraint"] = constraint_loss

    # Total weighted loss
    total = W_TASK * task_loss + W_DISTILL * distill_loss
    total += W_CONSTRAINT * constraint_loss

    losses["total"] = total
    return total, losses


def train_stage(model, dataloader, optimizer, stage: int, epochs: int, label: str = ""):
    """Train one stage."""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            loss, losses = compute_loss(model, batch, stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"  {label} E{epoch+1} B{i}: total={loss.item():.4f} "
                      f"task={losses['task'].item():.4f} distill={losses['distill'].item():.4f} "
                      f"sensor={losses['sensor'].item():.4f} const={losses['constraint'].item():.4f}",
                      flush=True)

        avg = epoch_loss / len(dataloader)
        print(f"  {label} Epoch {epoch+1}/{epochs} avg loss: {avg:.4f}", flush=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Stage 1: Representation Alignment ----
    print("=" * 60)
    print("Stage 1: Representation Alignment (freeze ViT)")
    print("=" * 60)

    config = GeoVLMConfig(hidden_dim=512)
    model = GeoVLM(config).to(DEVICE)

    # Freeze ViT
    for param in model.vision_encoder.vit.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT frozen. Trainable: {trainable:,}")

    ds1 = TeacherLabelDataset(LABELS_FILE, difficulty="easy")
    dl1 = DataLoader(ds1, batch_size=2, shuffle=True, collate_fn=collate_fn)

    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-2,
    )
    train_stage(model, dl1, opt1, stage=1, epochs=5, label="S1")

    torch.save({"model": model.state_dict(), "stage": 1},
               os.path.join(OUTPUT_DIR, "geovlm_stage1.pt"))
    print("Stage 1 complete.\n")

    # ---- Stage 2: Full Distillation ----
    print("=" * 60)
    print("Stage 2: Full Distillation (unfreeze ViT last 4 layers)")
    print("=" * 60)

    # Unfreeze ViT last 4 layers
    vit_layers = list(model.vision_encoder.vit.encoder.layer.children())
    for layer in vit_layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT last 4 layers unfrozen. Trainable: {trainable:,}")

    # Progressive difficulty
    for sub_stage, (difficulty, epochs) in enumerate([("easy", 3), ("all", 3), ("all", 4)]):
        print(f"\n  Stage 2.{sub_stage+1}: difficulty={difficulty}")
        ds2 = TeacherLabelDataset(LABELS_FILE, difficulty=difficulty)
        dl2 = DataLoader(ds2, batch_size=2, shuffle=True, collate_fn=collate_fn)

        opt2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5, weight_decay=1e-2,
        )
        train_stage(model, dl2, opt2, stage=2, epochs=epochs, label=f"S2.{sub_stage+1}")

    torch.save({"model": model.state_dict(), "stage": 2},
               os.path.join(OUTPUT_DIR, "geovlm_stage2.pt"))
    print("Stage 2 complete.\n")

    # ---- Stage 3: Sensor Grounding ----
    print("=" * 60)
    print("Stage 3: Sensor Grounding (noise augmentation)")
    print("=" * 60)

    model.config.sensor_noise_std = 0.15  # enable noise injection
    ds3 = TeacherLabelDataset(LABELS_FILE, difficulty="all")
    dl3 = DataLoader(ds3, batch_size=2, shuffle=True, collate_fn=collate_fn)

    opt3 = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    train_stage(model, dl3, opt3, stage=3, epochs=5, label="S3")

    torch.save({"model": model.state_dict(), "stage": 3, "config": config},
               os.path.join(OUTPUT_DIR, "geovlm_final.pt"))
    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}/geovlm_final.pt")


if __name__ == "__main__":
    main()
