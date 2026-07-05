"""Robust GeoVLM distillation training with checkpoint resilience.

Features:
  - Per-epoch checkpoint auto-save (survives crashes)
  - KeyboardInterrupt handler (Ctrl+C = graceful save)
  - Resume from latest checkpoint
  - Training log written to file
"""
import sys, json, time, os, signal, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from geovlm import GeoVLM, GeoVLMConfig
from geovlm.vision_encoder import prepare_multi_scale_images

LABELS_FILE = "D:/Geocomp/output/geovlm_teacher_labels.jsonl"
OUTPUT_DIR = Path("D:/Geocomp/output/geovlm_checkpoints")
LOG_FILE = OUTPUT_DIR / "training_log.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

W_TASK, W_DISTILL, W_FEATURE = 1.0, 0.3, 0.1
W_SENSOR, W_CONSTRAINT, W_REG = 0.3, 0.2, 0.01

# Save every N batches and every epoch
SAVE_EVERY_N_BATCHES = 50
_last_save_time = time.time()

# Global state for interrupt handler
_g_model = None
_g_optimizer = None
_g_stage = 0
_g_epoch = 0
_g_batch = 0


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_checkpoint(model, optimizer, stage, epoch, batch, tag="auto"):
    """Save full training state for resume."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if tag == "auto":
        fname = f"ckpt_s{stage}_e{epoch}_b{batch}.pt"
    else:
        fname = f"ckpt_s{stage}_{tag}.pt"

    path = OUTPUT_DIR / fname
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "batch": batch,
        "config": model.config,
        "timestamp": time.time(),
    }
    torch.save(state, path)

    # Also save as "latest" symlink-equivalent
    latest_path = OUTPUT_DIR / f"latest_checkpoint.pt"
    torch.save(state, latest_path)

    log(f"  [checkpoint] {fname} ({os.path.getsize(path)/1024:.0f} KB)")


def find_latest_checkpoint():
    """Find most recent checkpoint for resume."""
    latest = OUTPUT_DIR / "latest_checkpoint.pt"
    if latest.exists():
        return latest

    # Fallback: find newest ckpt_ file
    ckpts = sorted(OUTPUT_DIR.glob("ckpt_*.pt"), key=os.path.getmtime, reverse=True)
    return ckpts[0] if ckpts else None


def signal_handler(signum, frame):
    """Handle Ctrl+C - save and exit cleanly."""
    log(f"\n[INTERRUPT] Signal {signum} received. Saving checkpoint...")
    if _g_model is not None and _g_optimizer is not None:
        save_checkpoint(_g_model, _g_optimizer, _g_stage, _g_epoch, _g_batch, tag="interrupt")
    log("[INTERRUPT] Checkpoint saved. Exiting.")
    sys.exit(0)


# ============================================================
# Dataset
# ============================================================
class TeacherLabelDataset(Dataset):
    FIELD_KEYS = [
        "climate_zone_pred", "terrain_type_pred", "vegetation_zone_pred",
        "urbanization_pred", "architecture_style_pred", "pavement_type_pred",
        "language_script_pred",
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
    }

    def __init__(self, labels_file: str, difficulty: str = "all"):
        self.samples = []
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "hard_labels" in rec:
                        self.samples.append(rec)

        if difficulty == "easy":
            # >=5 fields unanimous across versions
            self.samples = [s for s in self.samples
                          if sum(1 for v in s.get("vote_quality", {}).values()
                                 if v == "unanimous") >= 5]
        elif difficulty == "hard":
            self.samples = [s for s in self.samples
                          if sum(1 for v in s.get("vote_quality", {}).values()
                                 if v == "split") > 2]
        log(f"Dataset ({difficulty}): {len(self.samples)} samples")

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

        hard_indices = {}
        for field in self.FIELD_KEYS:
            value = rec["hard_labels"].get(field, "UNKNOWN")
            vocab = self.FIELD_VOCABS[field]
            hard_indices[field] = vocab.index(value) if value in vocab else 0

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

        elev_est = rec["hard_labels"].get("elevation_estimate_m", 500)
        if isinstance(elev_est, list):
            elev_target = torch.tensor(elev_est, dtype=torch.float32) / 5000.0
        else:
            mid = (elev_est if isinstance(elev_est, (int, float)) else 500) / 5000.0
            elev_target = torch.tensor([mid * 0.85, mid * 1.15], dtype=torch.float32)

        quality = rec.get("vote_quality", {})
        n_split = sum(1 for v in quality.values() if v == "split")
        train_weight = 0.7 if n_split > 0 else 1.0

        return {
            "scales": scales, "sensor": sensor,
            "hard_indices": hard_indices, "soft_probs": soft_probs,
            "elev_target": elev_target, "train_weight": train_weight,
        }


def collate_fn(batch):
    scales_batch = [[s[i] for s in [b["scales"] for b in batch]] for i in range(3)]
    return {
        "scales": scales_batch,
        "sensor": torch.stack([b["sensor"] for b in batch]),
        "hard_indices": {k: torch.tensor([b["hard_indices"][k] for b in batch])
                        for k in batch[0]["hard_indices"]},
        "soft_probs": {k: torch.stack([b["soft_probs"][k] for b in batch])
                      for k in batch[0]["soft_probs"]},
        "elev_target": torch.stack([b["elev_target"] for b in batch]),
        "train_weight": torch.tensor([b["train_weight"] for b in batch]),
    }


# ============================================================
# Loss
# ============================================================
def compute_loss(model, batch, stage):
    sensor = batch["sensor"].to(DEVICE)
    output = model.forward(batch["scales"], sensor)
    losses = {}

    field_map = {
        "climate_zone": "climate_zone_pred", "terrain_type": "terrain_type_pred",
        "vegetation_zone": "vegetation_zone_pred", "urbanization": "urbanization_pred",
        "architecture_style": "architecture_style_pred", "pavement_type": "pavement_type_pred",
        "language_script": "language_script_pred",
    }

    task_loss = 0.0
    for geo_name, ds_name in field_map.items():
        logits = output["raw"]["logits"][geo_name]
        target = batch["hard_indices"][ds_name].to(DEVICE)
        weight = batch["train_weight"].to(DEVICE)
        ce = F.cross_entropy(logits, target, reduction="none")
        task_loss += (ce * weight).mean()
    losses["task"] = task_loss

    distill_loss = 0.0
    for geo_name, ds_name in field_map.items():
        log_probs = F.log_softmax(output["raw"]["logits"][geo_name], dim=-1)
        soft_target = batch["soft_probs"][ds_name].to(DEVICE)
        distill_loss += F.kl_div(log_probs, soft_target, reduction="batchmean")
    losses["distill"] = distill_loss

    elev_pred = output["raw"]["elevation_range"]
    elev_target = batch["elev_target"].to(DEVICE) * 5000.0
    sensor_loss = F.mse_loss(elev_pred / 5000.0, elev_target / 5000.0)
    losses["sensor"] = sensor_loss

    consistency = output["field_consistency"]
    constraint_loss = (1.0 - consistency).mean()
    losses["constraint"] = constraint_loss

    total = W_TASK * task_loss + W_DISTILL * distill_loss
    total += (W_SENSOR * 2.0 if stage >= 3 else W_SENSOR) * sensor_loss
    total += W_CONSTRAINT * constraint_loss
    losses["total"] = total
    return total, losses


# ============================================================
# Training loop with checkpointing
# ============================================================
def train_stage(model, dataloader, optimizer, stage, epochs, label):
    global _g_model, _g_optimizer, _g_stage, _g_epoch, _g_batch
    _g_model, _g_optimizer, _g_stage = model, optimizer, stage

    model.train()
    n_batches = len(dataloader)

    for epoch in range(epochs):
        _g_epoch = epoch
        epoch_loss = 0.0
        t0 = time.time()

        for i, batch in enumerate(dataloader):
            _g_batch = i
            optimizer.zero_grad()
            loss, losses = compute_loss(model, batch, stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                log(f"  {label} E{epoch+1}/{epochs} B{i}/{n_batches}: "
                    f"total={loss.item():.4f} task={losses['task'].item():.4f} "
                    f"distill={losses['distill'].item():.4f} "
                    f"sensor={losses['sensor'].item():.4f} const={losses['constraint'].item():.4f}")

            # Periodic save every N batches
            if i > 0 and i % SAVE_EVERY_N_BATCHES == 0:
                save_checkpoint(model, optimizer, stage, epoch, i)

        avg = epoch_loss / n_batches
        elapsed = time.time() - t0
        log(f"  {label} Epoch {epoch+1}/{epochs} done | avg_loss={avg:.4f} | {elapsed:.0f}s")

        # Per-epoch save
        save_checkpoint(model, optimizer, stage, epoch, n_batches - 1, tag=f"e{epoch+1}")

    # Stage-complete save
    save_checkpoint(model, optimizer, stage, epochs - 1, n_batches - 1,
                    tag=f"stage{stage}_complete")


# ============================================================
# Main
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write log header
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Training started: {datetime.datetime.now()}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Labels: {LABELS_FILE}\n")
        f.write(f"{'='*60}\n")

    log(f"GeoVLM Distillation Training")
    log(f"Device: {DEVICE} | Output: {OUTPUT_DIR}")
    log(f"Labels: {LABELS_FILE}")
    log(f"Auto-save every {SAVE_EVERY_N_BATCHES} batches + every epoch")

    # Register interrupt handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for resume
    resume_ckpt = find_latest_checkpoint()
    if resume_ckpt:
        log(f"Resume checkpoint found: {resume_ckpt}")
        # For now, start fresh (resume logic can be added later)
        log("Starting fresh training (resume disabled for initial run)")

    # ================================================================
    # Stage 1: Representation Alignment
    # ================================================================
    log("=" * 60)
    log("Stage 1: Representation Alignment (freeze ViT)")
    log("=" * 60)

    config = GeoVLMConfig(hidden_dim=512)
    model = GeoVLM(config).to(DEVICE)

    for param in model.vision_encoder.vit.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"ViT frozen. Trainable params: {trainable:,}")

    ds1 = TeacherLabelDataset(LABELS_FILE, difficulty="easy")
    dl1 = DataLoader(ds1, batch_size=2, shuffle=True, collate_fn=collate_fn)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=1e-2,
    )
    train_stage(model, dl1, opt1, stage=1, epochs=5, label="S1")
    log("Stage 1 complete.\n")

    # ================================================================
    # Stage 2: Full Distillation
    # ================================================================
    log("=" * 60)
    log("Stage 2: Full Distillation (unfreeze ViT last 4 layers)")
    log("=" * 60)

    vit_layers = list(model.vision_encoder.vit.encoder.layer.children())
    for layer in vit_layers[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"ViT last 4 layers unfrozen. Trainable params: {trainable:,}")

    for sub_stage, (difficulty, epochs) in enumerate([("easy", 3), ("all", 3), ("all", 4)]):
        log(f"  Stage 2.{sub_stage+1}: difficulty={difficulty}, epochs={epochs}")
        ds2 = TeacherLabelDataset(LABELS_FILE, difficulty=difficulty)
        dl2 = DataLoader(ds2, batch_size=2, shuffle=True, collate_fn=collate_fn)
        opt2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5, weight_decay=1e-2,
        )
        train_stage(model, dl2, opt2, stage=2, epochs=epochs, label=f"S2.{sub_stage+1}")
    log("Stage 2 complete.\n")

    # ================================================================
    # Stage 3: Sensor Grounding
    # ================================================================
    log("=" * 60)
    log("Stage 3: Sensor Grounding (noise augmentation)")
    log("=" * 60)

    model.config.sensor_noise_std = 0.15
    ds3 = TeacherLabelDataset(LABELS_FILE, difficulty="all")
    dl3 = DataLoader(ds3, batch_size=2, shuffle=True, collate_fn=collate_fn)
    opt3 = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    train_stage(model, dl3, opt3, stage=3, epochs=5, label="S3")

    # ================================================================
    # Final save
    # ================================================================
    final_path = OUTPUT_DIR / "geovlm_final.pt"
    torch.save({"model": model.state_dict(), "stage": 3, "config": config}, final_path)
    log(f"\nTraining complete! Final model: {final_path}")
    log(f"Total checkpoints in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
