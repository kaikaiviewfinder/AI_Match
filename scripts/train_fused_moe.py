"""3-phase FusedMoE training pipeline.

Trains a FusedMoERegressor that combines visual features + VLM hidden states.

Phase 1 - Expert pretraining:
    Each expert trains on its scene data with fused features (visual + vlm_fused → fusion_proj → expert).

Phase 2 - Router training:
    Freeze experts, train router. Full dual-input forward pass.

Phase 3 - Joint fine-tuning:
    Unfreeze all, low-lr polish.

Usage:
    python scripts/train_fused_moe.py \
        --features output/regression/fused_features.pt \
        --output output/fused_moe/ \
        --fusion-mode concat
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from regression.fused_moe import FusedMoERegressor
from regression.coord_regressor import HaversineLoss, haversine_distance_km, create_expert
from regression.moe import MoEHaversineLoss
from regression.moe_config import (
    SceneType, NUM_EXPERTS, SCENE_TYPE_NAMES,
    SPARSE_EXPERT_CONFIG, DEFAULT_EXPERT_CONFIG, SPARSE_EXPERT_THRESHOLD,
    ROUTER_HIDDEN_DIMS,
    DEFAULT_EPOCHS, ROUTER_EPOCHS,
    PATIENCE, BATCH_SIZE, WEIGHT_DECAY, GRAD_CLIP_NORM,
    JOINT_FINETUNE_LR, JOINT_FINETUNE_EPOCHS,
    SCENE_CLASS_WEIGHT,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data loading ─────────────────────────────────────────────────────────

def load_fused_data(features_file: str):
    """Load fused_features.pt and return tensors."""
    print("Loading fused features...")
    data = torch.load(features_file, map_location="cpu", weights_only=True)

    visual = data["visual"]          # (N, 1536)
    vlm_fused = data["vlm_fused"]    # (N, 1536)
    coords = data["coords"]          # (N, 2) normalized
    scene_types = data.get("scene_types", torch.full((visual.shape[0],), 3, dtype=torch.long))

    print(f"  Visual:      {visual.shape}")
    print(f"  VLM fused:   {vlm_fused.shape}")
    print(f"  Coords:      {coords.shape}")
    print(f"  Scene types: {scene_types.shape}")

    return visual, vlm_fused, coords, scene_types


def split_by_scene(visual, vlm_fused, coords, scene_types):
    """Split data by scene type for expert pretraining."""
    splits = {}
    for i in range(NUM_EXPERTS):
        mask = scene_types == i
        splits[i] = {
            "visual": visual[mask],
            "vlm_fused": vlm_fused[mask],
            "coords": coords[mask],
            "n": mask.sum().item(),
        }
    return splits


def get_expert_configs_from_splits(splits):
    """Determine expert configs based on sample counts."""
    configs = []
    for i in range(NUM_EXPERTS):
        n = splits[i]["n"]
        if n < SPARSE_EXPERT_THRESHOLD:
            configs.append(dict(SPARSE_EXPERT_CONFIG))
        else:
            configs.append(dict(DEFAULT_EXPERT_CONFIG))
    return configs


# ── Phase 1: Expert pretraining ─────────────────────────────────────────

def train_expert_with_fusion(
    expert_idx: int,
    scene_data: dict,
    fusion_proj: torch.nn.Linear,
    config: dict,
    pretrained_path: Optional[str],
    output_dir: str,
):
    """Pretrain a single expert on fused features (visual + vlm_fused).

    fusion_proj projects concat(visual, vlm_fused) from 3072 → 1536.
    The expert itself is a standard CoordRegressor trained on the projected features.
    """
    name = SCENE_TYPE_NAMES.get(expert_idx, f"expert{expert_idx}")
    n_samples = scene_data["n"]
    print(f"\n{'='*60}")
    print(f"Phase 1: Expert {expert_idx} ({name}) — {n_samples} samples")

    if n_samples == 0:
        print(f"  No samples — creating from pretrained or random")
        model = create_expert(config=config, pretrained_path=pretrained_path, device=DEVICE)
        torch.save(model.state_dict(), os.path.join(output_dir, f"expert_{expert_idx}.pt"))
        return model

    # Build fused features by projecting visual + vlm through fusion_proj
    visual = scene_data["visual"].to(DEVICE)
    vlm = scene_data["vlm_fused"].to(DEVICE)
    coords = scene_data["coords"].to(DEVICE)

    with torch.no_grad():
        fused_features = fusion_proj(torch.cat([visual, vlm], dim=-1))  # (n, 1536)

    # Train/val split
    n_val = max(1, int(n_samples * 0.1))
    perm = torch.randperm(n_samples)
    train_feat = fused_features[perm[n_val:]].to(DEVICE).float()
    val_feat = fused_features[perm[:n_val]].to(DEVICE).float()
    train_coords = coords[perm[n_val:]].to(DEVICE).float()
    val_coords = coords[perm[:n_val]].to(DEVICE).float()
    print(f"  Train: {train_feat.shape[0]}, Val: {val_feat.shape[0]}")

    model = create_expert(config=config, pretrained_path=pretrained_path, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DEFAULT_EPOCHS)
    criterion = HaversineLoss()

    best_loss = float("inf")
    no_improve = 0
    patience = PATIENCE if n_samples > 100 else max(PATIENCE, DEFAULT_EPOCHS)

    for epoch in range(1, DEFAULT_EPOCHS + 1):
        model.train()
        indices = torch.randperm(train_feat.shape[0])
        total_loss = 0.0
        n_batches = 0
        for i in range(0, train_feat.shape[0], BATCH_SIZE):
            idx = indices[i:i + BATCH_SIZE]
            feat_b = train_feat[idx]
            target_b = train_coords[idx]
            optimizer.zero_grad()
            pred = model(feat_b)
            loss = criterion(pred, target_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            pred_v = model(val_feat)
            val_loss = criterion(pred_v, val_coords).item()

        status = "*" if val_loss < best_loss else " "
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"expert_{expert_idx}.pt"))
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or status == "*":
            print(f"  [{status}] E{epoch:3d} | train={total_loss/max(n_batches,1):.1f}km "
                  f"val={val_loss:.1f}km best={best_loss:.1f}km")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (best={best_loss:.1f}km)")
            break

    # Load best
    model.load_state_dict(torch.load(os.path.join(output_dir, f"expert_{expert_idx}.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


# ── Phase 2: Router training ────────────────────────────────────────────

def train_router_phase(
    model: FusedMoERegressor,
    visual: torch.Tensor,
    vlm_fused: torch.Tensor,
    coords: torch.Tensor,
    scene_types: torch.Tensor,
    output_dir: str,
):
    """Train the router with frozen experts on full dataset."""
    print(f"\n{'='*60}")
    print(f"Phase 2: Router training — {visual.shape[0]} samples")
    print(f"  Fusion mode: {model.fusion_mode}")

    model.freeze_experts()
    model.unfreeze_router()
    if model.fusion_mode == "concat":
        # fusion_proj should also stay trainable
        for p in model.fusion_proj.parameters():
            p.requires_grad = True

    # Train/val split
    n = visual.shape[0]
    n_val = max(1, int(n * 0.1))
    perm = torch.randperm(n)
    tr_v, val_v = visual[perm[n_val:]], visual[perm[:n_val]]
    tr_h, val_h = vlm_fused[perm[n_val:]], vlm_fused[perm[:n_val]]
    tr_c, val_c = coords[perm[n_val:]], coords[perm[:n_val]]
    tr_s, val_s = scene_types[perm[n_val:]], scene_types[perm[:n_val]]

    # Build scene label tensor (mask: only use labeled samples)
    scene_labels_tensor = tr_s.clone()  # Copy from scene_types

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ROUTER_EPOCHS)
    criterion = MoEHaversineLoss(scene_class_weight=SCENE_CLASS_WEIGHT)
    haversine = HaversineLoss()

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, ROUTER_EPOCHS + 1):
        model.train()
        indices = torch.randperm(tr_v.shape[0])
        total_loss = 0.0
        n_batches = 0

        for i in range(0, tr_v.shape[0], BATCH_SIZE):
            idx = indices[i:i + BATCH_SIZE]
            vis_b = tr_v[idx].to(DEVICE).float()
            vlm_b = tr_h[idx].to(DEVICE).float()
            tgt_b = tr_c[idx].to(DEVICE).float()
            lbl_b = scene_labels_tensor[idx].to(DEVICE)

            optimizer.zero_grad()
            fused, weights, _, logits = model.forward(
                vis_b, vlm_b, return_weights=True, return_logits=True
            )

            # Build mask for labeled samples in this batch
            mask = lbl_b >= 0
            loss, comps = criterion(
                fused, tgt_b, weights,
                router_logits=logits,
                scene_labels=lbl_b if mask.any() else None,
                label_mask=mask if mask.any() else None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP_NORM
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Val
        model.eval()
        with torch.no_grad():
            vv = val_v.to(DEVICE).float()
            vh = val_h.to(DEVICE).float()
            vc = val_c.to(DEVICE).float()
            fused_v, weights_v, _ = model.forward(vv, vh, return_weights=True)
            val_km = haversine(fused_v, vc).item()

        status = "*" if val_km < best_loss else " "
        if val_km < best_loss:
            best_loss = val_km
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "fused_moe_router.pt"))
        else:
            no_improve += 1

        ent = comps.get("entropy", 0)
        print(f"  [{status}] E{epoch:3d} | train={total_loss/max(n_batches,1):.1f} "
              f"val={val_km:.1f}km | H={ent:.3f} | best={best_loss:.1f}km")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # Reload best
    model.load_state_dict(torch.load(os.path.join(output_dir, "fused_moe_router.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


# ── Phase 3: Joint fine-tuning ──────────────────────────────────────────

def train_joint_finetune(
    model: FusedMoERegressor,
    visual: torch.Tensor,
    vlm_fused: torch.Tensor,
    coords: torch.Tensor,
    output_dir: str,
):
    """Joint fine-tune all parameters at low learning rate."""
    print(f"\n{'='*60}")
    print(f"Phase 3: Joint fine-tuning — {visual.shape[0]} samples, lr={JOINT_FINETUNE_LR}")

    model.unfreeze_experts()
    model.train()

    n_val = max(1, int(visual.shape[0] * 0.1))
    perm = torch.randperm(visual.shape[0])
    tr_v = visual[perm[n_val:]].to(DEVICE).float()
    tr_h = vlm_fused[perm[n_val:]].to(DEVICE).float()
    tr_c = coords[perm[n_val:]].to(DEVICE).float()
    val_v = visual[perm[:n_val]].to(DEVICE).float()
    val_h = vlm_fused[perm[:n_val]].to(DEVICE).float()
    val_c = coords[perm[:n_val]].to(DEVICE).float()

    # Move everything to GPU once (should fit for ~7000 × 1536)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=JOINT_FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=JOINT_FINETUNE_EPOCHS)
    criterion = MoEHaversineLoss(entropy_weight=0.01, load_balance_weight=0.001)
    haversine = HaversineLoss()

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, JOINT_FINETUNE_EPOCHS + 1):
        model.train()
        indices = torch.randperm(tr_v.shape[0])
        total_loss = 0.0
        n_batches = 0
        for i in range(0, tr_v.shape[0], BATCH_SIZE):
            idx = indices[i:i + BATCH_SIZE]
            vis_b = tr_v[idx]
            vlm_b = tr_h[idx]
            tgt_b = tr_c[idx]

            optimizer.zero_grad()
            fused, weights, _ = model.forward(vis_b, vlm_b, return_weights=True)
            loss, _ = criterion(fused, tgt_b, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            fused_v, _, _ = model.forward(val_v, val_h, return_weights=True)
            val_km = haversine(fused_v, val_c).item()

        status = "*" if val_km < best_loss else " "
        if val_km < best_loss:
            best_loss = val_km
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "fused_moe_final.pt"))
        else:
            no_improve += 1

        print(f"  [{status}] E{epoch:3d} | train={total_loss/max(n_batches,1):.1f} "
              f"val={val_km:.1f}km | best={best_loss:.1f}km")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(os.path.join(output_dir, "fused_moe_final.pt"),
                                     map_location=DEVICE, weights_only=True))
    return model


# ── Main pipeline ───────────────────────────────────────────────────────

def train_fused_moe(
    features_file: str = "output/regression/fused_features.pt",
    pretrained_path: str = "output/regression/best_model.pt",
    output_dir: str = "output/fused_moe",
    fusion_mode: str = "concat",
    phases: str = "1,2,3",
):
    """Full 3-phase FusedMoE training pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    visual, vlm_fused, coords, scene_types = load_fused_data(features_file)

    # Split by scene
    splits = split_by_scene(visual, vlm_fused, coords, scene_types)
    for i in range(NUM_EXPERTS):
        print(f"  Scene {i} ({SCENE_TYPE_NAMES[i]}): {splits[i]['n']} samples")

    expert_configs = get_expert_configs_from_splits(splits)
    print(f"  Expert configs: {[cfg['hidden_dims'] for cfg in expert_configs]}")

    # Create FusedMoE model
    model = FusedMoERegressor(
        fusion_mode=fusion_mode,
        input_dim=1536,
        vlm_hidden_dim=1536,
        expert_configs=expert_configs,
    )
    print(f"\nModel: FusedMoE (fusion_mode={fusion_mode})")
    print(f"  Fusion params: {sum(p.numel() for p in model.fusion_proj.parameters()):,}")

    phases_list = [p.strip() for p in phases.split(",")]

    # ── Phase 1: Expert pretraining ──────────────────────────────────
    if "1" in phases_list:
        print(f"\n{'#'*60}")
        print("# Phase 1: Expert pretraining (fused features)")
        print(f"{'#'*60}")

        # Move fusion_proj to device for feature projection
        model.fusion_proj.to(DEVICE)
        model.fusion_proj.eval()  # Frozen during expert pretraining

        for i in range(NUM_EXPERTS):
            if splits[i]["n"] == 0:
                print(f"  Expert {i}: no data, skipping")
                continue
            expert = train_expert_with_fusion(
                expert_idx=i,
                scene_data=splits[i],
                fusion_proj=model.fusion_proj,
                config=expert_configs[i],
                pretrained_path=pretrained_path,
                output_dir=output_dir,
            )
            model.load_expert_state(i, expert.state_dict())

        torch.save(model.state_dict(), os.path.join(output_dir, "fused_moe_experts.pt"))
        print(f"\n  Experts saved to {output_dir}/fused_moe_experts.pt")

    # ── Phase 2: Router training ─────────────────────────────────────
    if "2" in phases_list:
        print(f"\n{'#'*60}")
        print("# Phase 2: Router training")
        print(f"{'#'*60}")

        # Load expert states if skipping Phase 1
        if "1" not in phases_list:
            state = torch.load(os.path.join(output_dir, "fused_moe_experts.pt"),
                              map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=False)

        model.to(DEVICE)
        model = train_router_phase(
            model, visual, vlm_fused, coords, scene_types, output_dir
        )

    # ── Phase 3: Joint fine-tuning ───────────────────────────────────
    if "3" in phases_list:
        print(f"\n{'#'*60}")
        print("# Phase 3: Joint fine-tuning")
        print(f"{'#'*60}")

        # Load router state if skipping Phase 2
        if "2" not in phases_list:
            state = torch.load(os.path.join(output_dir, "fused_moe_router.pt"),
                              map_location=DEVICE, weights_only=True)
            model.load_state_dict(state, strict=False)

        model = train_joint_finetune(
            model, visual, vlm_fused, coords, output_dir
        )

    print(f"\n{'='*60}")
    print(f"Training complete! Final model: {output_dir}/fused_moe_final.pt")
    print(f"{'='*60}")
    return model


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-phase FusedMoE training")
    parser.add_argument("--features", type=str, default="output/regression/fused_features.pt")
    parser.add_argument("--pretrained", type=str, default="output/regression/best_model.pt")
    parser.add_argument("--output", type=str, default="output/fused_moe")
    parser.add_argument("--fusion-mode", type=str, default="concat", choices=["concat", "gated"])
    parser.add_argument("--phases", type=str, default="1,2,3",
                        help="Comma-separated phases: 1=expert pretrain, 2=router, 3=joint")
    args = parser.parse_args()

    train_fused_moe(
        features_file=args.features,
        pretrained_path=args.pretrained,
        output_dir=args.output,
        fusion_mode=args.fusion_mode,
        phases=args.phases,
    )


if __name__ == "__main__":
    main()
