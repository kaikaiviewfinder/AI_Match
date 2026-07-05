"""3-phase MoE training pipeline.

Phase 1 — Expert pretraining:
    Warm-start each expert from best_model.pt, then fine-tune on scene-specific data.
    Small experts (alpine, karst_granite) get sparse configs + augmentation.

Phase 2 — Router training:
    Freeze experts, train RouterMLP to assign best expert per sample.
    Loss = Haversine(fused) + alpha * Entropy + beta * LoadBalance

Phase 3 — Joint fine-tuning (optional):
    Unfreeze all, low-lr polish on full dataset.

Usage:
    python -m src.regression.train_moe \\
        --features output/regression/features.pt \\
        --labels output/regression/scene_labels.json \\
        --pretrained output/regression/best_model.pt \\
        --output output/moe/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from src.regression.coord_regressor import CoordRegressor, HaversineLoss, create_expert
from src.regression.moe import (
    MoERegressor, MoEHaversineLoss, RouterMLP, get_expert_configs,
)
from src.regression.moe_config import (
    SceneType, NUM_EXPERTS, SCENE_TYPE_NAMES,
    SPARSE_EXPERT_CONFIG, DEFAULT_EXPERT_CONFIG, SPARSE_EXPERT_THRESHOLD,
    DEFAULT_EPOCHS, ROUTER_EPOCHS, PATIENCE, BATCH_SIZE, WEIGHT_DECAY, GRAD_CLIP_NORM,
    JOINT_FINETUNE_LR, JOINT_FINETUNE_EPOCHS,
    AUGMENTATION_ENABLED, SCENE_CLASS_WEIGHT,
)


# ── Data splitting utilities ────────────────────────────────────────────

def split_by_scene(
    features: torch.Tensor,
    coords: torch.Tensor,
    labels: dict[int, int],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Split pre-indexed (idx→scene_type) features/coords by scene type.

    Args:
        features: (N, 1536)
        coords: (N, 2) normalized
        labels: {sample_idx: scene_type (0-3)}

    Returns:
        {expert_idx: (features, coords)}
    """
    splits = {i: ([], []) for i in range(NUM_EXPERTS)}
    for idx, scene_type in labels.items():
        if 0 <= scene_type < NUM_EXPERTS and idx < len(features):
            splits[scene_type][0].append(features[idx])
            splits[scene_type][1].append(coords[idx])

    result = {}
    for i in range(NUM_EXPERTS):
        feats = torch.stack(splits[i][0]) if splits[i][0] else torch.empty(0, features.shape[1])
        crds = torch.stack(splits[i][1]) if splits[i][1] else torch.empty(0, coords.shape[1])
        result[i] = (feats, crds)
    return result


def load_scene_labels_json(labels_file: str, data_file: str) -> dict[int, int]:
    """Convert path-based scene_labels.json to index-based {idx: scene_type}.

    Assumes features.pt was computed from data_file in the same order.
    """
    with open(labels_file, "r", encoding="utf-8") as f:
        path_labels = json.load(f)
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    idx_labels = {}
    for idx, path in enumerate(data):
        if path in path_labels:
            idx_labels[idx] = int(path_labels[path])
    return idx_labels


# ── Phase 1: Expert pretraining ─────────────────────────────────────────

class ExpertTrainer:
    """Train a single CoordRegressor expert."""

    def __init__(self, model: CoordRegressor, lr: float, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=DEFAULT_EPOCHS)
        self.criterion = HaversineLoss()

    def train_epoch(self, features: torch.Tensor, coords: torch.Tensor,
                    batch_size: int = BATCH_SIZE) -> float:
        self.model.train()
        n = features.shape[0]
        if n == 0:
            return 0.0
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            feat = features[batch_idx].to(self.device).float()
            target = coords[batch_idx].to(self.device).float()

            self.optimizer.zero_grad()
            pred = self.model(feat)
            loss = self.criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, coords: torch.Tensor) -> dict:
        self.model.eval()
        if features.shape[0] == 0:
            return {"loss_km": float("inf"), "median_km": float("inf")}

        feat = features.to(self.device).float()
        target = coords.to(self.device).float()
        pred = self.model(feat)
        loss = self.criterion(pred, target)

        pred_lat = pred[:, 0] * 90.0
        pred_lng = pred[:, 1] * 180.0
        true_lat = target[:, 0] * 90.0
        true_lng = target[:, 1] * 180.0

        from src.regression.coord_regressor import haversine_distance_km
        dists = haversine_distance_km(pred_lat, pred_lng, true_lat, true_lng)

        return {
            "loss_km": loss.item(),
            "median_km": dists.median().item(),
            "mean_km": dists.mean().item(),
        }


def train_expert_pretrain(
    expert_idx: int,
    scene_features: torch.Tensor,
    scene_coords: torch.Tensor,
    pretrained_path: Optional[str],
    config: dict,
    output_dir: str,
    device: str = "cuda",
    input_dim: int = 1536,
) -> CoordRegressor:
    """Phase 1: pretrain a single expert on its scene data.

    Returns the trained expert model.
    """
    name = SCENE_TYPE_NAMES.get(expert_idx, f"expert{expert_idx}")
    n_samples = scene_features.shape[0]
    print(f"\n{'='*60}")
    print(f"Phase 1: Expert {expert_idx} ({name}) — {n_samples} samples")
    print(f"  Config: hidden_dims={config['hidden_dims']}, dropout={config['dropout']}, lr={config['lr']}")

    if n_samples == 0:
        print(f"  No samples — skipping (will use pretrained or random init)")
        model = create_expert(input_dim=input_dim, config=config, pretrained_path=pretrained_path, device=device)
        torch.save(model.state_dict(), os.path.join(output_dir, f"expert_{expert_idx}.pt"))
        return model

    # Train/val split
    n_val = max(1, int(n_samples * 0.1))
    perm = torch.randperm(n_samples)
    train_feat, val_feat = scene_features[perm[n_val:]], scene_features[perm[:n_val]]
    train_coords, val_coords = scene_coords[perm[n_val:]], scene_coords[perm[:n_val]]
    print(f"  Train: {train_feat.shape[0]}, Val: {val_feat.shape[0]}")

    model = create_expert(input_dim=input_dim, config=config, pretrained_path=pretrained_path, device=device)
    trainer = ExpertTrainer(model, lr=config["lr"], device=device)

    best_loss = float("inf")
    no_improve = 0
    patience = PATIENCE if n_samples > 100 else max(PATIENCE, DEFAULT_EPOCHS)

    for epoch in range(1, DEFAULT_EPOCHS + 1):
        train_loss = trainer.train_epoch(train_feat, train_coords)
        val_metrics = trainer.evaluate(val_feat, val_coords)

        if val_metrics["loss_km"] < best_loss:
            best_loss = val_metrics["loss_km"]
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f"expert_{expert_idx}.pt"))
            status = "*"
        else:
            no_improve += 1
            status = " "

        if epoch % 10 == 0 or epoch == 1 or status == "*":
            print(f"  [{status}] E{epoch:3d} | train={train_loss:.1f}km "
                  f"val={val_metrics['loss_km']:.1f}km median={val_metrics['median_km']:.1f}km")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (best={best_loss:.1f}km)")
            break

    # Reload best
    model.load_state_dict(torch.load(os.path.join(output_dir, f"expert_{expert_idx}.pt"),
                                     map_location=device, weights_only=True))
    return model


# ── Phase 2: Router training ────────────────────────────────────────────

class RouterTrainer:
    """Train the router with frozen experts + optional scene classification loss."""

    def __init__(self, moe: MoERegressor, lr: float = 1e-3, device: str = "cuda",
                 use_scene_class: bool = True):
        self.moe = moe.to(device)
        self.device = device
        self.moe.freeze_experts()
        self.optimizer = torch.optim.AdamW(moe.router.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=ROUTER_EPOCHS)
        self.criterion = MoEHaversineLoss(
            scene_class_weight=SCENE_CLASS_WEIGHT if use_scene_class else 0.0,
        )
        self.use_scene_class = use_scene_class

    def train_epoch(self, features: torch.Tensor, coords: torch.Tensor,
                    scene_labels_tensor: torch.Tensor = None,
                    batch_size: int = BATCH_SIZE) -> tuple[float, dict]:
        self.moe.train()
        n = features.shape[0]
        total_loss = 0.0
        n_batches = 0
        final_components = {}

        # Class-balanced batch sampling (prevents router collapse to majority class)
        if scene_labels_tensor is not None:
            class_indices = {}
            for c in range(NUM_EXPERTS):
                mask = scene_labels_tensor == c
                class_indices[c] = torch.where(mask)[0]

            per_class = max(1, batch_size // NUM_EXPERTS)
            n_batches_total = n // batch_size + 1

            for _ in range(n_batches_total):
                batch_idx = []
                for c in range(NUM_EXPERTS):
                    idx_pool = class_indices.get(c, torch.tensor([], dtype=torch.long))
                    if len(idx_pool) > 0:
                        sampled = idx_pool[torch.randint(0, len(idx_pool), (per_class,))]
                        batch_idx.append(sampled)
                batch_idx = torch.cat(batch_idx)
                if len(batch_idx) > batch_size:
                    batch_idx = batch_idx[:batch_size]
        else:
            indices = torch.randperm(n)

        batch_iter = 0
        while batch_iter * batch_size < n:
            if scene_labels_tensor is not None:
                # Re-sample class-balanced batch each iteration
                batch_idx = []
                for c in range(NUM_EXPERTS):
                    idx_pool = class_indices.get(c, torch.tensor([], dtype=torch.long))
                    if len(idx_pool) > 0:
                        sampled = idx_pool[torch.randint(0, len(idx_pool), (per_class,))]
                        batch_idx.append(sampled)
                batch_idx = torch.cat(batch_idx)
                if len(batch_idx) > batch_size:
                    batch_idx = batch_idx[:batch_size]
                # Stop after n // batch_size balanced batches
                if batch_iter >= n // batch_size:
                    break
            else:
                i = batch_iter * batch_size
                batch_idx = indices[i:i + batch_size]

            feat = features[batch_idx].to(self.device).float()
            target = coords[batch_idx].to(self.device).float()

            self.optimizer.zero_grad()

            if self.use_scene_class:
                fused, weights, _, logits = self.moe.forward(
                    feat, return_weights=True, return_logits=True
                )
            else:
                fused, weights, _ = self.moe.forward(feat, return_weights=True)
                logits = None

            # Build scene label mask for this batch
            scene_mask = None
            scene_lbl = None
            if scene_labels_tensor is not None and logits is not None:
                batch_labels = scene_labels_tensor[batch_idx].to(self.device)
                scene_mask = batch_labels >= 0
                scene_lbl = batch_labels

            loss, components = self.criterion(fused, target, weights,
                                              router_logits=logits,
                                              scene_labels=scene_lbl,
                                              label_mask=scene_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.moe.router.parameters(), GRAD_CLIP_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iter += 1
            final_components = components

        self.scheduler.step()
        return total_loss / max(n_batches, 1), final_components

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, coords: torch.Tensor,
                 scene_labels_tensor: torch.Tensor = None) -> dict:
        self.moe.eval()
        feat = features.to(self.device).float()
        target = coords.to(self.device).float()

        fused, weights, expert_preds, logits = self.moe.forward(
            feat, return_weights=True, return_logits=True
        )

        criterion = HaversineLoss()
        loss = criterion(fused, target)

        pred_lat = fused[:, 0] * 90.0
        pred_lng = fused[:, 1] * 180.0
        true_lat = target[:, 0] * 90.0
        true_lng = target[:, 1] * 180.0

        from src.regression.coord_regressor import haversine_distance_km
        dists = haversine_distance_km(pred_lat, pred_lng, true_lat, true_lng)

        top_expert = weights.argmax(dim=-1)
        usage = torch.bincount(top_expert, minlength=NUM_EXPERTS).float()
        usage = usage / usage.sum()

        result = {
            "loss_km": loss.item(),
            "median_km": dists.median().item(),
            "mean_km": dists.mean().item(),
            "expert_usage": {i: usage[i].item() for i in range(NUM_EXPERTS)},
        }

        # Scene classification accuracy on labeled samples
        if scene_labels_tensor is not None:
            labels_dev = scene_labels_tensor.to(self.device)
            mask = labels_dev >= 0
            if mask.any():
                pred_scene = logits[mask].argmax(dim=-1)
                acc = (pred_scene == labels_dev[mask]).float().mean()
                result["scene_acc"] = acc.item()

        return result


def train_router_pretrain(
    moe: MoERegressor,
    features: torch.Tensor,
    scene_labels_tensor: torch.Tensor,
    output_dir: str,
    epochs: int = 20,
    device: str = "cuda",
):
    """Phase 2a: pretrain router on scene classification only.

    This teaches the router to recognize scene types from visual features
    before it has to optimize for coordinate prediction.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2a: Router scene pretraining — {features.shape[0]} samples")

    moe.freeze_experts()
    moe.unfreeze_router()

    # Only use labeled samples
    mask = scene_labels_tensor >= 0
    if not mask.any():
        print("  No labeled samples, skipping")
        return moe

    feat_labeled = features[mask]
    labels_labeled = scene_labels_tensor[mask]
    print(f"  Labeled: {feat_labeled.shape[0]} samples")

    n_val = max(1, int(feat_labeled.shape[0] * 0.1))
    perm = torch.randperm(feat_labeled.shape[0])
    train_feat, val_feat = feat_labeled[perm[n_val:]], feat_labeled[perm[:n_val]]
    train_lbl, val_lbl = labels_labeled[perm[n_val:]], labels_labeled[perm[:n_val]]

    optimizer = torch.optim.AdamW(moe.router.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        moe.train()
        n = train_feat.shape[0]
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            feat = train_feat[batch_idx].to(device).float()
            target = train_lbl[batch_idx].to(device)

            optimizer.zero_grad()
            _, logits = moe.router(feat, return_logits=True)
            loss = ce_loss(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Eval
        moe.eval()
        with torch.no_grad():
            feat_v = val_feat.to(device).float()
            target_v = val_lbl.to(device)
            _, logits_v = moe.router(feat_v, return_logits=True)
            val_loss = ce_loss(logits_v, target_v).item()
            val_acc = (logits_v.argmax(dim=-1) == target_v).float().mean().item()

        status = "*" if val_acc > best_acc else " "
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(moe.state_dict(), os.path.join(output_dir, "moe_router_pretrain.pt"))

        print(f"  [{status}] E{epoch:3d} | train_loss={total_loss/max(n_batches,1):.3f} "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} best_acc={best_acc:.3f}")

    # Reload best
    if os.path.exists(os.path.join(output_dir, "moe_router_pretrain.pt")):
        moe.load_state_dict(torch.load(os.path.join(output_dir, "moe_router_pretrain.pt"),
                                       map_location=device, weights_only=True))
    print(f"  Router pretrain done. Best scene accuracy: {best_acc:.1%}")
    return moe


def train_router_phase(
    moe: MoERegressor,
    features: torch.Tensor,
    coords: torch.Tensor,
    output_dir: str,
    scene_labels: dict[int, int] = None,
    device: str = "cuda",
):
    """Phase 2: train the router on all data with frozen experts.

    If scene_labels is provided (idx → scene_type mapping), uses
    semi-supervised scene classification loss.
    """
    print(f"\n{'='*60}")
    print(f"Phase 2: Router training — {features.shape[0]} samples")

    # Build scene_labels_tensor: -1 for unlabeled, 0-3 for labeled
    n = features.shape[0]
    scene_labels_tensor = torch.full((n,), -1, dtype=torch.long)
    if scene_labels:
        n_labeled = 0
        for idx, st in scene_labels.items():
            if 0 <= idx < n and 0 <= int(st) < NUM_EXPERTS:
                scene_labels_tensor[idx] = int(st)
                n_labeled += 1
        print(f"  Labeled samples for aux loss: {n_labeled}/{n} ({100*n_labeled/n:.1f}%)")

    n_val = max(1, int(n * 0.1))
    perm = torch.randperm(n)
    train_feat, val_feat = features[perm[n_val:]], features[perm[:n_val]]
    train_coords, val_coords = coords[perm[n_val:]], coords[perm[:n_val]]
    train_labels = scene_labels_tensor[perm[n_val:]]
    val_labels = scene_labels_tensor[perm[:n_val]]

    trainer = RouterTrainer(moe, device=device,
                            use_scene_class=(scene_labels is not None))
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, ROUTER_EPOCHS + 1):
        train_loss, comps = trainer.train_epoch(train_feat, train_coords, train_labels)
        val_metrics = trainer.evaluate(val_feat, val_coords, val_labels)

        if val_metrics["loss_km"] < best_loss:
            best_loss = val_metrics["loss_km"]
            no_improve = 0
            torch.save(moe.state_dict(), os.path.join(output_dir, "moe_router.pt"))
            status = "*"
        else:
            no_improve += 1
            status = " "

        ent = comps.get("entropy", 0)
        lb = comps.get("load_balance", 0)
        sc_acc = val_metrics.get("scene_acc", -1)
        usage_str = " ".join(f"E{i}:{val_metrics['expert_usage'][i]:.2f}" for i in range(NUM_EXPERTS))

        print(f"  [{status}] E{epoch:3d} | train={train_loss:.1f} val={val_metrics['loss_km']:.1f}km "
              f"med={val_metrics['median_km']:.1f}km | H={ent:.3f} LB={lb:.4f} | "
              f"acc={sc_acc:.2f} | {usage_str}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # Reload best router
    moe.load_state_dict(torch.load(os.path.join(output_dir, "moe_router.pt"),
                                   map_location=device, weights_only=True))
    return moe


# ── Phase 3: Joint fine-tuning ──────────────────────────────────────────

def train_joint_finetune(
    moe: MoERegressor,
    features: torch.Tensor,
    coords: torch.Tensor,
    output_dir: str,
    device: str = "cuda",
):
    """Phase 3: joint fine-tuning with all parameters unfrozen at low lr."""
    print(f"\n{'='*60}")
    print(f"Phase 3: Joint fine-tuning — {features.shape[0]} samples, lr={JOINT_FINETUNE_LR}")

    moe.unfreeze_experts()
    moe.train()

    n_val = max(1, int(features.shape[0] * 0.1))
    perm = torch.randperm(features.shape[0])
    train_feat, val_feat = features[perm[n_val:]], features[perm[:n_val]]
    train_coords, val_coords = coords[perm[n_val:]], coords[perm[:n_val]]

    optimizer = torch.optim.AdamW(moe.parameters(), lr=JOINT_FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=JOINT_FINETUNE_EPOCHS)
    criterion = MoEHaversineLoss(entropy_weight=0.01, load_balance_weight=0.001)
    haversine = HaversineLoss()

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(1, JOINT_FINETUNE_EPOCHS + 1):
        moe.train()
        n = train_feat.shape[0]
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            feat = train_feat[batch_idx].to(device).float()
            target = train_coords[batch_idx].to(device).float()

            optimizer.zero_grad()
            fused, weights, _ = moe.forward(feat, return_weights=True)
            loss, _ = criterion(fused, target, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moe.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        moe.eval()
        with torch.no_grad():
            feat_v = val_feat.to(device).float()
            target_v = val_coords.to(device).float()
            fused_v, _, _ = moe.forward(feat_v, return_weights=True)
            val_loss = haversine(fused_v, target_v).item()

        status = " "
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(moe.state_dict(), os.path.join(output_dir, "moe_final.pt"))
            status = "*"
        else:
            no_improve += 1

        print(f"  [{status}] E{epoch:3d} | train={total_loss/max(n_batches,1):.1f} "
              f"val={val_loss:.1f}km | best={best_loss:.1f}km")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    moe.load_state_dict(torch.load(os.path.join(output_dir, "moe_final.pt"),
                                   map_location=device, weights_only=True))
    return moe


# ── Full pipeline ───────────────────────────────────────────────────────

def train_moe(
    features_file: str = "output/regression/features.pt",
    labels_file: str = "output/regression/scene_labels.json",
    data_file: str = "data/merged_geotagged.json",
    pretrained_path: str = "output/regression/best_model.pt",
    output_dir: str = "output/moe",
    phases: str = "1,2,3",
    device: str = None,
):
    """Full 3-phase MoE training pipeline.

    Args:
        features_file: precomputed features.pt
        labels_file: scene_labels.json from scene_labeler
        data_file: original geotagged JSON (for index alignment)
        pretrained_path: warm-start checkpoint for experts
        output_dir: where to save models
        phases: comma-separated phases to run, e.g. "1,2" or "1,2,3"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = torch.load(features_file, map_location="cpu", weights_only=True)
    features, coords = data["features"], data["coords"]
    input_dim = features.shape[1]
    print(f"  Features: {features.shape}, Coords: {coords.shape}, input_dim={input_dim}")

    # Load scene labels
    scene_labels = load_scene_labels_json(labels_file, data_file)
    print(f"  Scene labels: {len(scene_labels)} labeled / {features.shape[0]} total")

    # Fill unlabeled samples → Expert 3 (OTHER) as catch-all
    for idx in range(features.shape[0]):
        if idx not in scene_labels:
            scene_labels[idx] = int(SceneType.OTHER)

    # Split by scene
    splits = split_by_scene(features, coords, scene_labels)
    sample_counts = {i: splits[i][0].shape[0] for i in range(NUM_EXPERTS)}
    print(f"  Per-expert counts: {sample_counts}")

    expert_configs = get_expert_configs(sample_counts)
    for i in range(NUM_EXPERTS):
        name = SCENE_TYPE_NAMES.get(i, "?")
        cfg = expert_configs[i]
        print(f"    Expert {i} ({name}): {sample_counts[i]} samples → "
              f"hidden={cfg['hidden_dims']}, dropout={cfg['dropout']}")

    # ── Phase 1: Expert pretraining ──────────────────────────────────
    phases_to_run = [p.strip() for p in phases.split(",")]
    if "1" in phases_to_run:
        experts_trained = []
        for i in range(NUM_EXPERTS):
            expert = train_expert_pretrain(
                expert_idx=i,
                scene_features=splits[i][0],
                scene_coords=splits[i][1],
                pretrained_path=pretrained_path,
                config=expert_configs[i],
                output_dir=output_dir,
                device=device,
                input_dim=input_dim,
            )
            experts_trained.append(expert)

        print(f"\nPhase 1 complete. Experts saved to {output_dir}/expert_*.pt")
    else:
        print("\nSkipping Phase 1 (loading existing experts)")
        experts_trained = []
        for i in range(NUM_EXPERTS):
            path = os.path.join(output_dir, f"expert_{i}.pt")
            if os.path.exists(path):
                expert = create_expert(input_dim=input_dim, config=expert_configs[i], pretrained_path=path, device=device)
            elif pretrained_path and os.path.exists(pretrained_path):
                expert = create_expert(input_dim=input_dim, config=expert_configs[i], pretrained_path=pretrained_path, device=device)
            else:
                expert = create_expert(input_dim=input_dim, config=expert_configs[i], device=device)
            experts_trained.append(expert)

    # Create MoE with trained experts
    moe = MoERegressor(
        input_dim=input_dim,
        expert_configs=expert_configs,
    )
    moe.to(device)

    # Load phase 1 expert weights
    for i, expert in enumerate(experts_trained):
        moe.load_expert_state(i, expert.state_dict())

    # ── Phase 2a: Router scene pretraining ──────────────────────────
    # Build scene_labels_tensor for router training
    scene_labels_tensor = torch.full((features.shape[0],), -1, dtype=torch.long)
    if scene_labels:
        for idx, st in scene_labels.items():
            if 0 <= idx < features.shape[0] and 0 <= int(st) < NUM_EXPERTS:
                scene_labels_tensor[idx] = int(st)

    if "2a" in phases_to_run and scene_labels_tensor[scene_labels_tensor >= 0].numel() > 0:
        moe = train_router_pretrain(moe, features, scene_labels_tensor, output_dir, device=device)
        print(f"\nPhase 2a complete. Router pretrained on scene classification.")

    # ── Phase 2: Router training ─────────────────────────────────────
    if "2" in phases_to_run:
        moe = train_router_phase(moe, features, coords, output_dir,
                                 scene_labels=scene_labels, device=device)
        print(f"\nPhase 2 complete. Router saved to {output_dir}/moe_router.pt")
    else:
        print("\nSkipping Phase 2")

    # ── Phase 3: Joint fine-tuning ───────────────────────────────────
    if "3" in phases_to_run:
        moe = train_joint_finetune(moe, features, coords, output_dir, device)
        print(f"\nPhase 3 complete. Final model saved to {output_dir}/moe_final.pt")
    else:
        print("\nSkipping Phase 3")

    # Save final
    torch.save(moe.state_dict(), os.path.join(output_dir, "moe_final.pt"))

    print(f"\n{'='*60}")
    print(f"MoE training complete. Output: {output_dir}/")
    print(f"  expert_*.pt    — per-expert checkpoints")
    print(f"  moe_router.pt  — best router (phase 2)")
    print(f"  moe_final.pt   — final model (phase 3 or best phase)")

    return moe


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MoE coordinate regression")
    parser.add_argument("--features", type=str, default="output/regression/features.pt")
    parser.add_argument("--labels", type=str, default="output/regression/scene_labels.json")
    parser.add_argument("--data", type=str, default="data/merged_geotagged.json")
    parser.add_argument("--pretrained", type=str, default="output/regression/best_model.pt")
    parser.add_argument("--output", type=str, default="output/moe")
    parser.add_argument("--phases", type=str, default="1,2,3",
                        help="Comma-separated: 1=expert pretrain, 2=router, 3=joint")
    args = parser.parse_args()

    train_moe(
        features_file=args.features,
        labels_file=args.labels,
        data_file=args.data,
        pretrained_path=args.pretrained,
        output_dir=args.output,
        phases=args.phases,
    )


if __name__ == "__main__":
    main()
