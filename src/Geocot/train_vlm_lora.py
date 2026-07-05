"""LoRA fine-tune Qwen3-VL-2B for 3-stage GeoCoT geographic reasoning.

Supports two data formats:
  - "3stage": auto-labeled GPS→GeoKB→3-stage JSON (macro/regional/local)
  - "narrative": manual instruction-response pairs

Each 3stage record expands into 3 training examples, one per GeoCoT stage.
Dependencies: peft, bitsandbytes
Usage:
    python -m src.Geocot.train_vlm_lora \
        --data data/vlm_finetune/geocot_balanced_train.jsonl \
        --output output/vlm_lora_geocot \
        --epochs 2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ── Prompt templates (loaded from prompt files) ────────────────────────────

def _load_prompt_templates() -> Dict[str, str]:
    """Load Chinese GeoCoT prompt templates from prompts/ directory."""
    prompts_dir = Path(__file__).resolve().parent / "prompts"
    templates = {}
    for stage in ["macro", "regional", "local"]:
        path = prompts_dir / f"{stage}.txt"
        if path.exists():
            templates[stage] = path.read_text(encoding="utf-8").strip()
    return templates

PROMPT_TEMPLATES = _load_prompt_templates()


def build_stage_prompt(stage: str, prev_outputs: Optional[Dict[str, str]] = None) -> str:
    """Build the full prompt for a GeoCoT stage.

    Args:
        stage: "macro", "regional", or "local"
        prev_outputs: dict with keys matching template placeholders
                       (e.g. {"macro_output": "...", "regional_output": "..."})

    Returns:
        Complete prompt text for this stage.
    """
    template = PROMPT_TEMPLATES.get(stage, "")
    if not template:
        raise ValueError(f"No prompt template found for stage: {stage}")

    if prev_outputs:
        # regional.txt uses {prev_output}, local.txt uses {macro_output}/{regional_output}
        # Normalize both patterns
        if stage == "regional" and "prev_output" not in template:
            pass
        if stage == "local":
            macro = prev_outputs.get("macro_output", "{}")
            regional = prev_outputs.get("regional_output", "{}")
            template = template.replace("{macro_output}", macro)
            template = template.replace("{regional_output}", regional)
        elif stage == "regional":
            prev = prev_outputs.get("prev_output", prev_outputs.get("macro_output", "{}"))
            template = template.replace("{prev_output}", prev)

    return template


# ── Dataset ────────────────────────────────────────────────────────────────

class GeoCoTDataset(Dataset):
    """Dataset supporting both 3-stage GeoCoT and narrative formats.

    For "3stage" records: each image generates 3 samples (macro/regional/local).
    For "narrative" records: each image generates 1 sample (instruction→response).

    Each item returns a dict with keys:
        image, prompt_text, response_text, stage (or "narrative")
    """

    STAGES = ["macro", "regional", "local"]

    def __init__(
        self,
        jsonl_path: str,
        processor,
        max_length: int = 1024,
        stages_to_train: Optional[List[str]] = None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.stages_to_train = stages_to_train or self.STAGES
        self.samples: List[dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                img_path = rec.get("image", "")
                if not img_path or not os.path.exists(img_path):
                    continue

                fmt = rec.get("format", "narrative")

                if fmt == "3stage":
                    # Expand into 3 training examples (one per stage)
                    macro_output = rec.get("macro_output", "{}")
                    regional_output = rec.get("regional_output", "{}")
                    local_output = rec.get("local_output", "{}")

                    stage_prompts = {
                        "macro": build_stage_prompt("macro"),
                        "regional": build_stage_prompt("regional", {"prev_output": macro_output}),
                        "local": build_stage_prompt("local", {
                            "macro_output": macro_output,
                            "regional_output": regional_output,
                        }),
                    }
                    stage_responses = {
                        "macro": macro_output,
                        "regional": regional_output,
                        "local": local_output,
                    }

                    for stage in self.stages_to_train:
                        self.samples.append({
                            "image_path": img_path,
                            "prompt": stage_prompts[stage],
                            "response": stage_responses[stage],
                            "stage": stage,
                        })

                elif fmt == "narrative":
                    self.samples.append({
                        "image_path": img_path,
                        "prompt": rec.get("instruction", ""),
                        "response": rec.get("response", ""),
                        "stage": "narrative",
                    })

        if not self.samples:
            raise ValueError(f"No valid records found in {jsonl_path}")

        # Stats
        stage_counts = {}
        for s in self.samples:
            stage_counts[s["stage"]] = stage_counts.get(s["stage"], 0) + 1
        print(f"  Samples: {len(self.samples)} total")
        for stage, count in sorted(stage_counts.items()):
            print(f"    {stage}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["response"]},
            ]},
        ]

        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].squeeze(0)

        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def collate_vlm(batch):
    """Pad and collate VLM inputs."""
    from torch.nn.utils.rnn import pad_sequence

    pad_keys = {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}
    collated = {}
    for key in batch[0]:
        tensors = [item[key] for item in batch]
        if key in pad_keys:
            collated[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        elif key in ("image_grid_thw",):
            collated[key] = torch.stack(tensors)
        elif key in ("pixel_values",):
            collated[key] = torch.cat(tensors, dim=0)
        else:
            collated[key] = tensors[0] if isinstance(tensors[0], torch.Tensor) else tensors

    if "labels" in collated and "attention_mask" in collated:
        collated["labels"][collated["attention_mask"] == 0] = -100

    return collated


# ── Element-level validation ───────────────────────────────────────────────

# Fields expected in each stage's JSON output, with their types
MACRO_FIELDS = {
    "climate_zone": "enum",
    "terrain_type": "enum",
    "vegetation_zone": "enum",
    "urbanization": "enum",
}

REGIONAL_FIELDS = {
    "language_script": "enum",
    "architecture_style": "enum",
    "mountain_rock_type": "nullable",
    "sky_quality": "enum",
    "likely_country": "string",
}

LOCAL_FIELDS = {
    "scene_type": "enum",
    "landform_detail": "nullable",
    "rock_color": "nullable",
    "soil_color": "nullable",
    "water_visible": "bool",
    "water_type": "nullable",
    "confidence": "enum",
    "province": "string",
}

STAGE_FIELDS = {
    "macro": MACRO_FIELDS,
    "regional": REGIONAL_FIELDS,
    "local": LOCAL_FIELDS,
}


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON object from model output text."""
    # Try ```json ... ``` block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON object
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate_elements(model, processor, val_samples: List[dict], device: str = "cuda") -> dict:
    """Evaluate per-element prediction accuracy on validation samples.

    Each val_sample should have: image_path, stage, ground_truth (JSON string)

    Returns dict: {stage: {field: (correct, total, accuracy)}}
    """
    model.eval()
    results = {stage: {} for stage in STAGE_FIELDS}

    for sample in val_samples:
        stage = sample["stage"]
        gt_json = _extract_json(sample["response"])
        if gt_json is None:
            continue

        image = Image.open(sample["image_path"]).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ]},
        ]

        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        pred_json = _extract_json(output_text)
        if pred_json is None:
            continue

        fields = STAGE_FIELDS.get(stage, {})
        for field, ftype in fields.items():
            if field not in results[stage]:
                results[stage][field] = {"correct": 0, "total": 0}
            results[stage][field]["total"] += 1

            gt_val = gt_json.get(field)
            pred_val = pred_json.get(field)

            if ftype == "bool":
                if bool(pred_val) == bool(gt_val):
                    results[stage][field]["correct"] += 1
            elif ftype == "list":
                # For lists, check if there's overlap
                gt_set = set(gt_val) if isinstance(gt_val, list) else {gt_val}
                pred_set = set(pred_val) if isinstance(pred_val, list) else {pred_val}
                if gt_set & pred_set:
                    results[stage][field]["correct"] += 1
            elif ftype == "nullable":
                if (gt_val is None and pred_val is None) or (gt_val == pred_val):
                    results[stage][field]["correct"] += 1
            else:
                if str(gt_val).lower() == str(pred_val).lower():
                    results[stage][field]["correct"] += 1

    # Compute accuracies
    report = {}
    for stage, fields in results.items():
        report[stage] = {}
        for field, counts in fields.items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            report[stage][field] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": acc,
            }

    model.train()
    return report


def print_validation_report(report: dict):
    """Pretty-print element-level validation results."""
    print("\n  Element-Level Validation:")
    print(f"  {'Stage':<12} {'Field':<24} {'Acc':>6} {'Count':>8}")
    print(f"  {'-'*50}")
    all_correct = 0
    all_total = 0
    for stage in ["macro", "regional", "local"]:
        stage_data = report.get(stage, {})
        for field, info in sorted(stage_data.items()):
            print(f"  {stage:<12} {field:<24} {info['accuracy']:>5.1%} {info['total']:>7}")
            all_correct += info["correct"]
            all_total += info["total"]
    if all_total > 0:
        print(f"  {'─'*50}")
        print(f"  {'OVERALL':<12} {'':<24} {all_correct/all_total:>5.1%} {all_total:>7}")


# ── Model loading ──────────────────────────────────────────────────────────

def _detect_model_class(model_name: str):
    """Detect the correct model class from config (supports Qwen2-VL and Qwen3-VL)."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    if config.model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    elif config.model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    else:
        from transformers import Qwen2VLForConditionalGeneration
        print(f"  [WARN] Unknown model_type={config.model_type}, falling back to Qwen2VL")
        return Qwen2VLForConditionalGeneration


# ── Training ───────────────────────────────────────────────────────────────

def train_lora(
    data_file: str,
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    output_dir: str = "output/vlm_lora_geocot",
    epochs: int = 2,
    batch_size: int = 1,
    lr: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    load_in_4bit: bool = True,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 200,
    logging_steps: int = 10,
    max_length: int = 1024,
    val_split: float = 0.05,
    val_max_samples: int = 100,
):
    """Main LoRA training function for 3-stage GeoCoT.

    Args:
        data_file: JSONL file with 3stage and/or narrative format records
        model_name: HuggingFace model ID (default: Qwen3-VL-2B-Instruct)
        output_dir: where to save adapter weights
        epochs: number of training epochs
        batch_size: per-device batch size (keep 1 for 8GB VRAM)
        lr: learning rate
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha
        load_in_4bit: use 4-bit quantization (default True for 8GB VRAM)
        gradient_accumulation_steps: effective batch = batch_size × accum
        save_steps: checkpoint every N steps
        logging_steps: log every N steps
        max_length: max token length for text
        val_split: fraction of data for validation
        val_max_samples: max validation samples (element validation is slow)
    """
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_name}")

    # ── Load model ──────────────────────────────────────────────────
    ModelClass = _detect_model_class(model_name)
    print(f"  Model class: {ModelClass.__name__}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("  4-bit quantization: enabled (nf4, double-quant)")

    print(f"Loading {model_name}...")
    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        quantization_config=bnb_config,
        device_map="auto" if load_in_4bit else None,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=50176,    # 224×224 → ~128 visual tokens
        max_pixels=50176,    # Fixed resolution = consistent VRAM usage
        local_files_only=True,
    )

    if not load_in_4bit:
        model.to(device)

    # ── LoRA config ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Enable gradient checkpointing with non-reentrant mode (less memory)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()

    # ── Data ────────────────────────────────────────────────────────
    full_dataset = GeoCoTDataset(data_file, processor, max_length=max_length)
    print(f"Dataset: {len(full_dataset)} training samples")

    # Split off validation subset for element-level checks
    val_samples = []
    train_samples = list(range(len(full_dataset)))
    if val_split > 0 and len(full_dataset.samples) > 10:
        import random
        random.seed(42)
        val_indices = set(random.sample(
            train_samples,
            min(int(len(train_samples) * val_split), val_max_samples),
        ))
        val_samples = [full_dataset.samples[i] for i in sorted(val_indices)]
        # Keep the full dataset for training — validation is done separately
        print(f"  Validation samples: {len(val_samples)}")

    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_vlm,
    )

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(dataloader) * epochs) // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training loop ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    global_step = 0
    accum_loss = 0.0

    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs, {total_steps} steps")
    print(f"  Batch size: {batch_size}, Grad accum: {gradient_accumulation_steps}")
    print(f"  Effective batch: {batch_size * gradient_accumulation_steps}")
    print(f"  LR: {lr}, Rank: {lora_rank}, Alpha: {lora_alpha}")
    print(f"  Max sequence length: {max_length}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            del outputs

            accum_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Periodic cleanup to prevent VRAM fragmentation
                if global_step % 20 == 0:
                    torch.cuda.empty_cache()

                if global_step % logging_steps == 0:
                    pbar.set_postfix({
                        "loss": f"{accum_loss * gradient_accumulation_steps:.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })
                    accum_loss = 0.0

                if global_step % save_steps == 0:
                    checkpoint = os.path.join(output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(checkpoint)
                    print(f"\n  Saved checkpoint to {checkpoint}")

        # ── End-of-epoch validation ─────────────────────────────────
        if val_samples:
            print(f"\n  Running element-level validation (epoch {epoch})...")
            report = validate_elements(model, processor, val_samples, device)
            print_validation_report(report)

    # ── Save final ──────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\nLoRA adapter saved to {final_path}")

    # Merge and save full model
    try:
        print("Merging LoRA weights...")
        merged = model.merge_and_unload()
        merged_path = os.path.join(output_dir, "merged")
        merged.save_pretrained(merged_path)
        processor.save_pretrained(merged_path)
        print(f"Merged model saved to {merged_path}")
    except Exception as e:
        print(f"Merging skipped: {e}")

    return model


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen3-VL-2B for 3-stage GeoCoT geographic reasoning")
    parser.add_argument("--data", type=str, default="data/vlm_finetune/geocot_balanced_train.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output", type=str, default="output/vlm_lora_geocot")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (use bfloat16)")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--val-max-samples", type=int, default=100)
    args = parser.parse_args()

    use_4bit = not args.no_4bit

    train_lora(
        data_file=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        load_in_4bit=use_4bit,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        val_split=args.val_split,
        val_max_samples=args.val_max_samples,
    )


if __name__ == "__main__":
    main()
