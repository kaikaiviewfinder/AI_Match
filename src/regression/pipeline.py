"""Integrated pipeline: GeoCoT reasoning + coordinate regression head.

GeoCoTWithRegression:
  1. Run 3-stage GeoCoT reasoning → text-based coordinates
  2. Extract visual features → regression head → feature-based coordinates
  3. Fuse: use regression coordinates if model is trained, else GeoCoT text coords
  4. If MoE is loaded, generate full explanation report with confidence rating
  5. (NEW) Cascade mode: GeoCoT spatial narrowing → constrained MoE regression

Three inference paths (auto-selected by available models):
  - Cascade:  GeoCoT → Parser → SpatialKB → Constraint → MoE (best for wilderness)
  - MoE:      Router + 4 experts → fusion + explanation (balanced)
  - Single:   One CoordRegressor → coordinates (fallback)

This gives the best of both worlds:
  - GeoCoT provides reasoning chain and semantic location (city/country)
  - Regression head provides high-precision coordinates
  - MoE provides scene-specialized experts + explainability
  - Cascade provides spatial constraint narrowing for wilderness scenes
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Ensure src is on path for cross-package imports
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Geocot.Geocot import (
    GeoCoTPipeline, GeoCoTResult, GeoPrediction, extract_prediction,
    create_qwen2vl_model_fn, load_qwen2vl,
)
from regression.feature_extractor import VisualFeatureExtractor
from regression.coord_regressor import CoordRegressor, create_expert
from regression.cascade_pipeline import CascadePipeline


class GeoCoTWithRegression:
    """GeoCoT + coordinate regression hybrid pipeline.

    Supports two regression modes:
      - Single: one CoordRegressor for all scenes
      - MoE: 4 specialized experts + router → weighted prediction + explanation
    """

    def __init__(
        self,
        model,
        processor,
        model_fn=None,
        prompt_config: Optional[dict] = None,
        regressor_path: Optional[str] = None,
        moe_path: Optional[str] = None,
        cascade_config: Optional[dict] = None,
    ):
        if model_fn is None:
            model_fn = create_qwen2vl_model_fn(model, processor)
        self._geocot = GeoCoTPipeline(model_fn, prompt_config)
        self._extractor = VisualFeatureExtractor(model, processor)
        self._device = model.device

        # Single regressor
        self._regressor: Optional[CoordRegressor] = None
        self._regressor_loaded = False

        # MoE regressor (preferred when available)
        self._moe = None
        self._moe_loaded = False

        # Cascade pipeline (best when both GeoCoT and MoE are available)
        self._cascade = None
        self._cascade_loaded = False

        if moe_path and os.path.exists(moe_path):
            self._load_moe(moe_path)
            # Also init cascade if MoE loaded (needs both GeoCoT + MoE)
            self._init_cascade(cascade_config or {})
        elif regressor_path and os.path.exists(regressor_path):
            self._load_regressor(regressor_path)

    def _load_regressor(self, path: str):
        """Load a trained single regression head."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._regressor = create_expert(pretrained_path=path, device=self._device)
        self._regressor.eval()
        self._regressor_loaded = True

    def _load_moe(self, path: str):
        """Load a trained MoE regressor."""
        from regression.moe import MoERegressor
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._moe = MoERegressor()
        self._moe.load_state_dict(state)
        self._moe.to(self._device)
        self._moe.eval()
        self._moe_loaded = True

    def _init_cascade(self, config: dict):
        """Initialize cascade pipeline (requires MoE to be loaded first)."""
        if not self._moe_loaded:
            return
        try:
            self._cascade = CascadePipeline(
                geocot_pipeline=self._geocot,
                moe_model=self._moe,
                feature_extractor=self._extractor,
                fuser_config=config.get("fuser_config"),
            )
            self._cascade_loaded = True
        except Exception as e:
            print(f"[Cascade] Init failed: {e}")

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        use_regression: bool = True,
        use_moe: bool = True,
        use_cascade: bool = True,
    ) -> GeoCoTResult:
        """Run the full hybrid pipeline.

        Priority: Cascade > MoE > Single regressor > GeoCoT text-only

        Args:
            image: PIL RGB image
            use_regression: if True, use regression head (single or MoE)
            use_moe: if True and MoE loaded, use MoE path with explanation
            use_cascade: if True and cascade loaded, use cascade path
                         (GeoCoT spatial narrowing → constrained MoE)

        Returns:
            GeoCoTResult with final_prediction + explanation (if available)
        """
        # Cascade path: GeoCoT spatial narrowing → constrained MoE
        if use_cascade and self._cascade_loaded:
            return self._run_cascade_path(image)

        # 1. GeoCoT reasoning (always run for semantic context)
        result = self._geocot.run(image)

        # 2. Extract visual features (needed for both single and MoE paths)
        features = None
        try:
            features = self._extractor.extract(image).float()
        except Exception as e:
            result.stage_outputs["feature_error"] = str(e)

        # 3. Regression prediction
        if features is not None:
            if use_regression and use_moe and self._moe_loaded:
                self._run_moe_path(result, features)
            elif use_regression and self._regressor_loaded:
                self._run_single_path(result, features)

        return result

    def _run_single_path(self, result: GeoCoTResult, features: torch.Tensor):
        """Single regressor path (original behavior)."""
        try:
            reg_lat, reg_lng = self._regressor.predict(features.unsqueeze(0))
            reg_lat, reg_lng = reg_lat.item(), reg_lng.item()

            from Geocot.Geocot import _coords_to_nearest_city, _infer_continent, lookup_trail

            kb_city, kb_country = _coords_to_nearest_city(reg_lat, reg_lng)

            if kb_city:
                result.final_prediction.city = kb_city
                result.final_prediction.country = kb_country or "China"
                result.final_prediction.continent = _infer_continent(result.final_prediction.country)
                result.final_prediction.latitude = reg_lat
                result.final_prediction.longitude = reg_lng
            else:
                result.final_prediction.latitude = reg_lat
                result.final_prediction.longitude = reg_lng
                if not result.final_prediction.city or result.final_prediction.city == "Unknown":
                    result.final_prediction.city = "Unknown"
                    result.final_prediction.country = "China"
                    result.final_prediction.continent = "Asia"

            result.stage_outputs["regression"] = (
                f"COORDINATES: {reg_lat:.4f}, {reg_lng:.4f}\n"
                f"LOCATION: {result.final_prediction.city}, "
                f"{result.final_prediction.country}, "
                f"{result.final_prediction.continent}"
            )
        except Exception as e:
            result.stage_outputs["regression_error"] = str(e)

    def _run_moe_path(self, result: GeoCoTResult, features: torch.Tensor):
        """MoE path with full explainability report."""
        try:
            lat, lng, weights, expert_lats, expert_lngs = self._moe.predict(
                features.unsqueeze(0), return_expert_info=True
            )
            fused_lat = lat.item()
            fused_lng = lng.item()
            weights_squeezed = weights.squeeze(0)  # (4,)
            expert_lats_sq = expert_lats.squeeze(0)  # (4,)
            expert_lngs_sq = expert_lngs.squeeze(0)  # (4,)

            # KB reverse lookup
            from Geocot.Geocot import _coords_to_nearest_city, _infer_continent
            kb_city, kb_country = _coords_to_nearest_city(fused_lat, fused_lng)

            if kb_city:
                result.final_prediction.city = kb_city
                result.final_prediction.country = kb_country or "China"
                result.final_prediction.continent = _infer_continent(result.final_prediction.country)
            elif not result.final_prediction.city or result.final_prediction.city == "Unknown":
                result.final_prediction.city = "Unknown"
                result.final_prediction.country = "China"
                result.final_prediction.continent = "Asia"

            result.final_prediction.latitude = fused_lat
            result.final_prediction.longitude = fused_lng

            # Build explanation report
            geocot_text = "\n".join(
                v for k, v in result.stage_outputs.items()
                if k in ("macro", "regional", "local")
            )

            from regression.explanation import GeoExplanation
            explanation = GeoExplanation()
            explanation.build_from_moe(
                router_weights=weights_squeezed,
                expert_lats=expert_lats_sq,
                expert_lngs=expert_lngs_sq,
                fused_lat=fused_lat,
                fused_lng=fused_lng,
                geocot_city=result.final_prediction.city,
                geocot_country=result.final_prediction.country,
                geocot_continent=result.final_prediction.continent,
                geocot_output=geocot_text,
                geocot_scene_type=result.scene_type,
            )
            result.explanation = explanation

            result.stage_outputs["regression"] = (
                f"MoE COORDINATES: {fused_lat:.4f}, {fused_lng:.4f}\n"
                f"LOCATION: {result.final_prediction.city}, "
                f"{result.final_prediction.country}, "
                f"{result.final_prediction.continent}\n"
                f"CONFIDENCE: {explanation.confidence_stars}/5 stars"
            )
        except Exception as e:
            result.stage_outputs["moe_error"] = str(e)

    @torch.no_grad()
    def _run_cascade_path(self, image: Image.Image) -> GeoCoTResult:
        """Full cascade: GeoCoT spatial narrowing → constrained MoE → explanation."""
        from regression.cascade_pipeline import CascadeResult

        cascade_result: CascadeResult = self._cascade.run(image)

        # Convert CascadeResult to GeoCoTResult (compatible interface)
        result = cascade_result.geocot

        if cascade_result.error:
            result.stage_outputs["cascade_error"] = cascade_result.error
            return result

        # Store cascade info in stage outputs
        result.stage_outputs["cascade"] = str(cascade_result.cascade_info)
        result.explanation = cascade_result.explanation

        if cascade_result.latitude is not None:
            from Geocot.Geocot import _coords_to_nearest_city, _infer_continent

            kb_city, kb_country = _coords_to_nearest_city(
                cascade_result.latitude, cascade_result.longitude
            )

            if kb_city:
                result.final_prediction.city = kb_city
                result.final_prediction.country = kb_country or "China"
                result.final_prediction.continent = _infer_continent(kb_country or "China")
            elif not result.final_prediction.city or result.final_prediction.city == "Unknown":
                result.final_prediction.city = "Unknown"
                result.final_prediction.country = "China"
                result.final_prediction.continent = "Asia"

            result.final_prediction.latitude = cascade_result.latitude
            result.final_prediction.longitude = cascade_result.longitude

            # Format regression output
            const = cascade_result.constraint
            const_info = ""
            if const is not None and not const.is_default:
                const_info = (
                    f"\nCASCADE CONSTRAINT: radius={const.radius_km:.0f}km, "
                    f"confidence={const.confidence:.1%}, source={const.source}"
                )

            result.stage_outputs["regression"] = (
                f"CASCADE COORDINATES: {cascade_result.latitude:.4f}, "
                f"{cascade_result.longitude:.4f}\n"
                f"LOCATION: {result.final_prediction.city}, "
                f"{result.final_prediction.country}, "
                f"{result.final_prediction.continent}"
                + const_info
            )

        return result

    @property
    def has_regressor(self) -> bool:
        return self._regressor_loaded

    @property
    def has_moe(self) -> bool:
        return self._moe_loaded

    @property
    def has_cascade(self) -> bool:
        return self._cascade_loaded


def load_geocot_with_regression(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    prompts_dir: str = None,
    few_shot_path: str = None,
    regressor_path: str = None,
    moe_path: str = None,
    cascade_config: dict = None,
    load_in_4bit: bool = False,
    enable_knowledge: bool = True,
):
    """Load the full hybrid pipeline.

    Args:
        model_name: HuggingFace model ID
        prompts_dir: custom prompts directory
        few_shot_path: path to fewshot_china.json
        regressor_path: path to single CoordRegressor .pt
        moe_path: path to MoE model .pt (takes priority if both given)
        cascade_config: dict with optional keys:
            - fuser_config: dict for SpatialConstraintFuser params
        load_in_4bit: use 4-bit quantization
        enable_knowledge: enable knowledge injection in GeoCoT prompts

    Returns:
        (model, processor, pipeline)
    """
    model, processor, model_fn = load_qwen2vl(model_name, load_in_4bit=load_in_4bit)

    pipeline = GeoCoTWithRegression(
        model,
        processor,
        model_fn=model_fn,
        prompt_config={
            "prompts_dir": prompts_dir,
            "few_shot_path": few_shot_path,
            "enable_knowledge": enable_knowledge,
        },
        regressor_path=regressor_path,
        moe_path=moe_path,
        cascade_config=cascade_config,
    )

    return model, processor, pipeline
