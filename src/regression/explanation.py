"""Inference explanation: 3-layer reasoning chain + GeoCoT-router reconciliation.

Generates structured explanation dictionaries and formatted text reports
that show HOW the model arrived at its prediction, including:
  1. Scene analysis (which expert was activated)
  2. Expert collaboration (weighted fusion breakdown)
  3. KB anchoring (nearest landmark/city)
  4. GeoCoT-router reconciliation
  5. Confidence rating (1-5 stars)

Confidence ratings:
  ★★★★★ (5): MoE集中(>80%) + GeoCoT权威专家一致 + 坐标接近(<30km)
  ★★★★ (4): MoE集中(>80%) + GeoCoT一致 (无坐标验证)
  ★★★  (3): MoE集中 + GeoCoT分歧 → 以MoE为准
  ★★   (2): MoE分散(<50%) + GeoCoT一致 → 训练覆盖不足
  ★    (1): MoE分散 + GeoCoT分歧 → 高度不确定
"""

from typing import Optional

import torch

from .moe_config import SceneType, NUM_EXPERTS, SCENE_TYPE_LABELS_CN, SCENE_TYPE_NAMES
from .coord_regressor import haversine_distance_km


# ── Confidence rating ───────────────────────────────────────────────────

class Confidence:
    """Confidence rating with helper to assign stars based on conditions."""

    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1

    @staticmethod
    def compute(
        router_weights: torch.Tensor,
        geocot_scene_type: Optional[int] = None,
        moe_coords: Optional[tuple[float, float]] = None,
        geocot_coords: Optional[tuple[float, float]] = None,
    ) -> tuple[int, str]:
        """Compute confidence rating from MoE + GeoCoT signals.

        Args:
            router_weights: (4,) softmax weights from router
            geocot_scene_type: scene type inferred from GeoCoT text (optional)
            moe_coords: (lat, lng) from MoE fused prediction (optional)
            geocot_coords: (lat, lng) from GeoCoT text output (optional)

        Returns:
            (stars (1-5), reason_string)
        """
        max_weight = router_weights.max().item()
        top_expert = router_weights.argmax().item()
        is_concentrated = max_weight > 0.8
        is_dispersed = max_weight < 0.5

        # GeoCoT-MoE agreement check
        geocot_agrees = True
        if geocot_scene_type is not None:
            geocot_agrees = (geocot_scene_type == top_expert)

        # Coordinate agreement check
        coords_close = False
        if moe_coords is not None and geocot_coords is not None:
            dist = haversine_distance_km(
                moe_coords[0], moe_coords[1],
                geocot_coords[0], geocot_coords[1],
            )
            coords_close = dist < 30.0

        if is_concentrated and geocot_agrees and coords_close:
            return (Confidence.VERY_HIGH,
                    f"高度可信：MoE路由集中({max_weight:.0%})，GeoCoT文本推理一致，坐标接近")

        if is_concentrated and geocot_agrees:
            return (Confidence.HIGH,
                    f"较高可信：MoE路由集中({max_weight:.0%})，GeoCoT推理一致")

        if is_concentrated and not geocot_agrees:
            geocot_name = SCENE_TYPE_LABELS_CN.get(geocot_scene_type, "未知") if geocot_scene_type is not None else "未知"
            top_name = SCENE_TYPE_LABELS_CN.get(top_expert, "?")
            return (Confidence.MEDIUM,
                    f"中等可信：MoE路由集中({max_weight:.0%}→{top_name})，"
                    f"但GeoCoT判定为{geocot_name}，以MoE为准")

        if is_dispersed and geocot_agrees:
            return (Confidence.LOW,
                    f"低可信：MoE路由分散({max_weight:.0%})，可能是训练覆盖不足的场景，"
                    f"但GeoCoT推理一致")

        return (Confidence.VERY_LOW,
                f"极低可信：MoE路由分散({max_weight:.0%})且GeoCoT推理分歧，建议人工确认")


# ── Structured explanation ──────────────────────────────────────────────

class GeoExplanation:
    """Structured explanation for a single image prediction.

    Contains all the information needed for a human-readable report:
    scene analysis, expert collaboration, KB anchoring, and confidence.
    """

    def __init__(self):
        self.scene_analysis: dict = {}
        self.expert_collaboration: dict = {}
        self.kb_anchoring: dict = {}
        self.geocot_summary: str = ""
        self.reconciliation: dict = {}
        self.confidence_stars: int = Confidence.LOW
        self.confidence_reason: str = ""
        self.final_coords: tuple[float, float] = (0.0, 0.0)
        self.final_location: str = ""

    def build_from_moe(
        self,
        router_weights: torch.Tensor,       # (4,)
        expert_lats: torch.Tensor,          # (4,)
        expert_lngs: torch.Tensor,          # (4,)
        fused_lat: float,
        fused_lng: float,
        geocot_city: str = "",
        geocot_country: str = "",
        geocot_continent: str = "",
        geocot_output: str = "",
        geocot_scene_type: object = None,   # SceneType int from GeoCoT output
    ):
        """Build full explanation from MoE inference results + GeoCoT output.

        Args:
            router_weights: (4,) softmax from router
            expert_lats/lngs: (4,) each expert's prediction in degrees
            fused_lat/lng: MoE fused prediction in degrees
            geocot_*: GeoCoT text-based location info
            geocot_output: full GeoCoT reasoning text
        """
        top_idx = router_weights.argmax().item()
        top_weight = router_weights.max().item()

        # 1. Scene analysis
        self.scene_analysis = {
            "primary_scene": SCENE_TYPE_LABELS_CN.get(top_idx, str(top_idx)),
            "primary_scene_en": SCENE_TYPE_NAMES.get(top_idx, "unknown"),
            "router_confidence": round(top_weight, 4),
            "all_weights": {
                SCENE_TYPE_LABELS_CN.get(i, str(i)): round(router_weights[i].item(), 4)
                for i in range(NUM_EXPERTS)
            },
        }

        # 2. Expert collaboration
        self.expert_collaboration = {
            "fused_lat": round(fused_lat, 4),
            "fused_lng": round(fused_lng, 4),
            "experts": {},
        }
        for i in range(NUM_EXPERTS):
            name = SCENE_TYPE_LABELS_CN.get(i, str(i))
            weight = router_weights[i].item()
            self.expert_collaboration["experts"][name] = {
                "weight": round(weight, 4),
                "lat": round(expert_lats[i].item(), 4),
                "lng": round(expert_lngs[i].item(), 4),
                "contribution": round(weight * expert_lats[i].item(), 4),  # weighted lat
            }

        # 3. KB anchoring
        self.kb_anchoring = {
            "city": geocot_city or "Unknown",
            "country": geocot_country or "China",
            "continent": geocot_continent or "Asia",
            "coordinates": (round(fused_lat, 4), round(fused_lng, 4)),
        }

        # 4. GeoCoT summary (extract key points)
        self.geocot_summary = _summarize_geocot(geocot_output)

        # 5. Reconciliation
        self.reconciliation = _build_reconciliation(
            router_weights, top_idx, geocot_output, fused_lat, fused_lng,
        )

        # 6. Confidence
        self.confidence_stars, self.confidence_reason = Confidence.compute(
            router_weights,
            geocot_scene_type=geocot_scene_type,
        )
        self.final_coords = (round(fused_lat, 4), round(fused_lng, 4))
        self.final_location = f"{geocot_city}, {geocot_country}" if geocot_city else "Unknown"

    def to_dict(self) -> dict:
        """Return structured explanation as a serializable dict."""
        return {
            "scene_analysis": self.scene_analysis,
            "expert_collaboration": self.expert_collaboration,
            "kb_anchoring": self.kb_anchoring,
            "geocot_summary": self.geocot_summary,
            "reconciliation": self.reconciliation,
            "confidence": {
                "stars": self.confidence_stars,
                "reason": self.confidence_reason,
            },
            "final_prediction": {
                "latitude": self.final_coords[0],
                "longitude": self.final_coords[1],
                "location": self.final_location,
            },
        }

    def to_text(self) -> str:
        """Format as a human-readable Chinese reasoning report."""
        sa = self.scene_analysis
        ec = self.expert_collaboration
        kb = self.kb_anchoring
        rec = self.reconciliation
        conf = self.confidence_stars
        stars = "★" * conf + "☆" * (5 - conf)

        lines = [
            "=" * 60,
            "        地理定位推理报告 (Geo-MoE)",
            "=" * 60,
            "",
            f"可信度: {stars} ({conf}/5)",
            f"  原因: {self.confidence_reason}",
            "",
            "── 1. 场景分析 ──",
            f"  主导场景: {sa['primary_scene']} (router置信度 {sa['router_confidence']:.1%})",
            f"  场景分布:",
        ]
        for name, w in sa["all_weights"].items():
            bar = "█" * int(w * 50) + "░" * (50 - int(w * 50))
            lines.append(f"    {name:12s} {bar} {w:.1%}")

        lines.extend([
            "",
            "── 2. 专家协作 ──",
            f"  融合坐标: ({ec['fused_lat']:.4f}, {ec['fused_lng']:.4f})",
        ])
        for name, info in ec["experts"].items():
            lines.append(
                f"  {name:12s} weight={info['weight']:.2%} → "
                f"({info['lat']:.4f}, {info['lng']:.4f})"
            )

        lines.extend([
            "",
            "── 3. KB定位锚定 ──",
            f"  地点: {kb['city']}, {kb['country']}, {kb['continent']}",
            f"  坐标: ({kb['coordinates'][0]:.4f}, {kb['coordinates'][1]:.4f})",
            "",
            "── 4. GeoCoT推理摘要 ──",
            f"  {self.geocot_summary}",
            "",
            "── 5. 对账校验 ──",
        ])
        for item in rec.get("items", []):
            status_icon = "✔" if item.get("agree", False) else "✗"
            lines.append(f"  {status_icon} {item.get('dimension', '?')}: {item.get('detail', '')}")
        lines.append(f"  对账结果: {rec.get('verdict', '')}")

        lines.extend([
            "",
            f"最终预测: {self.final_coords[0]:.4f}, {self.final_coords[1]:.4f}",
            f"          ({self.final_location})",
            "=" * 60,
        ])
        return "\n".join(lines)


# ── Helper functions ────────────────────────────────────────────────────

def _summarize_geocot(text: str, max_chars: int = 200) -> str:
    """Extract a short summary from GeoCoT output."""
    if not text:
        return "无GeoCoT输出"

    # Find the key judgment line
    for keyword in ["判断:", "综合", "LOCATION:", "COORDINATES:"]:
        if keyword in text:
            idx = text.find(keyword)
            snippet = text[idx:idx + max_chars]
            return snippet.replace("\n", " ").strip()

    # Fallback: last few lines
    return text[-max_chars:].replace("\n", " ").strip()


def _build_reconciliation(
    router_weights: torch.Tensor,
    top_expert: int,
    geocot_output: str,
    fused_lat: float,
    fused_lng: float,
) -> dict:
    """Build reconciliation report between MoE router and GeoCoT reasoning."""
    items = []
    agree_count = 0
    total = 0

    # Dimension 1: Scene type agreement
    total += 1
    geocot_scene = _infer_scene_from_geocot(geocot_output)
    top_name = SCENE_TYPE_LABELS_CN.get(top_expert, str(top_expert))
    if geocot_scene is None:
        items.append({
            "dimension": "场景类型",
            "detail": f"MoE判定为{top_name}，GeoCoT无明确场景标签",
            "agree": True,
        })
        agree_count += 1
    elif geocot_scene == top_expert:
        items.append({
            "dimension": "场景类型",
            "detail": f"MoE + GeoCoT一致判定为{top_name}",
            "agree": True,
        })
        agree_count += 1
    else:
        geocot_name = SCENE_TYPE_LABELS_CN.get(geocot_scene, "未知")
        items.append({
            "dimension": "场景类型",
            "detail": f"MoE→{top_name}, GeoCoT→{geocot_name} (分歧，以MoE为准)",
            "agree": False,
        })

    # Dimension 2: Router concentration
    total += 1
    max_w = router_weights.max().item()
    if max_w > 0.8:
        items.append({
            "dimension": "路由集中度",
            "detail": f"高集中({max_w:.0%}) — 单一场景特征强烈",
            "agree": True,
        })
        agree_count += 1
    elif max_w > 0.5:
        items.append({
            "dimension": "路由集中度",
            "detail": f"中等({max_w:.0%}) — 混合场景特征",
            "agree": True,
        })
        agree_count += 1
    else:
        items.append({
            "dimension": "路由集中度",
            "detail": f"低({max_w:.0%}) — 场景特征模糊，可能需要更多训练样本",
            "agree": False,
        })

    # Dimension 3: Coordinate reasonableness
    total += 1
    if 18 <= fused_lat <= 54 and 73 <= fused_lng <= 135:
        items.append({
            "dimension": "坐标合理性",
            "detail": f"({fused_lat:.2f}, {fused_lng:.2f}) 在中国范围内，合理",
            "agree": True,
        })
        agree_count += 1
    else:
        items.append({
            "dimension": "坐标合理性",
            "detail": f"({fused_lat:.2f}, {fused_lng:.2f}) 可能在中国范围外",
            "agree": False,
        })

    agree_ratio = agree_count / total if total > 0 else 0
    if agree_ratio >= 0.8:
        verdict = "多维度一致，预测可信"
    elif agree_ratio >= 0.5:
        verdict = "部分维度一致，预测供参考"
    else:
        verdict = "多维度不一致，建议结合图像重新判断"

    return {"items": items, "agree_ratio": agree_ratio, "verdict": verdict}


def _infer_scene_from_geocot(text: str) -> Optional[int]:
    """Try to infer scene type from GeoCoT output text.

    Returns SceneType int or None if unclear.
    """
    if not text:
        return None
    text_lower = text.lower()

    # Urban indicators
    urban_keywords = ["城市", "建筑", "街道", "沥青", "路面", "urban", "city",
                      "road", "building", "traffic", "car", "shanghai", "beijing"]
    urban_score = sum(1 for kw in urban_keywords if kw in text_lower)

    # Karst/Granite indicators
    karst_keywords = ["峰林", "花岗岩", "喀斯特", "石灰岩", "石英砂岩",
                      "karst", "granite", "peak", "guilin", "huangshan",
                      "zhangjiajie", "sandstone"]
    karst_score = sum(1 for kw in karst_keywords if kw in text_lower)

    # Alpine indicators
    alpine_keywords = ["雪山", "冰川", "高山", "海拔", "alpine", "snow",
                       "mountain", "glacier", "四姑娘", "贡嘎", "梅里",
                       "plateau", "tibet"]
    alpine_score = sum(1 for kw in alpine_keywords if kw in text_lower)

    # Other indicators
    other_keywords = ["峡谷", "古镇", "草原", "沙漠", "湖泊", "gorge",
                      "old town", "grassland", "desert", "lake", "canyon",
                      "lijiang", "tiger leaping", "dali"]
    other_score = sum(1 for kw in other_keywords if kw in text_lower)

    scores = {
        SceneType.URBAN: urban_score,
        SceneType.KARST_GRANITE: karst_score,
        SceneType.ALPINE: alpine_score,
        SceneType.OTHER: other_score,
    }

    max_scene = max(scores, key=scores.get)
    if scores[max_scene] > 0:
        return int(max_scene)
    return None
