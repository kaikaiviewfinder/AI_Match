"""Manual image labeling tool for GeoCoT training data.

Opens each image in the system viewer and asks structured questions
about VISIBLE geographic elements. Saves progress after each image.

Usage:
    python scripts/manual_label_tool.py                    # start/resume from beginning
    python scripts/manual_label_tool.py --start 50         # resume from image 50
"""

import argparse
import json
import os
import sys
from pathlib import Path


QUESTIONS = [
    # (key, question, options)
    # options: None = free text, list = choose one, "multi" = choose multiple
    ("scene_type", "场景类型？", [
        "1. 城市街道 (city_street)",
        "2. 山路/盘山路 (mountain_road)",
        "3. 高速公路 (highway)",
        "4. 草原 (grassland)",
        "5. 沙漠/戈壁 (desert)",
        "6. 雪山/冰川 (alpine_snow)",
        "7. 喀斯特峰林 (karst)",
        "8. 丹霞/红层 (danxia)",
        "9. 海边/海岸 (coastal)",
        "10. 河流/河谷 (river_valley)",
        "11. 森林 (forest)",
        "12. 农田/耕地 (farmland)",
        "13. 小镇/乡村 (rural_village)",
        "14. 其他 (other)",
    ]),
    ("terrain_type", "地形？", [
        "1. 平坦 (flat)",
        "2. 丘陵 (rolling_hills)",
        "3. 山地 (mountains)",
        "4. 高原台地 (plateau)",
        "5. 峡谷 (canyon)",
        "6. 海岸 (coastal)",
    ]),
    ("vegetation_zone", "植被？", [
        "1. 无植被/稀疏 (sparse)",
        "2. 草原 (grassland)",
        "3. 针叶林 (coniferous_forest)",
        "4. 阔叶林 (broadleaf_forest)",
        "5. 混交林 (mixed_forest)",
        "6. 热带雨林 (rainforest)",
        "7. 沙漠灌丛 (desert_scrub)",
        "8. 农田 (cropland)",
        "9. 竹林 (bamboo)",
    ]),
    ("urbanization", "城市化程度？", [
        "1. 大都市 (metropolis)",
        "2. 城市 (city)",
        "3. 小镇 (small_town)",
        "4. 乡村 (rural)",
        "5. 野外无人区 (wilderness)",
    ]),
    ("visible_water", "能看到水体吗？", [
        "1. 看不到 (none)",
        "2. 河流 (river)",
        "3. 湖泊 (lake)",
        "4. 海洋 (ocean)",
        "5. 瀑布 (waterfall)",
        "6. 冰川 (glacier)",
    ]),
    ("architecture_style", "建筑风格？（看不到建筑填'无建筑'）", [
        "1. 无建筑 (none)",
        "2. 现代玻璃幕墙 (modern_glass)",
        "3. 现代住宅小区 (modern_residential)",
        "4. 老旧居民楼 (old_residential)",
        "5. 徽派白墙黑瓦 (hui_style)",
        "6. 藏式石砌 (tibetan_stone)",
        "7. 四合院/胡同 (courtyard)",
        "8. 骑楼 (arcade)",
        "9. 窑洞 (cave_dwelling)",
        "10. 傣式竹楼 (dai_bamboo)",
        "11. 石库门 (shikumen)",
        "12. 苏联式/工厂 (soviet_industrial)",
    ]),
    ("visible_text", "能看到文字/标识吗？（路牌、店招、标语等）", [
        "1. 看不到文字 (none)",
        "2. 简体中文 (simplified_chinese)",
        "3. 繁体中文 (traditional_chinese)",
        "4. 藏文 (tibetan)",
        "5. 维吾尔文 (uyghur)",
        "6. 中英双语 (bilingual_cn_en)",
    ]),
    ("sky_quality", "天空状况？", [
        "1. 晴朗蓝天 (clear_blue)",
        "2. 多云 (cloudy)",
        "3. 阴天灰白 (overcast_grey)",
        "4. 灰霾 (hazy)",
        "5. 沙尘黄 (dusty_yellow)",
        "6. 大雾 (thick_fog)",
    ]),
    ("distinctive_rock", "能看到独特岩石/地质吗？", [
        "1. 看不到 (none)",
        "2. 花岗岩球形风化 (granite_spheroidal)",
        "3. 石灰岩峰林 (limestone_karst)",
        "4. 红色砂岩丹霞 (red_sandstone_danxia)",
        "5. 黄土沟壑 (loess_gully)",
        "6. 沙丘 (sand_dune)",
        "7. 玄武岩柱状节理 (basalt_columnar)",
        "8. 其他 (other)",
    ]),
    ("description", "用一句话描述这张图里看到了什么（中文）", None),
]


def parse_answer(text, options):
    """Parse user input into a value string."""
    text = text.strip()
    # Try number
    try:
        idx = int(text) - 1
        if 0 <= idx < len(options):
            # Extract the key from the option string
            opt = options[idx]
            if '(' in opt and ')' in opt:
                return opt.split('(')[1].split(')')[0]
            return opt
    except ValueError:
        pass
    # If user typed a key directly
    for opt in options:
        if '(' in opt and ')' in opt:
            key = opt.split('(')[1].split(')')[0]
            if text.lower() == key.lower():
                return key
    # Return as-is (free text or unrecognized)
    return text


def main():
    parser = argparse.ArgumentParser(description="Manual image labeling tool")
    parser.add_argument("--start", type=int, default=0, help="Start from image N")
    parser.add_argument("--input", type=str,
                        default="data/vlm_finetune/manual_labeling/to_label.json")
    parser.add_argument("--output", type=str,
                        default="data/vlm_finetune/manual_labeling/labels.jsonl")
    args = parser.parse_args()

    # Load image list
    with open(args.input, "r", encoding="utf-8") as f:
        images = json.load(f)

    print(f"共 {len(images)} 张图片待标注")
    print(f"从第 {args.start + 1} 张开始")
    print()
    print("=" * 60)
    print("标注说明：")
    print("  - 每个问题输入数字选择，或直接输入选项关键词")
    print("  - 只标注图片里【能看到】的，不确定就选'看不到/无'")
    print("  - 输入 q 保存并退出，随时可以 --start 继续")
    print("  - 每张图标注完自动保存，不用担心丢失")
    print("=" * 60)
    print()

    # Load existing progress
    existing = []
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
        print(f"已有 {len(existing)} 条标注记录")

    existing_images = {e["image"] for e in existing}

    for i, item in enumerate(images):
        if i < args.start:
            continue

        img_path = item["image"]
        if img_path in existing_images:
            continue

        print(f"\n{'─' * 50}")
        print(f"[{i+1}/{len(images)}] {Path(img_path).name}")
        print(f"GPS: ({item['lat']:.4f}, {item['lng']:.4f})")
        print(f"{'─' * 50}")

        # Open image in system viewer
        abs_path = Path(img_path)
        if abs_path.exists():
            os.startfile(str(abs_path))
        else:
            # Try relative to geocomp root
            alt_path = Path("D:/Geocomp") / img_path
            if alt_path.exists():
                os.startfile(str(alt_path))
            else:
                print(f"  !! 找不到图片: {img_path}")

        # Ask questions
        answers = {}
        skip = False
        for key, question, options in QUESTIONS:
            print(f"\n  {question}")
            if options:
                for opt in options:
                    print(f"    {opt}")
                ans = input(f"  > ").strip()
            else:
                print(f"  （自由输入）")
                ans = input(f"  > ").strip()

            if ans.lower() == 'q':
                skip = True
                break

            if options:
                answers[key] = parse_answer(ans, options)
            else:
                answers[key] = ans

        if skip:
            print("\n保存并退出...")
            break

        # Save
        record = {
            "image": img_path,
            "lat": item["lat"],
            "lng": item["lng"],
            **answers,
        }
        existing.append(record)
        existing_images.add(img_path)

        with open(args.output, "w", encoding="utf-8") as f:
            for rec in existing:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\n  ✓ 已保存 ({len(existing)} 条)")

    print(f"\n{'=' * 60}")
    print(f"完成！共标注 {len(existing)} 张")
    print(f"保存在: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
