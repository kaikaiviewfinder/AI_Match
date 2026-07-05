"""Select 500 most valuable images for 8B-Thinking labeling.

Strategy:
1. Exclude already-labeled images (45 + 25)
2. Stratified sampling by region + scene type
3. Prioritize hard cases (North China plain, similar-looking cities)
4. Ensure geographic diversity
"""

import json, random, os
from collections import defaultdict
from pathlib import Path

random.seed(42)

GEOTAGGED = "D:/Geocomp/data/merged_geotagged.json"
EXISTING_45 = "D:/Geocomp/data/vlm_finetune/thinking_8b_v100_45.jsonl"
EXISTING_30 = "D:/Geocomp/data/vlm_finetune/thinking_30_test.jsonl"
OUTPUT_DIR = Path("D:/Geocomp/data/vlm_finetune/selected_500")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all geotagged images
with open(GEOTAGGED, encoding="utf-8") as f:
    all_images = json.load(f)
print(f"Total geotagged images: {len(all_images)}")

# Load existing labeled IDs
existing_ids = set()
for fpath in [EXISTING_45, EXISTING_30]:
    try:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                img = rec.get("image", "")
                fid = Path(img).stem
                existing_ids.add(fid)
    except:
        pass
print(f"Already labeled: {len(existing_ids)}")

# Filter to available, unlabeled images
available = {}
for path, coords in all_images.items():
    fid = Path(path).stem
    if fid not in existing_ids and Path(path).exists():
        available[path] = coords

print(f"Available unlabeled: {len(available)}")

# Classify by region
def classify_region(lat, lng):
    if lat > 40: return "northeast"
    if lat > 35: return "north_china_plain"
    if lat > 30: return "central_china"
    if lat > 25: return "south_china"
    return "southwest_tropical"

def classify_elevation(lat, lng):
    # rough elevation zones based on geography
    if lng < 100 and lat > 30:
        return "tibetan_plateau"
    if lng < 105 and lat < 30:
        return "yunnan_guizhou"
    if lng > 120 and lat > 30:
        return "eastern_coastal"
    if 110 < lng < 120 and 25 < lat < 35:
        return "central_hills"
    if lng < 90:
        return "xinjiang"
    return "other"

# Bin images by region + elevation zone
bins = defaultdict(list)
for path, (lat, lng) in available.items():
    region = classify_region(lat, lng)
    elev_zone = classify_elevation(lat, lng)
    key = f"{region}_{elev_zone}"
    bins[key].append((path, lat, lng))

print(f"\nGeographic distribution:")
for key in sorted(bins.keys()):
    print(f"  {key}: {len(bins[key])}")

# Allocation strategy: aim for proportional coverage but boost hard regions
# total=500
allocations = {
    "north_china_plain": 120,  # BOOST: current blind spot (北京/南京/西安)
    "northeast": 50,
    "central_china": 90,       # BOOST: diverse terrain
    "south_china": 80,
    "southwest_tropical": 60,
    "yunnan_guizhou": 50,
    "tibetan_plateau": 40,
    "eastern_coastal": 30,
    "xinjiang": 30,
    "other": 50,  # reserve
}

# Actually allocate per-combined-bin
# Simplify: group by latitude bands
selected = []

# Band 1: north_china_plain (35-40N) - hardest, biggest blind spot
band1 = [item for item in available.items() if 35 < item[1][0] <= 40]
random.shuffle(band1)
selected.extend(band1[:120])

# Band 2: above 40N (northeast, inner mongolia)
band2 = [item for item in available.items() if item[1][0] > 40]
random.shuffle(band2)
selected.extend(band2[:60])

# Band 3: 30-35N (central: 秦岭/武汉/上海)
band3 = [item for item in available.items() if 30 < item[1][0] <= 35]
random.shuffle(band3)
selected.extend(band3[:100])

# Band 4: 25-30N (south: 湖南/江西/福建/广东北)
band4 = [item for item in available.items() if 25 < item[1][0] <= 30]
random.shuffle(band4)
selected.extend(band4[:90])

# Band 5: below 25N (tropical south, Yunnan)
band5 = [item for item in available.items() if item[1][0] <= 25]
random.shuffle(band5)
selected.extend(band5[:70])

# Fill remaining to 500 with random diverse picks
remaining_needed = 500 - len(selected)
print(f"\nAfter band allocation: {len(selected)}, need {remaining_needed} more")

if remaining_needed > 0:
    already = {s[0] for s in selected}
    remaining = [(p, c) for p, c in available.items() if p not in already]
    random.shuffle(remaining)
    selected.extend(remaining[:remaining_needed])

print(f"Final selection: {len(selected)}")

# Save selection summary
os.makedirs(str(OUTPUT_DIR), exist_ok=True)

# Create symlink/copy list
manifest = []
for path, (lat, lng) in selected:
    manifest.append({"image": path, "lat": lat, "lng": lng})

with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

# Print stats
lats = [lat for _, (lat, _) in selected]
lngs = [lng for _, (_, lng) in selected]
print(f"\nGeographic stats:")
print(f"  Lat range: {min(lats):.1f} - {max(lats):.1f}")
print(f"  Lng range: {min(lngs):.1f} - {max(lngs):.1f}")
print(f"  Lat mean: {sum(lats)/len(lats):.1f}")

# Band distribution
band_counts = defaultdict(int)
for _, (lat, _) in selected:
    if lat > 40: band_counts["NE (>40N)"] += 1
    elif lat > 35: band_counts["N_Plain (35-40N)"] += 1
    elif lat > 30: band_counts["Central (30-35N)"] += 1
    elif lat > 25: band_counts["South (25-30N)"] += 1
    else: band_counts["Tropical (<25N)"] += 1
print("\nFinal band distribution:")
for k, v in sorted(band_counts.items()):
    print(f"  {k}: {v}")

# Copy to a JSONL for generate_thinking_labels.py
with open(OUTPUT_DIR / "selected_500.jsonl", "w", encoding="utf-8") as f:
    for m in manifest:
        json.dump(m, f, ensure_ascii=False)
        f.write("\n")

print(f"\nOutput: {OUTPUT_DIR}/selected_500.jsonl")
print(f"Manifest: {OUTPUT_DIR}/manifest.json")
