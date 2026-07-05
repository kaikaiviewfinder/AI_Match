"""Split 63 regions into ~250 sub-regions with differentiated fingerprints.

Each region is split into 4 sub-regions (quadrants around the center).
Sub-region fingerprints blend parent fingerprint with nearest external neighbor,
creating a spatial gradient that allows the scoring function to differentiate.

Also sub-divides sensor ranges so sensor affinity can help select sub-regions.
"""
import json, math, copy
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "src/regression/geo_ontology_v2.json"
OUTPUT = Path(__file__).resolve().parent.parent / "src/regression/geo_ontology_v3.json"

FIELDS = [
    "climate_zone", "terrain_type", "vegetation_zone", "urbanization",
    "architecture_style", "pavement_type", "language_script",
    "soil_color", "sky_quality", "mountain_rock_type",
    "tree_species", "building_height", "water_type",
    "landform_detail", "scene_type",
]


def haversine_km(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlng / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def offset_point(lat, lng, d_km, direction):
    """Offset (lat, lng) by d_km in a given direction (N/S/E/W)."""
    lat_offset = d_km / 111.32
    cos_lat = math.cos(math.radians(lat))
    lng_offset = d_km / (111.32 * max(cos_lat, 0.1))

    if direction == "NE":
        return lat + lat_offset, lng + lng_offset
    elif direction == "NW":
        return lat + lat_offset, lng - lng_offset
    elif direction == "SE":
        return lat - lat_offset, lng + lng_offset
    elif direction == "SW":
        return lat - lat_offset, lng - lng_offset
    return lat, lng


def blend_fingerprints(fp_a, fp_b, weight_b=0.12):
    """Blend two fingerprints: result = (1-w)*fp_a + w*fp_b."""
    result = {}
    all_fields = set(list(fp_a.keys()) + list(fp_b.keys()))
    for field in all_fields:
        result[field] = {}
        a_vals = fp_a.get(field, {})
        b_vals = fp_b.get(field, {})
        all_vals = set(list(a_vals.keys()) + list(b_vals.keys()))
        for val in all_vals:
            aw = a_vals.get(val, 0.0)
            bw = b_vals.get(val, 0.0)
            result[field][val] = round(aw * (1 - weight_b) + bw * weight_b, 4)
    return result


def main():
    with open(INPUT, encoding="utf-8") as f:
        onto = json.load(f)

    parents = onto["regions"]
    print(f"Input: {len(parents)} regions")

    # Build index for finding nearest external neighbors
    parent_centers = [(r["id"], r["center_lat"], r["center_lng"]) for r in parents]

    sub_regions = []
    sub_id = 1

    for parent in parents:
        pid = parent["id"]
        pname = parent["name"]
        plat, plng = parent["center_lat"], parent["center_lng"]
        pradius = parent["radius_km"]
        pfp = parent["fingerprint"]

        # Find the 2 nearest regions from DIFFERENT parents (for fingerprint blending)
        distances = []
        for rid, rlat, rlng in parent_centers:
            if rid == pid:
                continue
            d = haversine_km(plat, plng, rlat, rlng)
            distances.append((rid, d))
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:2]

        # Get fingerprints of nearest neighbors
        neighbor_fps = []
        for nid, _ in nearest_neighbors:
            for r in parents:
                if r["id"] == nid:
                    neighbor_fps.append(r["fingerprint"])
                    break

        # Create 4 sub-regions (quadrants)
        offset_dist = pradius * 0.35  # spread sub-centers ~70% of radius apart
        sub_centers = [
            offset_point(plat, plng, offset_dist, "NE"),
            offset_point(plat, plng, offset_dist, "NW"),
            offset_point(plat, plng, offset_dist, "SE"),
            offset_point(plat, plng, offset_dist, "SW"),
        ]

        # Sub-divide sensor ranges
        elev = parent.get("elevation_range", [0, 5000])
        temp = parent.get("temp_range", [-20, 40])
        humid = parent.get("humid_range", [10, 90])
        elev_mid = (elev[0] + elev[1]) / 2
        temp_mid = (temp[0] + temp[1]) / 2
        humid_mid = (humid[0] + humid[1]) / 2

        sub_sensor_ranges = [
            # NE: higher elevation, warmer (typically continental interior)
            [[elev_mid, elev[1]], [temp_mid, temp[1]], [humid[0], humid_mid]],
            # NW: higher elevation, cooler
            [[elev_mid, elev[1]], [temp[0], temp_mid], [humid[0], humid_mid]],
            # SE: lower elevation, warmer, more humid (typically coastal)
            [[elev[0], elev_mid], [temp_mid, temp[1]], [humid_mid, humid[1]]],
            # SW: lower elevation, cooler, more humid
            [[elev[0], elev_mid], [temp[0], temp_mid], [humid_mid, humid[1]]],
        ]

        for i, (slat, slng) in enumerate(sub_centers):
            # Blend fingerprint: mostly parent, slight influence from nearest neighbor
            if neighbor_fps:
                blend_from = neighbor_fps[i % len(neighbor_fps)]
            else:
                blend_from = pfp
            sfp = blend_fingerprints(pfp, blend_from, weight_b=0.10)

            sub = {
                "id": f"r{sub_id:03d}",
                "name": f"{pname}-{['东北','西北','东南','西南'][i]}",
                "parent_id": pid,
                "center_lat": round(slat, 4),
                "center_lng": round(slng, 4),
                "radius_km": round(pradius * 0.6),
                "elevation_range": list(map(int, sub_sensor_ranges[i][0])),
                "temp_range": list(map(int, sub_sensor_ranges[i][1])),
                "humid_range": list(map(int, sub_sensor_ranges[i][2])),
                "fingerprint": sfp,
            }
            sub_regions.append(sub)
            sub_id += 1

    # ── Compute adjacency ──────────────────────────────────────────────
    adjacency = {}
    for sr in sub_regions:
        sid = sr["id"]
        adj = []
        for sr2 in sub_regions:
            if sr2["id"] == sid:
                continue
            d = haversine_km(
                sr["center_lat"], sr["center_lng"],
                sr2["center_lat"], sr2["center_lng"])
            # Adjacent if within 2x the avg radius
            threshold = (sr["radius_km"] + sr2["radius_km"]) * 1.5
            if d < threshold:
                adj.append(sr2["id"])
        adjacency[sid] = adj

    adj_count = sum(len(v) for v in adjacency.values())
    print(f"Output: {len(sub_regions)} sub-regions, {adj_count} adjacency edges")

    # ── Build output ──────────────────────────────────────────────────
    output = {
        "field_hierarchy": onto.get("field_hierarchy", {}),
        "anchor_fields": onto.get("anchor_fields", []),
        "constraints": onto.get("constraints", {}),
        "adjacency": adjacency,
        "regions": sub_regions,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
