"""TransE-based KG embedding for region fingerprint similarity.

Builds a knowledge graph from the ontology regions, trains TransE embeddings,
and produces region vectors for semantic similarity scoring in ontology_fusion.

Key idea: regions with similar field profiles get close embedding vectors,
enabling soft matching beyond discrete fingerprint overlap.
"""
import json
import math
import os
import pickle
from pathlib import Path
import numpy as np

ONTOLOGY_PATH = Path(__file__).resolve().parent / "geo_ontology_v3.json"
EMBEDDING_PATH = Path(__file__).resolve().parent / "kg_embeddings_v3.pkl"

FIELD_NAMES = [
    "climate_zone", "terrain_type", "vegetation_zone", "urbanization",
    "architecture_style", "pavement_type", "language_script",
    "soil_color", "sky_quality", "mountain_rock_type",
    "tree_species", "building_height", "water_type",
    "landform_detail", "scene_type",
]

RELATION_PREFIX = "has_"
ADJACENCY_REL = "adjacent_to"


def build_kg(ontology: dict) -> tuple[list[tuple[str, str, str]], dict[str, int], dict[str, int]]:
    """Build knowledge graph triples and entity/relation indices from ontology.

    Returns:
        triples: list of (head, relation, tail) string triples
        entity2idx: entity name → integer index
        relation2idx: relation name → integer index
    """
    triples = []
    entities = set()
    relations = set()

    regions = ontology["regions"]
    adjacency = ontology.get("adjacency", {})

    for region in regions:
        rid = region["id"]
        entities.add(rid)

        # Region → field_value triples
        fp = region.get("fingerprint", {})
        for field in FIELD_NAMES:
            rel = RELATION_PREFIX + field
            relations.add(rel)
            field_dist = fp.get(field, {})
            for value, weight in field_dist.items():
                entities.add(value)
                if weight >= 0.05:  # filter very low-weight entries
                    triples.append((rid, rel, value))

    # Adjacency triples
    relations.add(ADJACENCY_REL)
    for rid, neighbors in adjacency.items():
        for nid in neighbors:
            triples.append((rid, ADJACENCY_REL, nid))

    # Build indices
    entity_list = sorted(entities)
    entity2idx = {e: i for i, e in enumerate(entity_list)}
    relation_list = sorted(relations)
    relation2idx = {r: i for i, r in enumerate(relation_list)}

    return triples, entity2idx, relation2idx


class TransE:
    """Bilinear TransE model: score(h, r, t) = -||h + r - t||_2"""

    def __init__(self, n_entities: int, n_relations: int, dim: int = 128,
                 margin: float = 1.0, lr: float = 0.01, l2_reg: float = 0.001):
        self.dim = dim
        self.margin = margin
        self.lr = lr
        self.l2_reg = l2_reg
        self.n_entities = n_entities
        self.n_relations = n_relations

        # Xavier init
        bound = 6.0 / math.sqrt(dim)
        self.entity_emb = np.random.uniform(-bound, bound, (n_entities, dim))
        self.relation_emb = np.random.uniform(-bound, bound, (n_relations, dim))
        # Normalize entity embeddings
        self.entity_emb = self._normalize(self.entity_emb)

    @staticmethod
    def _normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return x / norms

    def train(self, triples: list[tuple[int, int, int]],
              epochs: int = 200, batch_size: int = 256, verbose: bool = True):
        """Train TransE with negative sampling."""
        n_triples = len(triples)
        if n_triples == 0:
            return

        # Pre-convert to numpy for fast access
        triples_arr = np.array(triples, dtype=np.int32)
        n_batches = max(1, n_triples // batch_size)

        for epoch in range(epochs):
            np.random.shuffle(triples_arr)
            total_loss = 0.0

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_triples)
                batch = triples_arr[start:end]

                h_idx = batch[:, 0]
                r_idx = batch[:, 1]
                t_idx = batch[:, 2]

                # Positive scores
                h = self.entity_emb[h_idx]
                r = self.relation_emb[r_idx]
                t = self.entity_emb[t_idx]
                pos_scores = -np.linalg.norm(h + r - t, axis=1)

                # Negative samples (corrupt tail)
                neg_t = np.random.randint(0, self.n_entities, size=len(batch))
                neg_t_scores = -np.linalg.norm(h + r - self.entity_emb[neg_t], axis=1)

                # Hinge loss
                loss = np.maximum(0, self.margin - pos_scores + neg_t_scores)
                total_loss += loss.sum()

                # Gradient: d(loss)/d(score) for violated constraints
                violated = loss > 0
                if not violated.any():
                    continue

                # Gradient for violated triples
                h_v = h[violated]
                r_v = r[violated]
                t_v = t[violated]
                neg_t_v = self.entity_emb[neg_t[violated]]

                # d(pos)/d(h), d(pos)/d(r), d(pos)/d(t)  for L2 norm
                # d(-||h+r-t||) = -(h+r-t)/||h+r-t||
                pos_diff = h_v + r_v - t_v
                pos_norm = np.linalg.norm(pos_diff, axis=1, keepdims=True)
                pos_norm = np.maximum(pos_norm, 1e-12)
                pos_grad = -pos_diff / pos_norm  # gradient pushes h+r toward t

                neg_diff = h_v + r_v - neg_t_v
                neg_norm = np.linalg.norm(neg_diff, axis=1, keepdims=True)
                neg_norm = np.maximum(neg_norm, 1e-12)
                neg_grad = neg_diff / neg_norm  # gradient pushes h+r away from neg_t

                # Update
                lr = self.lr
                grad_h = lr * (pos_grad - neg_grad)
                grad_r = lr * (pos_grad - neg_grad)
                grad_t_pos = lr * (-pos_grad)
                grad_t_neg = lr * neg_grad

                # Accumulate (handle repeated entity indices in batch)
                h_indices = h_idx[violated]
                r_indices = r_idx[violated]
                t_indices = t_idx[violated]
                n_indices = neg_t[violated]

                for i in range(len(h_indices)):
                    self.entity_emb[h_indices[i]] -= grad_h[i]
                    self.relation_emb[r_indices[i]] -= grad_r[i]
                    self.entity_emb[t_indices[i]] -= grad_t_pos[i]
                    self.entity_emb[n_indices[i]] -= grad_t_neg[i]

                # L2 regularization
                self.entity_emb -= lr * self.l2_reg * self.entity_emb
                self.relation_emb -= lr * self.l2_reg * self.relation_emb
                self.entity_emb = self._normalize(self.entity_emb)

            if verbose and (epoch + 1) % 50 == 0:
                avg_loss = total_loss / n_triples
                print(f"  Epoch {epoch+1}/{epochs}: avg loss = {avg_loss:.4f}")

    def get_entity_vec(self, idx: int) -> np.ndarray:
        return self.entity_emb[idx]

    def get_relation_vec(self, idx: int) -> np.ndarray:
        return self.relation_emb[idx]


def compute_region_embedding(region: dict, entity2idx: dict, model: TransE) -> np.ndarray:
    """Compute region embedding as mean of its top field value embeddings."""
    vecs = []
    fp = region.get("fingerprint", {})
    for field in FIELD_NAMES:
        field_dist = fp.get(field, {})
        if not field_dist:
            continue
        # Take top-2 values per field weighted by fingerprint weight
        for value, weight in sorted(field_dist.items(), key=lambda x: -x[1])[:2]:
            if value in entity2idx:
                vecs.append(model.get_entity_vec(entity2idx[value]) * weight)
    if not vecs:
        return np.zeros(model.dim)
    return np.mean(vecs, axis=0)


def build_query_embedding(elements: dict, entity2idx: dict, model: TransE) -> np.ndarray:
    """Build embedding for a query from field predictions."""
    vecs = []
    weights = []
    for field in FIELD_NAMES:
        val = elements.get(field)
        if isinstance(val, dict):
            conf = val.get("confidence", 0.5)
            val = val.get("value", "UNKNOWN")
        else:
            conf = 0.5
            val = str(val) if val else "UNKNOWN"

        if val == "UNKNOWN" or val not in entity2idx:
            continue
        vecs.append(model.get_entity_vec(entity2idx[val]) * conf)
        weights.append(conf)
    if not vecs:
        return np.zeros(model.dim)
    return np.average(vecs, axis=0, weights=weights)


def train_and_save(ontology_path: str | None = None, dim: int = 128, epochs: int = 300):
    """Train KG embeddings and save to embedding file."""
    onto_path = Path(ontology_path) if ontology_path else ONTOLOGY_PATH
    with open(onto_path, encoding="utf-8") as f:
        ontology = json.load(f)

    print(f"Building KG from {len(ontology['regions'])} regions...")
    triples_str, entity2idx, relation2idx = build_kg(ontology)
    print(f"  Entities: {len(entity2idx)}, Relations: {len(relation2idx)}")
    print(f"  Triples: {len(triples_str)}")

    # Convert to indices
    triples_idx = []
    for h, r, t in triples_str:
        if h in entity2idx and r in relation2idx and t in entity2idx:
            triples_idx.append((entity2idx[h], relation2idx[r], entity2idx[t]))

    print(f"  Valid triples: {len(triples_idx)}")

    # Train TransE
    model = TransE(
        n_entities=len(entity2idx),
        n_relations=len(relation2idx),
        dim=dim,
        margin=1.0,
        lr=0.01,
        l2_reg=0.0005,
    )
    print(f"Training TransE (dim={dim}, epochs={epochs})...")
    model.train(triples_idx, epochs=epochs, batch_size=256, verbose=True)

    # Compute region embeddings
    idx2entity = {v: k for k, v in entity2idx.items()}
    region_embeddings = {}
    for region in ontology["regions"]:
        rid = region["id"]
        if rid in entity2idx:
            # Combine learned entity embedding + field-mean embedding
            learned = model.get_entity_vec(entity2idx[rid])
            field_mean = compute_region_embedding(region, entity2idx, model)
            region_embeddings[rid] = ((learned + field_mean) / 2.0).tolist()

    # Compute field value embeddings for query building
    field_value_embeddings = {}
    for entity, idx in entity2idx.items():
        field_value_embeddings[entity] = model.get_entity_vec(idx).tolist()

    # Save
    output = {
        "dim": dim,
        "region_embeddings": region_embeddings,
        "field_value_embeddings": field_value_embeddings,
        "entity2idx": entity2idx,
        "relation2idx": relation2idx,
        "field_names": FIELD_NAMES,
    }
    output_path = EMBEDDING_PATH
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Embeddings saved to: {output_path}")
    print(f"  Region embeddings: {len(region_embeddings)}")
    print(f"  Field value embeddings: {len(field_value_embeddings)}")

    # Quick sanity check
    _sanity_check(ontology, region_embeddings, entity2idx)


def _sanity_check(ontology, region_embeddings, entity2idx):
    """Verify embeddings make geographic sense."""
    regions = ontology["regions"]
    rid_to_name = {r["id"]: r["name"] for r in regions}
    rids = list(region_embeddings.keys())
    if len(rids) < 2:
        return

    # Check: adjacent regions should have higher cosine similarity
    adjacency = ontology.get("adjacency", {})
    adj_sims = []
    non_adj_sims = []
    rng = np.random.RandomState(42)

    for rid, vec in region_embeddings.items():
        neighbors = adjacency.get(rid, [])
        v = np.array(vec)
        for nid in neighbors:
            if nid in region_embeddings:
                nv = np.array(region_embeddings[nid])
                sim = np.dot(v, nv) / (np.linalg.norm(v) * np.linalg.norm(nv) + 1e-12)
                adj_sims.append(sim)
        # Random non-adjacent
        non_neighbors = [r for r in rids if r != rid and r not in neighbors]
        if non_neighbors:
            samples = rng.choice(non_neighbors, min(3, len(non_neighbors)), replace=False)
            for nid in samples:
                nv = np.array(region_embeddings[nid])
                sim = np.dot(v, nv) / (np.linalg.norm(v) * np.linalg.norm(nv) + 1e-12)
                non_adj_sims.append(sim)

    if adj_sims and non_adj_sims:
        print(f"\n  Sanity: adjacent cos_sim={np.mean(adj_sims):.3f}, "
              f"non-adjacent cos_sim={np.mean(non_adj_sims):.3f}")
        if np.mean(adj_sims) > np.mean(non_adj_sims):
            print("  [OK] Adjacent regions have higher embedding similarity")
        else:
            print("  [WARN] Adjacent regions NOT more similar -- may need more training")


def load_embeddings() -> dict:
    """Load pre-trained KG embeddings."""
    with open(EMBEDDING_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    train_and_save()
