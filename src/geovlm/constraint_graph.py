"""Constraint Graph Layer: 16-node GNN enforcing physical/geographic constraints.

Each node = one geographic field. Edges encode physical constraints.

2 rounds of GraphSAGE-style message passing:
  Round 1: collect neighbor info -> detect prediction contradictions
  Round 2: update based on Round 1 results

Edge weights are LEARNABLE — model discovers constraint strength from data.
"""
import torch
import torch.nn as nn


# Predefined constraint edges (bidirectional, learned weight)
# Indices: 0=climate 1=terrain 2=vegetation 3=urbanization
#          4=architecture 5=pavement 6=language 7=visible_text
#          8=soil_color 9=sky_quality 10=mountain_rock 11=tree_species
#          12=building_height 13=water_type 14=landform_detail 15=scene_type
_CONSTRAINT_EDGES = [
    # Original edges
    (0, 2),   # climate <-> vegetation
    (1, 2),   # terrain <-> vegetation
    (0, 1),   # climate <-> terrain
    (3, 4),   # urbanization <-> architecture
    (3, 5),   # urbanization <-> pavement
    (4, 5),   # architecture <-> pavement
    (6, 7),   # language_script <-> visible_text
    (3, 7),   # urbanization <-> visible_text
    # New edges
    (0, 8),   # climate <-> soil_color (pedology)
    (2, 8),   # vegetation <-> soil_color
    (0, 9),   # climate <-> sky_quality
    (1, 10),  # terrain <-> mountain_rock_type (geology)
    (14, 10), # landform_detail <-> mountain_rock_type
    (0, 11),  # climate <-> tree_species (ecology)
    (2, 11),  # vegetation <-> tree_species
    (3, 12),  # urbanization <-> building_height
    (1, 13),  # terrain <-> water_type
    (1, 14),  # terrain <-> landform_detail
    (1, 15),  # terrain <-> scene_type
    (2, 15),  # vegetation <-> scene_type
]

FIELD_NAMES = [
    "climate_zone", "terrain_type", "vegetation_zone", "urbanization",
    "architecture_style", "pavement_type", "language_script", "visible_text",
    "soil_color", "sky_quality", "mountain_rock_type", "tree_species",
    "building_height", "water_type", "landform_detail", "scene_type",
]


class ConstraintGraphLayer(nn.Module):
    """2-round GraphSAGE message passing over 16 field nodes."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = len(FIELD_NAMES)

        # Build adjacency from predefined constraint edges (bidirectional)
        edge_index = []
        for src, dst in _CONSTRAINT_EDGES:
            edge_index.extend([(src, dst), (dst, src)])
        self.register_buffer(
            "edge_index",
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

        # Learnable edge weights (one scalar per directed edge)
        num_edges = len(edge_index)
        self.edge_weights = nn.Parameter(torch.ones(num_edges) * 0.5)

        # Message functions (round 1 and 2)
        self.msg_linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.msg_linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Update functions (GRU cells for iterative refinement)
        self.update_gru1 = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_gru2 = nn.GRUCell(hidden_dim, hidden_dim)

        # Global consistency score head
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_nodes, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_features: torch.Tensor) -> tuple:
        """Args:  node_features [B, 8, hidden_dim] from Q-Former output
           Returns: (updated_features [B, 8, hidden_dim], consistency [B, 1])
        """
        B = node_features.shape[0]
        device = node_features.device

        # ---- Round 1: Detect contradictions via neighbor messages ----
        messages1 = torch.zeros(B, self.num_nodes, self.hidden_dim, device=device)
        msg_counts = torch.zeros(B, self.num_nodes, 1, device=device)

        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            src_feat = node_features[:, src, :]
            dst_feat = node_features[:, dst, :]
            weight = self.edge_weights[i].sigmoid()  # [0, 1]

            concat = torch.cat([src_feat, dst_feat], dim=-1)
            msg = self.msg_linear1(concat) * weight
            messages1[:, dst, :] += msg
            msg_counts[:, dst, 0] += 1

        # Average messages
        msg_counts = msg_counts.clamp(min=1)
        messages1 = messages1 / msg_counts

        # GRU update round 1
        flat_nodes = node_features.reshape(B * self.num_nodes, self.hidden_dim)
        flat_msgs = messages1.reshape(B * self.num_nodes, self.hidden_dim)
        updated1 = self.update_gru1(flat_msgs, flat_nodes)
        updated1 = updated1.view(B, self.num_nodes, self.hidden_dim)

        # ---- Round 2: Refine based on round-1 updated features ----
        messages2 = torch.zeros(B, self.num_nodes, self.hidden_dim, device=device)

        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            src_feat = updated1[:, src, :]
            dst_feat = updated1[:, dst, :]
            weight = self.edge_weights[i].sigmoid()

            concat = torch.cat([src_feat, dst_feat], dim=-1)
            msg = self.msg_linear2(concat) * weight
            messages2[:, dst, :] += msg

        messages2 = messages2 / msg_counts

        flat_nodes2 = updated1.reshape(B * self.num_nodes, self.hidden_dim)
        flat_msgs2 = messages2.reshape(B * self.num_nodes, self.hidden_dim)
        updated2 = self.update_gru2(flat_msgs2, flat_nodes2)
        updated2 = updated2.view(B, self.num_nodes, self.hidden_dim)

        # Global consistency score from all node features
        flat_all = updated2.reshape(B, -1)
        consistency = self.consistency_head(flat_all)

        return updated2, consistency
