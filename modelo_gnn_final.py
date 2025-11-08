import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GINEConv, global_mean_pool
from torch_geometric.nn import GINEConv, JumpingKnowledge

class EncoderGNN(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_dim: int,
                 num_layers: int, dropout: float = 0.1, jk_mode: str = "max"):
        """
        jk_mode:
          - "max": recomendado (mantiene dimensión H)
          - "cat": concatena capas (dim = H * L) -> requeriría ajustar las cabezas aguas arriba
          - "sum" o "last": alternativas
        """
        super().__init__()
        assert num_layers >= 1, "num_layers debe ser >= 1"
        self.dropout = nn.Dropout(dropout)
        self.jk_mode = jk_mode

        # Proyección inicial de nodos
        self.node_proj = nn.Linear(in_node_features, hidden_dim)

        # Capas GINE + LayerNorm (residual post-norm)
        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(mlp, edge_dim=in_edge_features))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Jumping Knowledge
        # "max" mantiene la dimensión -> compatible con tus cabezas actuales
        self.jk = JumpingKnowledge(jk_mode)

    def forward(self, x, edge_index, edge_attr):
        h = self.node_proj(x)
        hs = []  # guardamos la salida de cada capa

        for conv, ln in zip(self.layers, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h = ln(h + h_new)   # residual + norm
            h = F.relu(h)
            h = self.dropout(h)
            hs.append(h)

        # Combina representaciones de todas las capas
        h_out = self.jk(hs)  # "max" => [N, H]; "cat" => [N, H * L]
        return h_out
