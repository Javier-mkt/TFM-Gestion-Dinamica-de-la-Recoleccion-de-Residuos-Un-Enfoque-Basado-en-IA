import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, GINEConv, global_mean_pool

# Se prepara los datos del grafo para introducirlos a la GNN

def tensorizacion_grafo(obs):
    nodos_indice = obs["grafo"]["nodos_indice"]
    aristas_indice = obs["grafo"]["aristas_indice"]

    # Nodos
    x = torch.tensor([
        [
            nodo["contenedor"],
            nodo["capacidad_contenedor"],
            nodo["llenado"],
            nodo["posicion_camion"],
            nodo["llenado_camion"]
        ]
        for nodo in nodos_indice.values()
    ], dtype = torch.float)


    # Aristas
    edge_index = torch.tensor([

        [arista["desde"] for arista in aristas_indice.values()],
        [arista["hasta"] for arista in aristas_indice.values()]

    ], dtype = torch.long)


    # Atributos aristas
    edge_attr = torch.tensor([
        [arista["distancia"], arista["tiempo_recorrido"]] # Ampliable a más elementos, como tiempo total recorrido (o concatenarlo en el forard de la GNN), velocidad_maxima, etc.
        for arista in aristas_indice.values()
    ], dtype = torch.float )

    return x, edge_index, edge_attr


class EncoderGNN(nn.Module):
    """
    Encoder GNN con GINEConv (edge_dim=in_edge_features).
    Mantiene edge_attr con coste ~O(E·hidden), sin la explosión de NNConv.
    """
    def __init__(self, in_node_features: int, in_edge_features: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()

        # Capa 1: F_in -> H
        mlp1 = nn.Sequential(
            nn.Linear(in_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers.append(GINEConv(mlp1, edge_dim=in_edge_features))

        # Capas siguientes: H -> H
        for _ in range(num_layers - 1):
            mlp_h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINEConv(mlp_h, edge_dim=in_edge_features))

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv in self.layers:
            h = F.relu(conv(h, edge_index, edge_attr))
        return h
    

class ActorCriticGNN(nn.Module):
    def __init__(self, hidden_dim, num_nodes, num_acciones_tipo = 2, mascara = True):
        super().__init__()
        self.mascara = mascara
        self.actor_tipo = nn.Linear(hidden_dim, num_acciones_tipo)
        self.actor_destino = nn.Linear(hidden_dim, num_nodes)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, h, batch, mascara_acciones = None):
        global_h = global_mean_pool(h, batch)

        # Actor
        tipo_logits = self.actor_tipo(global_h)
        destino_logits = self.actor_destino(h)
        
        # Máscaras
        if self.mascara and mascara_acciones is not None:
            if "tipo" in mascara_acciones:
                tipo_mascara_acciones = mascara_acciones["tipo"].to(tipo_logits.device)
                tipo_logits = tipo_logits.masked_fill(~tipo_mascara_acciones, -1e9)
            if "destino" in mascara_acciones:
                destino_mascara_acciones = mascara_acciones["destino"].to(destino_logits.device)
                destino_logits = destino_logits.masked_fill(~destino_mascara_acciones, -1e9)

        # Crítico
        value = self.critic(global_h)

        return tipo_logits, destino_logits, value
    
# https://stable-baselines3.readthedocs.io/en/master/guide/algos.html