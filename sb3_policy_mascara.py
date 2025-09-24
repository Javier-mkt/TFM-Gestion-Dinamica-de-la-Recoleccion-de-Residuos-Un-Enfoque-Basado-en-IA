# sb3_policy_mascara.py
from typing import Any, Dict, Tuple
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from modelo_gnn_final import EncoderGNN  # tu backbone GNN

class A2CPolicyGNNMasked(ActorCriticPolicy):
    """
    Observación esperada (Dict):
      - x:               (N, 5)  float32   ó (B, N, 5)
      - edge_index:      (2, E)  int64     ó (B, 2, E)
      - edge_attr:       (E, 2)  float32   ó (B, E, 2)
      - mascara_tipo:       (2,) int8      ó (B, 2)           (alias aceptado: tipo_mask)
      - mascara_destino:    (N,) int8      ó (B, N)           (alias aceptado: destino_mask)
      - mask2_table:     (2, N)  int8      ó (B, 2, N)

    Action space: MultiDiscrete([n_tipos, n_nodos]) con índices 0-based.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args,
        hidden_dim: int = 256,
        in_node_features: int = 5,
        in_edge_features: int = 2,
        n_tipos: int = 2,
        max_nodes: int | None = None,
        gnn_layers: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        if max_nodes is None:
            assert isinstance(action_space, spaces.MultiDiscrete), "Se espera MultiDiscrete para la acción."
            max_nodes = int(action_space.nvec[1])
        self.hidden_dim = hidden_dim
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.n_tipos = n_tipos
        self.max_nodes = max_nodes
        self.gnn_layers = gnn_layers

        # Backbone GNN + cabezas
        self.encoder = EncoderGNN(in_node_features, in_edge_features, hidden_dim, num_layers=gnn_layers)
        self.pi_tipo = nn.Linear(hidden_dim, n_tipos)   # logits por grafo (tipo)
        self.pi_dest = nn.Linear(hidden_dim, 1)         # 1 logit por nodo (destino)
        self.v_head  = nn.Linear(hidden_dim, 1)         # valor por grafo

        self.encoder.to(self.device)
        self.pi_tipo.to(self.device)
        self.pi_dest.to(self.device)
        self.v_head.to(self.device)

        # Desactivar heads por defecto del ActorCriticPolicy
        for p in self.mlp_extractor.parameters(): p.requires_grad = False
        for p in self.action_net.parameters(): p.requires_grad = False
        for p in self.value_net.parameters(): p.requires_grad = False

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                hidden_dim=self.hidden_dim,
                in_node_features=self.in_node_features,
                in_edge_features=self.in_edge_features,
                n_tipos=self.n_tipos,
                max_nodes=self.max_nodes,
                gnn_layers=self.gnn_layers,
            )
        )
        return data

    # ---------- utilidades ----------
    @staticmethod
    def _ensure_batch(obs_dict):
        import numpy as np
        import torch as th

        # Especificación esperada por clave (solo dtypes; shapes se validan en el forward)
        expected_np = {
            "x": np.float32,
            "edge_index": np.int64,
            "edge_attr": np.float32,
            "mascara_tipo": np.int8,
            "mascara_destino": np.int8,
            "mask2_table": np.int8,
        }
        to_torch = {np.float32: th.float32, np.int64: th.int64, np.int8: th.int8}

        out = {}
        for k, v in obs_dict.items():
            dt_np = expected_np.get(k, np.float32)
            dt_th = to_torch[dt_np]

            # 1) Si llega la CLASE (np.float32/np.int8/...) en vez de un valor/array → escalar 0
            if isinstance(v, type) and (
                v in (np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, bool, np.bool_)
            ):
                v = np.array(0, dtype=dt_np)

            # 2) Normaliza a tensor sin usar from_numpy
            if isinstance(v, th.Tensor):
                t = v.to(dtype=dt_th)
            elif isinstance(v, np.ndarray):
                if v.dtype != dt_np:
                    v = v.astype(dt_np, copy=False)
                t = th.tensor(v, dtype=dt_th)
            elif isinstance(v, (list, tuple)):
                t = th.tensor(v, dtype=dt_th)
            elif np.isscalar(v) or isinstance(v, (np.generic,)):
                t = th.tensor(v, dtype=dt_th)
            else:
                # objeto raro → tensor escalar 0
                t = th.tensor(0, dtype=dt_th)

            # 3) Añade batch-dim si falta
            if k in ("x", "edge_attr") and t.dim() == 2:
                t = t.unsqueeze(0)
            elif k == "edge_index" and t.dim() == 2:
                t = t.unsqueeze(0)
            elif k in ("mascara_tipo", "mascara_destino") and t.dim() == 1:
                t = t.unsqueeze(0)
            elif k == "mask2_table" and t.dim() == 2:
                t = t.unsqueeze(0)

            out[k] = t
        return out


    def _build_pyg_batch(self, obs: Dict[str, th.Tensor]) -> Tuple[Batch, th.Tensor]:
        """Construye Batch de PyG asumiendo grafo completo."""
        x_all  = obs["x"]              # (B, N, F)
        ei_all = obs["edge_index"]     # (B, 2, E)
        ea_all = obs["edge_attr"]      # (B, E, Fe)

        B = x_all.shape[0]
        data_list, n_valid = [], []

        for b in range(B):
            x_b = x_all[b].to(th.float32)     # (N,F)
            n_b = x_b.shape[0]
            n_valid.append(n_b)

            ei_b = ei_all[b].to(th.long)      # (2,E)
            ea_b = ea_all[b].to(th.float32)   # (E,Fe)

            src, dst = ei_b[0], ei_b[1]
            valid_e = (src >= 0) & (dst >= 0) & (src < n_b) & (dst < n_b)
            src = src[valid_e]
            dst = dst[valid_e]
            ea_b = ea_b[valid_e]

            data_list.append(Data(x=x_b, edge_index=th.stack([src, dst], 0), edge_attr=ea_b))

        batch = Batch.from_data_list(data_list)
        return batch, th.tensor(n_valid, dtype=th.long, device=batch.x.device)

    def _logits_and_value(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Devuelve: tipo_logits (B,2), destino_logits (B,N), values (B,)"""
        batch, n_valid = self._build_pyg_batch(obs)
        batch = batch.to(self.device)

        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)  # (sum_nodes, H)
        g = global_mean_pool(h, batch.batch)                          # (B, H)

        tipo_logits = self.pi_tipo(g)                                 # (B, 2)
        values = self.v_head(g).squeeze(-1)                           # (B,)

        node_logits = self.pi_dest(h).squeeze(-1)                     # (sum_nodes,)
        B = g.shape[0]
        destino_logits = th.full((B, self.max_nodes), -1e9, device=self.device)
        for b in range(B):
            idx = (batch.batch == b).nonzero(as_tuple=False).squeeze(1)
            k = int(n_valid[b].item())
            if k > 0:
                destino_logits[b, :k] = node_logits[idx][:k]
        return tipo_logits, destino_logits, values

    # ---------- SB3 API ----------
    def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
        if not isinstance(observation, dict):
            raise ValueError("Esta policy espera observaciones tipo Dict.")
        obs = {k: v.to(self.device) for k, v in self._ensure_batch(observation).items()}
        mascara_tipo = obs["mascara_tipo"].to(th.bool)   # (B,2)
        mask2_tbl    = obs["mask2_table"].to(th.bool)    # (B,2,N)

        tipo_logits, destino_logits, _ = self._logits_and_value(obs)

        eps = 1e-8
        mascara_tipo[(mascara_tipo.sum(dim=1) == 0), 0] = True
        dist_tipo = Categorical(logits=tipo_logits + (mascara_tipo.float() + eps).log())
        a_tipo = dist_tipo.mode if deterministic else dist_tipo.sample()

        b_idx = th.arange(a_tipo.shape[0], device=self.device)
        cond_mask = mask2_tbl[b_idx, a_tipo, :]          # (B,N)
        cond_mask[(cond_mask.sum(dim=1) == 0), 0] = True
        dist_dest = Categorical(logits=destino_logits + (cond_mask.float() + eps).log())
        a_dest = dist_dest.mode if deterministic else dist_dest.sample()

        actions = th.stack([a_tipo, a_dest], dim=1)      # (B,2)
        try:
            act_np = actions.detach().cpu().numpy()
        except Exception:
            act_np = np.asarray(actions.detach().cpu().tolist(), dtype=np.int64)

        return act_np, state

    def _predict(self, observation: Any, deterministic: bool = False):
        # SB3 llama a este camino con tensores ya en device (dict)
        obs = observation  # ya tensores
        mascara_tipo = obs["mascara_tipo"].to(th.bool)
        mask2_tbl    = obs["mask2_table"].to(th.bool)

        tipo_logits, destino_logits, _ = self._logits_and_value(obs)

        eps = 1e-8
        mascara_tipo[(mascara_tipo.sum(dim=1) == 0), 0] = True
        dist_tipo = Categorical(logits=tipo_logits + (mascara_tipo.float() + eps).log())
        a_tipo = dist_tipo.mode if deterministic else dist_tipo.sample()

        b_idx = th.arange(a_tipo.shape[0], device=self.device)
        cond_mask = mask2_tbl[b_idx, a_tipo, :]
        cond_mask[(cond_mask.sum(dim=1) == 0), 0] = True
        dist_dest = Categorical(logits=destino_logits + (cond_mask.float() + eps).log())
        a_dest = dist_dest.mode if deterministic else dist_dest.sample()

        actions = th.stack([a_tipo, a_dest], dim=1)
        return actions

    def evaluate_actions(self, obs: Dict[str, th.Tensor], actions: th.Tensor):
        # 1) Acciones en long (SB3 las pasa como float desde el buffer)
        actions = actions.long()
        a_tipo, a_dest = actions[:, 0], actions[:, 1]

        # 2) Máscaras al device correcto y en bool
        #    (usar el device de los logits para evitar mezclas CPU/GPU)
        tipo_logits_device = self.device  # o lo tomaremos tras _logits_and_value

        # 3) Logits y valor desde la GNN
        tipo_logits, destino_logits, values = self._logits_and_value(obs)
        tipo_logits_device = tipo_logits.device  # asegura el device real

        mascara_tipo = obs["mascara_tipo"].to(device=tipo_logits_device, dtype=th.bool)
        mask2_tbl    = obs["mask2_table"].to(device=tipo_logits_device, dtype=th.bool)

        # 4) Distribuciones con máscara
        eps = 1e-8

        # Tipo
        mascara_tipo[(mascara_tipo.sum(dim=1) == 0), 0] = True
        dist_tipo = Categorical(logits=tipo_logits + (mascara_tipo.float() + eps).log())

         # Destino condicionado por el tipo elegido
        b_idx = th.arange(a_tipo.shape[0], device=tipo_logits_device)
        cond_mask = mask2_tbl[b_idx, a_tipo, :]              # (B, N)
        cond_mask[(cond_mask.sum(dim=1) == 0), 0] = True
        dist_dest = Categorical(logits=destino_logits + (cond_mask.float() + eps).log())

        # 5) Log-prob conjunta y entropías
        log_prob = dist_tipo.log_prob(a_tipo) + dist_dest.log_prob(a_dest)
        entropy  = dist_tipo.entropy() + dist_dest.entropy()

        return values, log_prob, entropy
    
    def forward(self, obs: Dict[str, th.Tensor], deterministic: bool = False):
        """
        SB3 llama aquí durante learn().
        Devuelve: actions (B,2), values (B,), log_prob (B,)
        """
        mascara_tipo = obs["mascara_tipo"].to(th.bool)   # (B,2)
        mask2_tbl    = obs["mask2_table"].to(th.bool)    # (B,2,N)

        tipo_logits, destino_logits, values = self._logits_and_value(obs)

        eps = 1e-8
        mascara_tipo[(mascara_tipo.sum(dim=1) == 0), 0] = True
        dist_tipo = Categorical(logits=tipo_logits + (mascara_tipo.float() + eps).log())
        a_tipo = dist_tipo.mode if deterministic else dist_tipo.sample()

        b_idx = th.arange(a_tipo.shape[0], device=tipo_logits.device)
        cond_mask = mask2_tbl[b_idx, a_tipo, :]          # (B,N)
        cond_mask[(cond_mask.sum(dim=1) == 0), 0] = True
        dist_dest = Categorical(logits=destino_logits + (cond_mask.float() + eps).log())
        a_dest = dist_dest.mode if deterministic else dist_dest.sample()

        log_prob = dist_tipo.log_prob(a_tipo) + dist_dest.log_prob(a_dest)
        actions = th.stack([a_tipo, a_dest], dim=1)      # (B,2)
        return actions, values, log_prob


    def predict_values(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """
        SB3 llama aquí para el bootstrap del último valor en la recogida de rollouts.
        Debe devolver un tensor (B,) en el device correcto.
        """
        # obs ya viene tensorizado por SB3 (dict de tensores en self.device)
        _, _, values = self._logits_and_value(obs)
        return values

    
