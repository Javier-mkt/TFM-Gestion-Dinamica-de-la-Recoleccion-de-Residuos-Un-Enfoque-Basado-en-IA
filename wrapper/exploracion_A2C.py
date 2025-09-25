# Decaimineto entropía
from stable_baselines3.common.callbacks import BaseCallback


class DecaimientoEntropiaCallback(BaseCallback):
    """
    Decae linealmente el ent_coef de A2C desde `inicio` hasta `final`
    a lo largo de `total_steps`. Registra 'train/ent_coef' en cada paso.
    """
    def __init__(self, inicio: float, final: float, total_steps: int, verbose: int = 0):
        super().__init__(verbose)
        assert total_steps > 0, "total_steps debe ser > 0"
        self.start = float(inicio)
        self.end = float(final)
        self.total_steps = int(total_steps)

    def _on_training_start(self) -> None:
        # Fija explícitamente el valor inicial y deja rastro en los logs
        self.model.ent_coef = float(self.start)  # type: ignore[attr-defined]
        self.model.logger.record("train/ent_coef", float(self.model.ent_coef))  # type: ignore[attr-defined]

    def _on_step(self) -> bool:
        # t ∈ [0, total_steps]
        t = min(int(self.num_timesteps), self.total_steps)
        frac = t / self.total_steps
        cur = self.start + (self.end - self.start) * frac
        self.model.ent_coef = float(cur)  # type: ignore[attr-defined]
        # Log continuo para progress.csv / TensorBoard
        self.model.logger.record("train/ent_coef", float(self.model.ent_coef))  # type: ignore[attr-defined]
        return True

    def _on_training_end(self) -> None:
        # Asegura el valor final exacto al terminar
        self.model.ent_coef = float(self.end)  # type: ignore[attr-defined]
        self.model.logger.record("train/ent_coef", float(self.model.ent_coef))  # type: ignore[attr-defined]




'''
import gymnasium as gym
from collections import defaultdict
import numpy as np

class RecompensIntrinseca(gym.Wrapper):
    """
    Añade r_int = beta / sqrt(N_visitas[(nodo_actual, bin_carga)]).
    Expone info['r_int'] para logging en monitor_*.csv.
    """
    def __init__(self, env: gym.Env, beta: float = 0.05,
                 carga_bins=(0.0, 0.25, 0.5, 0.75, 1.01)):
        super().__init__(env)
        self.beta = float(beta)
        self.carga_bins = np.array(carga_bins, dtype=float)
        self.count = defaultdict(int)

    def reset(self, **kwargs):
        self.count.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_ext, term, trunc, info = self.env.step(action)

        # Lee estado del propio env (ya lo tienes)
        nid = int(getattr(self.env, "nodo_actual"))
        cap = float(getattr(self.env, "capacidad_camion", 1.0))
        carga = float(getattr(self.env, "carga_camion", 0.0)) / max(cap, 1e-6)

        b = int(np.digitize([carga], self.carga_bins)[0])
        key = (nid, b)
        self.count[key] += 1

        r_int = self.beta / (self.count[key] ** 0.5)
        r_tot = float(r_ext + r_int)

        # Para que quede en monitor CSV:
        info = info or {}
        info["r_ext"] = float(r_ext)
        info["r_int"] = float(r_int)
        info["r_tot"] = float(r_tot)

        return obs, r_tot, term, trunc, info
'''

import gymnasium as gym
import numpy as np
from collections import defaultdict

class RecompensIntrinsecaGlobal(gym.Wrapper):
    """
    Intrínseca global por nodo con decaimiento perezoso por EPISODIOS.
    Tabla compartida entre subprocesos vía multiprocessing.Manager.

    Variables compartidas:
      - shared_counts: dict proxy (nodo_id -> float)
      - shared_last_ep: dict proxy (nodo_id -> int)
      - shared_global_ep: Value('i') episodio global
    """
    def __init__(self, env: gym.Env, beta: float = 0.05, alpha: float = 0.9,
                 shared_counts = None, shared_last_ep = None, shared_global_ep = None):
        super().__init__(env)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.shared_counts = shared_counts
        self.shared_last_ep = shared_last_ep
        self.shared_global_ep = shared_global_ep
        self.eps = 1e-8

        # set local para no decaer más de una vez por nodo en el mismo episodio
        self.visited_this_ep = set()

    def reset(self, **kwargs):
        # nuevo episodio global (+1)
        if self.shared_global_ep is not None:
            self.shared_global_ep.value += 1
        self.visited_this_ep.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_ext, term, trunc, info = self.env.step(action)
        info = info or {}

        # Identifica nodo actual
        nid = int(getattr(self.env, "nodo_actual"))

        # Lee episodio global
        G = int(self.shared_global_ep.value) if self.shared_global_ep is not None else 0

        # Lee estado previo
        C = float(self.shared_counts.get(nid, 0.0)) if self.shared_counts is not None else 0.0
        L = int(self.shared_last_ep.get(nid, 0)) if self.shared_last_ep is not None else 0

        # Decaimiento perezoso solo una vez por episodio/nodo (primera visita del ep)
        if nid not in self.visited_this_ep and G > L:
            delta = G - L
            C = C * (self.alpha ** delta)

        # bump por visita
        C = C + 1.0

        # r_int
        r_int = self.beta / np.sqrt(C + self.eps)
        r_tot = float(r_ext + r_int)

        # Persistencia compartida
        if self.shared_counts is not None:
            self.shared_counts[nid] = C
        if self.shared_last_ep is not None:
            self.shared_last_ep[nid] = G

        # marca visita en este episodio
        self.visited_this_ep.add(nid)

        # logging
        info["r_ext"] = float(r_ext)
        info["r_int"] = float(r_int)
        info["r_tot"] = float(r_tot)
        info["novelty_C"] = float(C)
        info["novelty_last_ep"] = int(G)

        return obs, r_tot, term, trunc, info
