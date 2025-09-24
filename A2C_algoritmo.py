# A2C_algoritmo.py
import os
import json
import time
import multiprocessing as mp
from typing import Optional, Dict, Any

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from sb3_policy_mascara import A2CPolicyGNNMasked
from env_basuras_final import RecogidaBasurasEnv


# ---------------------------
# Helpers de guardado a CARPETA
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _save_model_to_folder(model: A2C, folder: str, meta: Optional[Dict[str, Any]] = None):
    """
    Guarda:
      - policy_state.pt            -> model.policy.state_dict()
      - optimizer_state.pt         -> model.policy.optimizer.state_dict()
      - meta.json                  -> info útil (timesteps, fecha, device, etc.)
    """
    _ensure_dir(folder)
    import torch

    torch.save(model.policy.state_dict(), os.path.join(folder, "policy_state.pt"))
    if hasattr(model.policy, "optimizer") and model.policy.optimizer is not None:
        torch.save(model.policy.optimizer.state_dict(), os.path.join(folder, "optimizer_state.pt"))

    meta_dict = meta or {}
    meta_dict.setdefault("num_timesteps", int(getattr(model, "num_timesteps", 0)))
    meta_dict.setdefault("device", str(getattr(model.policy, "device", "cpu")))
    meta_dict.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)


class FolderCheckpointCallback(BaseCallback):
    """
    Callback que guarda checkpoints en CARPETAS cada `save_freq` timesteps.
    Estructura:
      models_dir/
        run_name/
          step_100000/
            policy_state.pt
            optimizer_state.pt
            meta.json
          step_200000/
            ...
    """
    def __init__(self, models_dir: str, run_name: str, save_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.models_dir = models_dir
        self.run_name = run_name
        self.save_freq = int(save_freq)
        self.last_saved = 0
        _ensure_dir(self.models_dir)
        _ensure_dir(os.path.join(self.models_dir, self.run_name))

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)  # provisto por BaseCallback
        if t - self.last_saved >= self.save_freq:
            self.last_saved = t
            step_dir = os.path.join(self.models_dir, self.run_name, f"step_{t}")
            _save_model_to_folder(self.model, step_dir, meta={"checkpoint_step": t})
            if self.verbose:
                print(f"[CHKPT] Guardado checkpoint en {step_dir}")
        return True


# ---------------------------
# Factories de entorno 
# ---------------------------
def make_env_thunk(nodos_indice, aristas_indice, seed=22, steps_maximo=1200, mascara=True,
                   rank=0, monitor_dir="./logs/monitor"):
    def _fn():
        e = RecogidaBasurasEnv(
            nodos_indice=nodos_indice,
            aristas_indice=aristas_indice,
            steps_maximo=steps_maximo,
            mascara=mascara,
            seed=seed,
        )
        os.makedirs(monitor_dir, exist_ok=True)
        filename = os.path.join(monitor_dir, f"monitor_{rank}.csv")
        return Monitor(e, filename=filename)  # ← guarda episodios a CSV
    return _fn


def _ensure_spawn():
    # En Windows/Jupyter asegura método 'spawn' para multiproceso
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


# ---------------------------
# Entrenamiento principal
# ---------------------------
def train_a2c(nodos_indice,
              aristas_indice,
              total_timesteps=1_200_000,
              n_envs=4,
              n_steps=16,
              learning_rate=3e-4,
              ent_coef=0.05,
              gamma=0.985,
              seed=22,
              device="cuda",
              run_name="run_default",
              models_dir="./models/a2c",
              tb_dir="./logs/tb-a2c",
              save_freq=100_000):
    """
    - Multiprocessing: SubprocVecEnv (fallback a DummyVecEnv si hiciera falta)
    - Checkpoints en carpetas cada `save_freq` (por defecto, 100k)
    - Guardado final a carpeta `models_dir/run_name/final`
    - policy_kwargs fijos (para poder recargar pesos sin cambiar arquitectura)
    """
    _ensure_dir(models_dir)
    _ensure_dir(tb_dir)
    _ensure_dir(os.path.join(models_dir, run_name))
    _ensure_spawn()

    # VecEnv paralelo como en tu idea
    monitor_dir = os.path.join(tb_dir, run_name, "monitor")  # ← carpeta específica del run

    try:
        thunks = [
            make_env_thunk(
                nodos_indice, aristas_indice,
                seed=seed+i,
                rank=i,                      # ← rank único por subproceso
                monitor_dir=monitor_dir
            )
            for i in range(n_envs)
        ]
        vec_env = SubprocVecEnv(thunks)      # paralelismo real
        used_vec = "SubprocVecEnv"
    except Exception as e:
        print(f"[AVISO] SubprocVecEnv falló ({e}). Fallback a DummyVecEnv (sin paralelismo).")
        thunks = [make_env_thunk(
            nodos_indice, aristas_indice,
            seed=seed,
            rank=0,
            monitor_dir=monitor_dir
        )]
        vec_env = DummyVecEnv(thunks)
        used_vec = "DummyVecEnv"

    vec_env = VecMonitor(vec_env)
    print(f"[INFO] VecEnv usado: {used_vec} | n_envs={vec_env.num_envs}")

    # policy_kwargs fijos
    n_nodes = len(nodos_indice)
    policy_kwargs = dict(
        hidden_dim=128, in_node_features=5, in_edge_features=2,
        n_tipos=2, max_nodes=n_nodes, gnn_layers=3,
    )

    
    # Modelo A2C
    model = A2C(
        policy=A2CPolicyGNNMasked,
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=1.0,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        device=device,
        tensorboard_log=tb_dir,
    )


    # --- CSV LOGGER (progress.csv + tensorboard) ---
    from stable_baselines3.common.logger import configure

    csv_logdir = os.path.join(tb_dir, run_name, "csv")
    os.makedirs(csv_logdir, exist_ok=True)

    # escribe a stdout, csv y tensorboard simultáneamente
    logger = configure(csv_logdir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)


    # Checkpoints en CARPETAS
    folder_ckpt = FolderCheckpointCallback(
        models_dir=models_dir,
        run_name=run_name,
        save_freq=save_freq,
        verbose=1,
    )

    # Entrenamiento (desde cero; no arrastra pesos)
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=True,
        callback=[folder_ckpt],
        tb_log_name=run_name,
    )

    # Guardado final a carpeta
    final_dir = os.path.join(models_dir, run_name, "final")
    _save_model_to_folder(model, final_dir, meta={"final": True})
    print(f"[OK] Modelo final guardado en carpeta: {final_dir}")

    return model, final_dir

