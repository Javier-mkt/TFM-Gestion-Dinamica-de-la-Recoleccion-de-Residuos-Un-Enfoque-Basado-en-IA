

import math
import os
import sys
import time

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Para multiproceso
import torch.multiprocessing as mp
from torch.distributions import Categorical

### Global variables

MODEL_PATH = 'recogida_basuras_a3c.pth'
ENV_NAME = "RecogidaBasurasEnv"  # solo a modo informativo
SEED = 22
NUM_PROCESSES = 1                 # ≥2 para A3C real (1 test + 1 train). Si pones 1, entreno en foreground.
EPISODES_TRAINING = 50
EPISODES_TESTING = 10
VALUE_LOSS_COEF = 0.5
ENTROPY_BETA = 0.01               # bonus de entropía para explorar mejor (opcional)
GAMMA = 0.99
LR = 2.5e-4

TRAINING_PARAMETERS = {
    'trajectory_steps': 10000,
    'num_processes': NUM_PROCESSES
}

### Define model architecture

from modelo_gnn_final import ActorCriticGNN, EncoderGNN, tensorizacion_grafo
from env_basuras_final import RecogidaBasurasEnv

# Factory del entorno (ajusta si necesitas parámetros como nodos_indice, aristas_indice)
def make_env():
    # return RecogidaBasurasEnv(nodos_indice, aristas_indice)
    return RecogidaBasurasEnv()

# Wrapper que acopla EncoderGNN + ActorCriticGNN
class ActorCritic(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, num_acciones_tipo=2):
        super().__init__()
        # Encoder con 5 features por nodo y 2 por arista (según tu tensorización)
        self.encoder = EncoderGNN(
            in_node_features=5,
            in_edge_features=2,
            hidden_dim=hidden_dim
        )
        # Actor-Crítico: sabe cuántos nodos tiene tu grafo
        self.ac = ActorCriticGNN(
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_acciones_tipo=num_acciones_tipo
        )

    def forward(self, x, edge_index, edge_attr, batch, mascara_acciones=None):
        h = self.encoder(x, edge_index, edge_attr)
        return self.ac(h, batch, mascara_acciones=mascara_acciones)



# Máscara a logits (pone -inf donde la acción no es válida)
def apply_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits [B, A], mask [B, A] con 1 válidas y 0 inválidas
    invalid = (mask == 0)
    logits = logits.masked_fill(invalid, float("-inf"))
    return logits

# Transfiere gradientes del modelo local al global (sin return prematuro)
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            pass
        shared_param._grad = param.grad

# Selección de acción (una decisión para 'tipo' y otra para 'destino')
def select_action(model, obs, info, device="cpu"):
    # Preparar entrada
    x, edge_index, edge_attr = tensorizacion_grafo(obs)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

    # Máscaras booleanas
    mask_tipo = torch.tensor(info["mascara"]["mascara_tipo"], dtype=torch.bool, device=device).unsqueeze(0)
    mask_dest = torch.tensor(info["mascara"]["mascara_destino"], dtype=torch.bool, device=device).unsqueeze(0)
    mascara = {"tipo": mask_tipo, "destino": mask_dest}
    
    #print(f"[DEBUG select_action] x.shape={x.shape}, edge_index.shape={edge_index.shape}, edge_attr.shape={edge_attr.shape}")
    #print(f"[DEBUG select_action] mask_tipo sum={mask_tipo.sum().item()}, mask_dest sum={mask_dest.sum().item()}")


    # Forward
    tipo_logits, destino_logits, value = model(x, edge_index, edge_attr, batch, mascara_acciones=mascara)

    #print(f"\n[DEBUG select_action]")
    #print(f"mask_tipo={mask_tipo.tolist()}  (orden esperado: [mover, recoger])")
    #print(f"tipo_logits={tipo_logits.detach().cpu().tolist()[0]}")
    #print(f"destino_logits primeros 10={destino_logits.detach().cpu().tolist()[0][:10]}")

    # Distribuciones
    tipo_dist = torch.distributions.Categorical(logits=tipo_logits)
    dest_dist = torch.distributions.Categorical(logits=destino_logits.squeeze(0))  # 👈 quita batch

    tipo_a = tipo_dist.sample()
    dest_a = dest_dist.sample()

    #print(f"[DEBUG acción elegida] tipo={tipo_a.item()} destino={dest_a.item()}")

    assert mask_tipo[0, tipo_a].item(), f"Se eligió tipo inválido {tipo_a.item()} con máscara {mask_tipo.tolist()}"
    assert mask_dest[0, dest_a].item(), f"Se eligió destino inválido {dest_a.item()} con máscara {mask_dest.tolist()}"

    logprob = tipo_dist.log_prob(tipo_a) + dest_dist.log_prob(dest_a)
    entropy = tipo_dist.entropy() + dest_dist.entropy()

    # Asegurar que son escalares
    action = {"tipo": int(tipo_a.item()), "destino": int(dest_a.item())}

    return action, logprob, value.squeeze(-1), entropy

### Funciones para el entrenamiento de un agente mediante A3C

def train(rank, episodes, training_params, shared_model, counter, lock,
          nodos_indice, aristas_indice, optimizer=None, models_path='models'):
    
    # Prueba guardar pesos
    if models_path is None:
        models_path = os.path.abspath("models")
    os.makedirs(models_path, exist_ok=True)
    print(f"[DEBUG train] rank={rank} usando models_path={models_path}", flush=True)


    print(f"[DEBUG train] rank={rank} iniciado con {episodes} episodios")
    torch.manual_seed(SEED + rank)
    device = "cpu"

    # Instancia entorno y modelo local
    env = RecogidaBasurasEnv(nodos_indice, aristas_indice)
    local_model = ActorCritic(num_nodes=len(nodos_indice), hidden_dim=64)
    local_model.load_state_dict(shared_model.state_dict())
    local_model.train()

    optimizer = optim.Adam(shared_model.parameters(), lr=LR)

    episodic_losses = []
    episodic_rewards = []

    print("Inicio Entrenamineto")

    for ep in range(int(episodes)):
        obs, info = env.reset(seed=SEED + rank)
        done = False

        values, log_probs, rewards, entropies = [], [], [], []
        steps = 0
        ep_reward = 0.0

        while (not done) and (steps < training_params['trajectory_steps']):
            action, logprob, value, entropy = select_action(local_model, obs, info, device=device)
            # DEBUG VALIDACIÓN
            #print(f"[DEBUG main] acción seleccionada = {action}")
            #print(f"[DEBUG main] máscara tipo = {info['mascara']['mascara_tipo']}, suma máscara destino = {np.sum(info['mascara']['mascara_destino'])}")

            #if action["tipo"] == 0:  # mover
            #    valido = info["mascara"]["mascara_destino"][action["destino"]]
            #    print(f"[DEBUG main] destino {action['destino']} válido? {valido}")
            #    assert valido == 1, f"Destino inválido {action['destino']} pese a máscara {info['mascara']['mascara_destino']}"
            #elif action["tipo"] == 1:  # recoger
            #    valido = info["mascara"]["mascara_tipo"][1]
            #    print(f"[DEBUG main] recoger válido? {valido}")
            #    assert valido == 1, f"Acción recoger inválida en nodo {info}"

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            with lock:
                counter.value += 1

            values.append(value)
            log_probs.append(logprob)
            entropies.append(entropy)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            ep_reward += float(reward)  #Necesario el float()?

            obs, info = next_obs, next_info
            steps += 1

        with torch.no_grad():
            if not done:
                _, _, next_value, _ = select_action(local_model, obs, info, device=device)
                R = next_value
            else:
                R = torch.tensor(0.0)

        print(f"Actualización pesos en episodio {ep+1}/{episodes}")

        policy_loss = torch.tensor(0.0)
        value_loss = torch.tensor(0.0)
        entropy_bonus = torch.tensor(0.0)

        R_t = R
        for t in reversed(range(len(rewards))):
            R_t = rewards[t] + GAMMA * R_t
            advantage = R_t - values[t]
            value_loss = value_loss + advantage.pow(2)
            policy_loss = policy_loss - log_probs[t] * advantage.detach()
            entropy_bonus = entropy_bonus + entropies[t]

        loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_BETA * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        ensure_shared_grads(local_model, shared_model)
        optimizer.step()

        local_model.load_state_dict(shared_model.state_dict())

        episodic_losses.append(loss.item())
        episodic_rewards.append(ep_reward)

        print(f"[DEBUG train] rank={rank}, episodio {ep+1}/{episodes} completado | loss={loss.item():.4f} | steps={steps} | recompensa = {ep_reward}")

        # === Guardar modelo cada X episodios (ej. cada 10) ===
        if models_path and rank == 0 and (ep + 1) % 20 == 0:
            models_path = os.path.abspath("models")
            os.makedirs(models_path, exist_ok=True)

            save_path = os.path.join(models_path, f"checkpoint_ep{ep+1}.pt")
            torch.save(shared_model.state_dict(), save_path)
            print(f"[DEBUG train] Guardado modelo en {save_path}", flush = True)  # Prueba flush

    logs_path = os.path.join(models_path, "logs")
    os.makedirs(logs_path, exist_ok=True)

    np.save(os.path.join(logs_path, f"losses_rank{rank}.npy"), episodic_losses)
    np.save(os.path.join(logs_path, f"rewards_rank{rank}.npy"), episodic_rewards)

    print(f"[DEBUG train] Guardadas curvas de entrenamiento en {logs_path}")
    if models_path and rank == 0:
        final_path = os.path.join(models_path, f"final_model_rank{rank}.pt")
        torch.save(shared_model.state_dict(), final_path)
        print(f"[DEBUG train] Modelo final guardado en {final_path}")


def test(rank, episodes, training_params, shared_model, counter,
         nodos_indice, aristas_indice, render=False, models_path='models'):
    print(f"[DEBUG test] rank={rank} iniciado con {episodes} episodios")
    torch.manual_seed(SEED + 100)
    device = "cpu"

    env = RecogidaBasurasEnv(nodos_indice, aristas_indice)
    model = ActorCritic(num_nodes=len(nodos_indice), hidden_dim=64)
    model.eval()

    for ep in range(int(episodes)):
        model.load_state_dict(shared_model.state_dict())
        obs, info = env.reset(seed=SEED + 100 + ep)
        done = False
        ep_reward, steps = 0.0, 0

        while (not done) and (steps < 5000):
            x, edge_index, edge_attr = tensorizacion_grafo(obs)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
            x, edge_index, edge_attr = x.to(device), edge_index.to(device), edge_attr.to(device)

            mask_tipo = torch.tensor(info["mascara"]["mascara_tipo"], dtype=torch.bool, device=device).unsqueeze(0)
            mask_dest = torch.tensor(info["mascara"]["mascara_destino"], dtype=torch.bool, device=device).unsqueeze(0)
            mascara = {"tipo": mask_tipo, "destino": mask_dest}

            with torch.no_grad():
                tipo_logits, destino_logits, _ = model(x, edge_index, edge_attr, batch, mascara_acciones=mascara)
                tipo_logits = apply_mask_to_logits(tipo_logits, mask_tipo)
                destino_logits = apply_mask_to_logits(destino_logits, mask_dest)
                tipo_a = torch.argmax(tipo_logits, dim=-1).item()
                dest_a = torch.argmax(destino_logits, dim=-1).item()
                action = {"tipo": int(tipo_a), "destino": int(dest_a)}

            obs, reward, terminated, truncated, info = env.step((action))
            done = terminated or truncated
            ep_reward += float(reward)
            steps += 1

        print(f"[DEBUG test] rank={rank}, episodio {ep+1}/{episodes} terminado | reward={ep_reward:.2f} | steps={steps} | steps_total={counter.value}")
