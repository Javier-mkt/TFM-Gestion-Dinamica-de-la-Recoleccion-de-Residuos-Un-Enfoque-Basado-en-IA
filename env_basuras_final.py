import math
import os
import sys
import time
import random

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import networkx as nx 
import osmnx as ox

class RecogidaBasurasEnv(gym.Env):

    def __init__(self, nodos_indice, aristas_indice, capacidad_camion = 120.0, steps_maximo = 2500, mascara = True, seed = None): # añadida máscara para indicar si el agente solo elije las acciones permitidas o pueda elegir todas las acciones posibles (incluso las prohibidas)
        super().__init__()
        self.nodos_indice = nodos_indice
        self.aristas_indice = aristas_indice
        self.capacidad_camion = capacidad_camion
        self.carga_camion = 0
        self.steps_maximo = steps_maximo
        self.steps = 0
        self.tiempo_total = 0 #s
        self.mascara = mascara

        self.nodo_inicial = 103 #Entrada pueblo.
        self.nodo_actual = self.nodo_inicial
        self.nodo_anterior = None

        self.adjacencia = self._nodos_adjacentes()  

        self.seed_value = seed
        if seed is not None:
            self.set_seed(seed)

        # Espacio de acciones 
        self.action_space = spaces.Dict({
            "tipo" : spaces.Discrete(2), # 1 recoger basura, 0 moverse
            "destino" : spaces.Discrete(len(nodos_indice)) 
        })

        # Espacio de observaciones
        self.observation_space = spaces.Dict({
            "posicion_camion" : spaces.Discrete(len(nodos_indice)),
            "llenado_camion" : spaces.Box(0.0, self.capacidad_camion, shape=()),
            "contenedor" : spaces.Discrete(2),
            "llenado_contenedor" : spaces.Box(0.0, 1.0, shape=()) # Nivel lleando contenedores normalizado
        })



    # Creación dle diccionario de nodos accesibles a partir de uno 
    def _nodos_adjacentes(self):  
        adj = {nid: [] for nid in self.nodos_indice.keys()}
        for _, data in self.aristas_indice.items():
            u = data["desde"]
            v = data["hasta"]
            adj[u].append(v)
        return adj
    
    def set_seed(self, seed):
        self.seed_value = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self, seed = None, options = None):  
        if seed is not None:
            self.set_seed(seed)
        elif self.seed_value is not None:
            self.set_seed(self.seed_value)

        super().reset(seed = seed)
        self.nodo_actual = self.nodo_inicial
        self.carga_camion = 0.0
        self.steps = 0
        self.tiempo_total = 0

        # Reinicio de los nodos: rellenado de contenedores e incialización posicion inicial camión 
        for indice, nodo in self.nodos_indice.items():
            nodo["llenado"] = 0.5 if nodo["contenedor"] == 1 else 0 #Futuro, np.random(0.0, 0.90)
            nodo["llenado_camion"] = 0.0
            nodo["posicion_camion"] = 1 if indice == self.nodo_inicial else 0

        # Inicializacion nuevas condiciones inciales tráfico

        obs = self._obtener_observacion()
        info = {"mascara": self._mascara_acciones()} if self.mascara else {}
        return obs, info
    


    def _obtener_observacion(self):
        nodo = self.nodos_indice[self.nodo_actual]

        obs_simple = {
            "posicion_camion": self.nodo_actual,
            "llenado_camion": float(min(1.0, self.carga_camion / self.capacidad_camion)),
            "contenedor": int(nodo["contenedor"]),
            "llenado_contenedor": float(nodo["llenado"])
        }

        
        obs_grafo = {
            "nodos_indice" : self.nodos_indice,
            "aristas_indice" : self.aristas_indice 
        }

        return {"simple" : obs_simple, "grafo" : obs_grafo}



    def step(self, action):
        recompensa = 0
        info = {}
        self.steps += 1
        
        tipo = action["tipo"]
        destino = action["destino"]

        if tipo == 1:
            recompensa += self._recogida_basura()
        elif tipo == 0:
            if destino in self.adjacencia[self.nodo_actual]:
                # Cambio de nodo del camion
                self.nodo_anterior = self.nodo_actual
                self.nodos_indice[self.nodo_actual]["posicion_camion"] = 0
                self.nodo_actual = destino
                self.nodos_indice[self.nodo_actual]["posicion_camion"] = 1

                recompensa += self._recorrido_camion()
            else:
                recompensa += -1
        else: 
            recompensa += -1

        terminado = False
        truncado = False

        # Condiciones finalización
        # Terminado
        if self.carga_camion >= self.capacidad_camion:
            terminado = True
        if self.nodo_actual == self.nodo_inicial and self.steps > 1:
            terminado = True
        
        # Truncado
        if self.steps >= self.steps_maximo:
            truncado = True

        # Recompensa final
        if terminado or truncado:
            recompensa += self._recompensa_final()

        obs = self._obtener_observacion()
        info = {"mascara": self._mascara_acciones()} if self.mascara else {}
        return obs, recompensa, terminado, truncado, info
    


    def _recogida_basura(self):
        recompensa = 0
        nodo = self.nodos_indice[self.nodo_actual]
        if nodo["contenedor"] == 1 and nodo["llenado"] > 0:
            basura_disponible = nodo["llenado"] * nodo["capacidad_contenedor"]
            self.carga_camion += basura_disponible

            carga_camion_norm = min(1.0, self.carga_camion / self.capacidad_camion)

            nodo["llenado"] = 0 

            for n in self.nodos_indice.values():
                n["llenado_camion"] = carga_camion_norm
            
            # Tiempo recogida 
            self.tiempo_total += 30 #sec, tiempo aprox recogida (cambiarlo a variable)

            # Recompensas 
            recompensa = (basura_disponible / nodo["capacidad_contenedor"]) * 2  # 1 factor arbitrario (recompensa inicial y sencilla) (si es menor al 50/70%, añadir mini penalización)
            return recompensa
        
        elif nodo["contenedor"] == 1 and nodo["llenado"] == 0:
            recompensa = 0
            return recompensa

        else:
            recompensa = -1
            return recompensa


    def _recorrido_camion(self):
        recompensa = 0

        # Recompensas por tiempo recorrido y distancia recorrida

        return recompensa



    def _recompensa_final(self):
        recompensa = 0

        # Añadir recompensas y penalizaciones

        return recompensa
    



    def render(self):
        print(f"Nodo actual: {self.nodo_actual} | Carga camión: {self.carga_camion:.2f} kg | Step: {self.steps}")




    #def _get_accessible_nodes(self):
    #        return self.adjacencia[self.nodo_actual]
    


    def _mascara_acciones(self):
        mascara_tipo = np.array([True, True], dtype=bool)  
        mascara_destino = np.zeros(len(self.nodos_indice), dtype=bool)

        adjacentes = self._nodos_adjacentes()[self.nodo_actual]

    # Excluir volver al nodo anterior, salvo callejón sin salida
        if self.nodo_anterior is not None:
            vecinos_validos = [v for v in adjacentes if v != self.nodo_anterior]
            if len(vecinos_validos) == 0:
                # caso callejón sin salida → permitimos volver
                vecinos_validos = [self.nodo_anterior]
        else:
            vecinos_validos = adjacentes

        mascara_destino[vecinos_validos] = True

        nodo = self.nodos_indice[self.nodo_actual]
        if not (nodo["contenedor"] == 1 and nodo["llenado"] > 0):
            mascara_tipo[0] = False

        return {
            "mascara_tipo": mascara_tipo,
            "mascara_destino": mascara_destino,
        }
    
    
# Función testeo del entorno

def agente_aleatorio(env, max_steps=20):
    obs, info = env.reset()
    terminated, truncated = False, False
    contenedores_visitados = 0
    recompensa_acumulada = 0

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        print(f"Observación: {obs['simple']}")
        print(f"Info: {info}")

        nodo_tiene_contenedor = obs["simple"]["contenedor"] == 1
        lleno_contenedor = obs["simple"]["llenado_contenedor"] > 0

        if info["mascara"]["mascara_tipo"][0]:
            # 100% probabilidad de recoger, 0% de moverse
            contenedores_visitados += 1
            if random.random() <= 1.0:
                action = {"tipo": 1, "destino": 0}
            else:
                posibles = np.where(info["mascara"]["mascara_destino"])[0]
                destino = int(random.choice(posibles)) if len(posibles) > 0 else 0
                action = {"tipo": 0, "destino": destino}
        else:
            # Nodo sin contenedor o contenedor vacío → siempre moverse
            posibles = np.where(info["mascara"]["mascara_destino"])[0]
            destino = int(random.choice(posibles)) if len(posibles) > 0 else 0
            action = {"tipo": 0, "destino": destino}

        print(f"Acción elegida: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(terminated)
        print(truncated)
        print(f"Recompensa: {reward}")
        recompensa_acumulada += reward
        env.render()

        if terminated or truncated:
            print("Episodio terminado.")
            print(f"Contenedores_visitados = {contenedores_visitados}")
            print(f"Recompensa acumulada = {recompensa_acumulada}")
            break