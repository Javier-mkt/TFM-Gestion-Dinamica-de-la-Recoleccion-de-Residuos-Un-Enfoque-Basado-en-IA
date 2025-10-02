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
from collections import defaultdict

class RecogidaBasurasEnv(gym.Env):

    def __init__(self, nodos_indice, aristas_indice, capacidad_camion = 120.0, steps_maximo = 250, mascara = True, seed = None): # añadida máscara para indicar si el agente solo elije las acciones permitidas o pueda elegir todas las acciones posibles (incluso las prohibidas)
        super().__init__()
        self.nodos_indice = nodos_indice
        self.aristas_indice = aristas_indice
        self.capacidad_camion = capacidad_camion
        self.carga_camion = 0
        self.steps_maximo = steps_maximo
        self.steps = 0
        self.tiempo_total = 0 #s
        self.mascara = mascara

        self.num_nodos = len(self.nodos_indice)
        self.num_aristas = len(self.aristas_indice)

        #self.nodo_inicial = 103 # Entrada pueblo Benimàmet (nodos total Benimàmet)
        #self.nodo_inicial = 79    # Entrada pueblo Benimàmet (nodos norte Benimàmet)
        self.nodo_inicial = 0     # Ebtrada pueblo Benimàmet (nodos norte reducido Benimàmet)
        # self.nodo_inicial = ...       # Entrada pueblo Benimàmet (nodos sud Benimàmet)
        self.nodo_actual = self.nodo_inicial
        self.nodo_anterior = None
        self.nodo_actual_recogido = False

        self.adjacencia = self._nodos_adjacentes()  

        # Recompensa extrínseca visita nuevos nodos
        self.nodos_visitados_ep = defaultdict(int)

        self.seed_value = seed
        if seed is not None:
            self.set_seed(seed)


        # Espacio de acciones 
        self.action_space = spaces.MultiDiscrete([2, self.num_nodos])  #tipo (0 -> mover, 1 -> recoger), destino

        # Espacio de observaciones
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodos, 4), dtype=np.float32),
            "edge_index": spaces.Box(low=-1, high=self.num_nodos, shape=(2, self.num_aristas), dtype=np.int64),
            "edge_attr": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_aristas, 2), dtype=np.float32),
            "mascara_tipo": spaces.MultiBinary(2),
            "mascara_destino": spaces.MultiBinary(self.num_nodos),
            "mask2_table": spaces.MultiBinary((2, self.num_nodos)),
        })



    # Creación dle diccionario de nodos accesibles a partir de uno 
    def _nodos_adjacentes(self):  
        adj = {nid: [] for nid in self.nodos_indice.keys()}
        for _, data in self.aristas_indice.items():
            u = data["desde"]
            v = data["hasta"]
            adj[u].append(v)
        return adj
    


    # Semilla
    def set_seed(self, seed):
        self.seed_value = seed
        np.random.seed(seed)
        #torch.manual_seed(seed)



    # Tensorización grafos
    def _tensorizacion_grafo(self):
        x = np.zeros((self.num_nodos, 4), dtype=np.float32)
        for nid, nodo in self.nodos_indice.items():
                x[nid, 0] = float(nodo["contenedor"])
                # x[nid, 1] = float(nodo["capacidad_contenedor"])   # Se quita por redundante (capacidad contenedores cte.)
                x[nid, 1] = float(nodo["llenado"])
                x[nid, 2] = float(nodo["posicion_camion"])
                x[nid, 3] = float(nodo["llenado_camion"])

        edge_index = np.zeros((2, self.num_aristas), dtype = np.int64)
        edge_attr = np.zeros((self.num_aristas, 2), dtype = np.float32)

        k = 0
        for _, arista in self.aristas_indice.items():
            edge_index[0, k] = arista["desde"]
            edge_index[1, k] = arista["hasta"]
            edge_attr[k, 0] = float(arista["distancia"])
            edge_attr[k, 1] = float(arista["tiempo_recorrido"])    # Cuando se añada dinamismo, como max triplicar. 
            k += 1

        return x, edge_index, edge_attr
    


    # Máscara de acciones
    def _mascara_acciones(self):
        
        # Condiciones de la máscara
        HABILITAR_RECOGER_SIEMPRE = True
        HABILITAR_QUEDARSE_NODO = False

        adjacentes = self._nodos_adjacentes()[self.nodo_actual]

        # Excluir volver al nodo anterior, salvo callejón sin salida
        if self.nodo_anterior is not None:
            vecinos_validos = [v for v in adjacentes if v != self.nodo_anterior]
            if len(vecinos_validos) == 0:
                # caso callejón sin salida → permitimos volver
                vecinos_validos = [self.nodo_anterior]
        else:
            vecinos_validos = adjacentes

        # Mascara destino [# Nº nodos] (1 adjacente, 0 no adjacente)
        mascara_destino = np.zeros((self.num_nodos,), dtype = np.int8)
        mascara_destino[vecinos_validos] = 1

        if HABILITAR_QUEDARSE_NODO:
            mascara_destino[self.nodo_actual] = 1

        # Mascara tipo [mover, recoger] (1)(0 no recoger, 1 sí recoger)
        if HABILITAR_RECOGER_SIEMPRE:
            mascara_tipo = np.array([1, 1], dtype = np.int8)

        else:
            nodo = self.nodos_indice[self.nodo_actual]
            recoger = (nodo["contenedor"] == 1 and not self.nodo_actual_recogido)
            mascara_tipo = np.array([1, 1 if recoger else 0], dtype = np.int8)
        
        # Mask2_table
        mask2_table = np.zeros((2, self.num_nodos), dtype=np.int8)
        mask2_table[0, :] = mascara_destino
        one_hot = np.zeros((self.num_nodos,), dtype=np.int8)
        one_hot[self.nodo_actual] = 1
        mask2_table[1, :] = one_hot

        return mascara_tipo, mascara_destino, mask2_table


    # Observación
    def _obtener_observacion(self):
        x, edge_index, edge_attr = self._tensorizacion_grafo()
        if self.mascara:
            mascara_tipo, mascara_destino, mask2_table = self._mascara_acciones()
        else:
            mascara_tipo = np.ones((2,), dtype=np.int8)
            mascara_destino = np.ones((self.num_nodos,), dtype=np.int8)
            mask2_table = np.ones((2, self.num_nodos), dtype=np.int8)

        obs = {
        "x":            np.asarray(x, dtype=np.float32).reshape(self.num_nodos, 4),
        "edge_index":   np.asarray(edge_index, dtype=np.int64).reshape(2, self.num_aristas),
        "edge_attr":    np.asarray(edge_attr, dtype=np.float32).reshape(self.num_aristas, 2),
        "mascara_tipo":    np.asarray(mascara_tipo, dtype=np.int8).reshape(2,),
        "mascara_destino": np.asarray(mascara_destino, dtype=np.int8).reshape(self.num_nodos,),
        "mask2_table":  np.asarray(mask2_table, dtype=np.int8).reshape(2, self.num_nodos),
    }
        return obs



    # Reinicio entorno
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
        self.nodos_visitados_ep.clear()
        self.nodos_visitados_ep[self.nodo_inicial] = 1
        self.nodo_anterior = None
        self.nodo_actual_recogido = False

        # Reinicio de los nodos: rellenado de contenedores e incialización posicion inicial camión 
        for indice, nodo in self.nodos_indice.items():
            nodo["llenado"] = 0.5 if nodo["contenedor"] == 1 else 0 #Futuro, np.random(0.0, 0.90)
            nodo["llenado_camion"] = 0.0
            nodo["posicion_camion"] = 1 if indice == self.nodo_inicial else 0

        # Inicializacion nuevas condiciones inciales tráfico

        obs = self._obtener_observacion()
        info = {}
        return obs, info



    # Paso entorno
    def step(self, action):
        recompensa = 0
        self.steps += 1
        
        tipo = int(action[0])
        destino = int(action[1])

        if tipo == 1:
            self.nodo_actual_recogido = True
            recompensa += self._recogida_basura()
            
            
        elif tipo == 0:
            if destino in self.adjacencia[self.nodo_actual]:

                # Cambio de nodo del camion
                self.nodo_anterior = self.nodo_actual
                self.nodos_indice[self.nodo_actual]["posicion_camion"] = 0
                self.nodo_actual = destino
                self.nodos_indice[self.nodo_actual]["posicion_camion"] = 1

                self.nodo_actual_recogido = False

                recompensa += self._recorrido_camion()

            elif destino == self.nodo_actual:
                
                #Penalización por no moverse, es decir, por quedarse
                recompensa -= 0.3

            else:
                print("Acción inválida, error máscara destino (destino no válido)")
                recompensa += -0.2
        else: 
            print("Acción inválida, error máscara tipo (fuera rango)")
            recompensa += -0.2

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
            recompensa += self._recompensa_final(terminado, truncado)

        obs = self._obtener_observacion()
        info = {}
        return obs, recompensa, terminado, truncado, info
    


    def _recogida_basura(self):
        recompensa = 0
        nodo = self.nodos_indice[self.nodo_actual]
        if nodo["contenedor"] == 1 and nodo["llenado"] > 0.05:
            basura_disponible = nodo["llenado"] * nodo["capacidad_contenedor"]
            self.carga_camion += basura_disponible

            carga_camion_norm = min(1.0, self.carga_camion / self.capacidad_camion)

            nodo["llenado"] = 0 

            for n in self.nodos_indice.values():
                n["llenado_camion"] = carga_camion_norm
            
            # Tiempo recogida 
            self.tiempo_total += 30 #sec, tiempo aprox recogida (cambiarlo a variable)

            # Recompensas 
            recompensa += 0.8 + 0.2 * (basura_disponible / nodo["capacidad_contenedor"])  # 1 factor arbitrario (recompensa inicial y sencilla) (si es menor al 50/70%, añadir mini penalización)
            return recompensa
        
        elif nodo["contenedor"] == 1:
            recompensa += -0.15  #Penalización por recoger en nodo sin basura suficiente
            return recompensa

        else:
            recompensa += -0.3
            return recompensa    #Penalizacion por recoger en nodo no contenedor



    def _recorrido_camion(self):
        recompensa = 0

        factor = 0.11
        alpha = 0.25 
        beta = 0.75   
        norm_pen = 0.12/17
        for _, arista in self.aristas_indice.items():
            if arista["desde"] == self.nodo_anterior and arista["hasta"] == self.nodo_actual:
                distancia = arista.get("distancia", 1000.0)  
                tiempo = arista.get("tiempo_recorrido", 100.0)
                break
        
        recompensa -= (alpha*(distancia*factor) + beta*tiempo)*norm_pen

        # Recompensa por visitar el nodo por primera vez y penalización por => 4

        recompensa_nuevo_nodo = 0
        penalizacion_repeticion_nodo = 0

        # Recompensa por primera visita
        self.nodos_visitados_ep[self.nodo_actual] += 1
        if self.nodos_visitados_ep[self.nodo_actual] <= 1:
            recompensa += recompensa_nuevo_nodo

        # Penalización por exceso de visitas
        if self.nodos_visitados_ep[self.nodo_actual] >= 4:
            exceso = self.nodos_visitados_ep[self.nodo_actual] - 3
            penalizacion = penalizacion_repeticion_nodo * exceso
            recompensa -= min(penalizacion, 0.15)

        return recompensa



    def _recompensa_final(self, terminado, truncado):
        recompensa = 0

        # Recompensa si vuelve a nodo inicial
        if self.nodo_actual == self.nodo_inicial:
            recompensa += 0.1                        

        # Penalización si no acaba en nodo inicial
        if truncado:
            recompensa -= 0.1
        
        # Recompensa por ratio de basura recogida 
        ratio_recogida = self.carga_camion / self.capacidad_camion
        recompensa += ratio_recogida*0                

        # Penalizacion por dejar contenedores sin recoger:
        count_contenedores_total = 0
        count_contenedores_llenos = 0

        for indice, nodo in self.nodos_indice.items():
            if nodo["contenedor"] == 1:
                count_contenedores_total += 1
            if nodo["contenedor"] == 1 and nodo["llenado"] > 0.05:
                count_contenedores_llenos += 1
        
        recompensa += ((count_contenedores_total - count_contenedores_llenos)/count_contenedores_total) * 0.5
    
        return recompensa
    


    def render(self):
        print(f"Nodo actual: {self.nodo_actual} | Carga camión: {self.carga_camion:.2f} kg | Step: {self.steps}")

    
    
    


# Función testeo del entorno (antigua versión)

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