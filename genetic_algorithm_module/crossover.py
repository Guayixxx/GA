# crossover.py
import numpy as np


class Crossover:
    def __init__(self):
        pass

    def uniform_crossover(self, parent1, parent2):
        # Crear hijos con genes seleccionados aleatoriamente de ambos padres
        child1 = np.array([gene1 if np.random.random() < 0.5 else gene2 for gene1, gene2 in zip(parent1, parent2)])
        child2 = np.array([gene2 if np.random.random() < 0.5 else gene1 for gene1, gene2 in zip(parent1, parent2)])
        
        return child1, child2
    
    
    def single_point_crossover(self, parent1, parent2):
        # Elegir un punto de cruce al azar
        crossover_point = np.random.randint(1, len(parent1) - 1)
        
        # Crear los hijos intercambiando segmentos de los padres
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        
        return child1, child2