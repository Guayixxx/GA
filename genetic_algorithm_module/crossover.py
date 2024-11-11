# crossover.py
import random   
import numpy as np


class Crossover:
    def __init__(self):
        pass

    def uniform_crossover(self, parent1, parent2):
        # Crear hijos con genes seleccionados aleatoriamente de ambos padres
        child1 = np.array([gene1 if random.random() < 0.5 else gene2 for gene1, gene2 in zip(parent1, parent2)])
        child2 = np.array([gene2 if random.random() < 0.5 else gene1 for gene1, gene2 in zip(parent1, parent2)])
        
        return child1, child2