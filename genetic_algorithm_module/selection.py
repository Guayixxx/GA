# En genetic_algorithm_module/selection.py
import numpy as np
from numba import jit

# Función independiente que implementa la ruleta
@jit(nopython=True)
def roulette_wheel(population, fitness_values):
    # Calcular el fitness total usando fitness_values
    total_fitness = np.sum(fitness_values)
    probabilities = np.cumsum(fitness_values / total_fitness)
    
    def select_one():
        r = np.random.random()
        return population[np.searchsorted(probabilities, r)]
    
    # Seleccionar dos padres
    parent1 = select_one()
    parent2 = select_one()
    
    return parent1, parent2

# Clase Selection sin @jit
class Selection:
    def __init__(self):
        pass
    
    def select_parents(self, population, fitness_values):
        return roulette_wheel(population, fitness_values)
