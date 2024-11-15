# mutation.py
import numpy as np
import random

class Mutation:
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def swap_mutation(self, individual):
        genes = individual[:]  # Copia de los genes
        i, j = random.sample(range(len(genes)), 2)
        genes[i], genes[j] = genes[j], genes[i]
        return genes  # Solo devolver los genes
    
    def scramble_mutation(self, individual):
        genes = individual[:]  # Copia de los genes
        # Seleccionar al azar un rango de índices
        start, end = sorted(random.sample(range(len(genes)), 2))
        # Barajar los genes dentro del rango seleccionado
        subset = genes[start:end+1]
        random.shuffle(subset)
        genes[start:end+1] = subset
        return genes  # Devolver los genes mutados
    
    def flip_bit_mutation(self, individual):
        genes = individual[:]  # Copia de los genes
        for i in range(len(genes)):
            # Mutar cada bit con probabilidad igual a la tasa de mutación
            if random.random() < self.mutation_rate:
                genes[i] = 1 - genes[i]  # Cambiar 0 a 1 o 1 a 0
        return genes  # Devolver los genes mutados