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