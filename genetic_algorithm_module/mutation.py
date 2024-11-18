# mutation.py

import numpy as np
import random

class Mutation:
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def swap_mutation(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual