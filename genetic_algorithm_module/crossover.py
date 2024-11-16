import numpy as np

class Crossover:
    @staticmethod
    def uniform_crossover(parent1, parent2):
        mask = np.random.rand(len(parent1)) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2
