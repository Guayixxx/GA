import numpy as np

def roulette_wheel_selection(population, fitness_values):
    """Selección mediante ruleta basada en fitness."""
    probabilities = np.cumsum(fitness_values / np.sum(fitness_values))
    select = lambda: population[np.searchsorted(probabilities, np.random.random())]
    parent1 = select()
    parent2 = select()
    return parent1, parent2  # Retorna los dos padres seleccionados
