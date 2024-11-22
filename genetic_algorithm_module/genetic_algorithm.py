# genetic_algorithm.py

import numpy as np
from PIL import Image
import os

from .selection import roulette_wheel_selection
from .crossover import Crossover
from .mutation import Mutation


def evaluate_fitness_hamming(population, target):
    """Calcula fitness como 1 - distancia Hamming normalizada."""
    return 1 - np.mean(population != target, axis=1)

class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, mutation_rate, crossover_rate, elitism_rate, target, save_interval=10, log_filename="fitness_log.txt"):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.save_interval = save_interval
        self.target = target
        self.chromosome_length = len(self.target)
        self.population = np.random.randint(0, 256, (population_size, self.chromosome_length))

        self.crossover = Crossover()
        self.mutation = Mutation(mutation_rate)
        self.fitness_values = np.zeros(self.population_size)  # Inicializamos los valores de fitness

        self.generation = 0
        self.best_solution = None
        self.best_fitness = 0
        self.no_improvement_generations = 0
        
        self.log_filename = log_filename

    def run(self):
        """Ejecuta el algoritmo genético."""
        with open(self.log_filename, "w") as f:
            f.write("Generación,Mejor Fitness\n")

        for generation in range(self.max_generations):
            print(f"Generación {generation + 1} de {self.max_generations}...")

            # Evaluar fitness de la población
            self.evaluate_fitness()

            new_population = []

            # Generar nueva población
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover.uniform_crossover(parent1, parent2)
                new_population.extend(
                    [self.mutation.swap_mutation(child1), self.mutation.swap_mutation(child2)]
                )

            # Aplicar elitismo
            self.population = self.apply_elitism(new_population)
            self.generation += 1

            # Obtener el mejor individuo de esta generación
            current_best_fitness = np.max(self.fitness_values)
            current_best_solution = self.population[np.argmax(self.fitness_values)]

            # Actualizar el mejor individuo final
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution
                self.no_improvement_generations = 0
            else:
                self.no_improvement_generations += 1

            # Guardar en el log cada 10 generaciones
            if generation % 100 == 0:
                with open(self.log_filename, "a") as f:
                    f.write(f"{generation + 1},{current_best_fitness:.4f}\n")

            print(f"Generación {generation + 1} - Mejor Fitness: {self.best_fitness:.4f}")

            # Comprobar condiciones de parada
            if self.best_fitness >= 0.8:
                print("Fitness objetivo alcanzado.")
                break
            # if self.no_improvement_generations >= 200:
            #     print("No hubo mejora en 50 generaciones. Deteniendo el algoritmo.")
            #     break


    def apply_elitism(self, new_population):
        """Aplica elitismo para mantener a los mejores individuos utilizando el método get_elites."""
        elites = self.get_elites(self.fitness_values)  # Usamos el método get_elites para obtener los mejores individuos

        # Llenamos el resto de la población con la nueva población (excepto los élites)
        non_elite_population = np.array(new_population)[:self.population_size - len(elites)]

        # Concatenamos los élites con la nueva población
        return np.concatenate((elites, non_elite_population))
    
    def get_elites(self, fitness_values):
        """Retorna la élite (individuos con mejor fitness)."""
        elite_count = max(1, int(self.elitism_rate * self.population_size))
        elite_indices = np.argsort(fitness_values)[-elite_count:]
        return self.population[elite_indices]

    def evaluate_fitness(self):
        """Evaluar el fitness de la población utilizando la distancia Hamming."""
        self.fitness_values = evaluate_fitness_hamming(self.population, self.target)

    def select_parents(self):
        """Selecciona dos padres usando la selección por ruleta."""
        parent1, parent2 = roulette_wheel_selection(self.population, self.fitness_values)
        return parent1, parent2