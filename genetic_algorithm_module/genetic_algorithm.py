import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

from .selection import Selection
from .crossover import Crossover
from .mutation import Mutation  


@jit(nopython=True)
def evaluate_fitness_numba(population):
        return np.sum(population, axis=1)

class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, mutation_rate, crossover_rate, elitism_rate, chromosome_length):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.chromosome_length = chromosome_length
        self.elitism_rate = elitism_rate
        self.population = []  # Almacena la población actual de individuos
        
        self.selection = Selection()  # Instancia de la clase Selection
        self.crossover = Crossover()  # Instancia de la clase Crossover
        self.mutation = Mutation(self.mutation_rate)  # Instancia de la clase Mutation
        
        self.generation = 0  # Contador de generaciones actuales
        self.statistics = []
        
        # Para almacenar los resultados de cada generación
        self.best_fitness_per_generation = []
        self.worst_fitness_per_generation = []
        self.average_fitness_per_generation = []
        self.variance_fitness_per_generation = []


    def initialize_population(self):
        # Inicializa la población con genes aleatorios
        self.population = np.random.randint(0, 2, (self.population_size, self.chromosome_length))
        # Verificación
        assert all(len(ind) == self.chromosome_length for ind in self.population), \
            "Error: Hay individuos con longitud de cromosoma inconsistente en la población inicial."


    def evaluate_fitness(self):
        # Calcula el fitness de cada individuo y lo guarda en una lista separada
        self.fitness_values = evaluate_fitness_numba(self.population)


    def calculate_statistics(self, generation):
        # Calcula las estadísticas de fitness y las añade a la lista `statistics`
        fitness_values = self.population[:, -1]  # Última columna es el fitness
        avg_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        var_fitness = np.var(fitness_values)

        # Añadir estadísticas al registro de la generación actual
        self.statistics.append({
            'Generation': generation,
            'Average Fitness': avg_fitness,
            'Max Fitness': max_fitness,
            'Min Fitness': min_fitness,
            'Fitness Variance': var_fitness
        })
        
        # Almacena los valores para graficar después
        self.best_fitness_per_generation.append(max_fitness)
        self.worst_fitness_per_generation.append(min_fitness)
        self.average_fitness_per_generation.append(avg_fitness)


    def export_statistics_to_excel(self, filename):
        df = pd.DataFrame(self.statistics)
        df.to_csv(filename, index=False)


    def save_fitness_over_generations(self, filename="fitness_over_generations.png"):
        """ Guarda un gráfico de fitness promedio, mejor y peor individuo contra el número de generaciones. """
        generations = np.arange(1, self.max_generations + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.average_fitness_per_generation, label="Fitness Promedio", color='blue')
        plt.plot(generations, self.best_fitness_per_generation, label="Mejor Fitness", color='green')
        plt.plot(generations, self.worst_fitness_per_generation, label="Peor Fitness", color='red')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Fitness Promedio, Mejor y Peor Individuo por Generación')
        plt.legend()
        plt.grid(True)

        # Guardar la imagen en un archivo
        plt.savefig(filename)
        plt.close()  # Cerrar para liberar memoria


    def save_boxplot_fitness(self, filename="fitness_boxplot.png"):
        """ Guarda un diagrama de cajas con los valores de fitness a lo largo de las generaciones """
        generations = np.arange(1, self.max_generations + 1)
        fitness_data = []

        for generation in range(self.max_generations):
            fitness_data.append(self.fitness_values)  # Puedes agregar todos los valores de fitness en cada generación

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=fitness_data, color='lightblue')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Distribución de Fitness por Generación (Boxplot)')

        # Guardar la imagen en un archivo
        plt.savefig(filename)
        plt.close()  # Cerrar para liberar memoria


    def select_parents(self):
        # Llama al método de selección ruleta para obtener dos padres
        parent1, parent2 = self.selection.select_parents(self.population, self.fitness_values)
        return parent1, parent2


    def apply_crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            child1, child2 = self.crossover.uniform_crossover(parent1, parent2)
            assert len(child1) == self.chromosome_length and len(child2) == self.chromosome_length, \
                "Error: Los hijos generados en crossover no tienen la longitud esperada."
            return child1, child2
        else:
            return parent1, parent2  # Retorna los padres originales si no ocurre cruce


    def apply_mutation(self, individual):
        if random.random() < self.mutation_rate:
            mutated_individual = self.mutation.swap_mutation(individual)
            assert len(mutated_individual) == self.chromosome_length, \
                "Error: El individuo mutado no tiene la longitud esperada."
            return mutated_individual
        return individual


    def apply_elitism(self, new_population):
        # Ordena la población actual junto con sus valores de fitness
        sorted_indices = np.argsort(self.fitness_values)[::-1]  # Índices de mayor a menor fitness
        num_elites = int(self.elitism_rate * self.population_size)

        # Selecciona los mejores individuos (elite) usando los índices ordenados
        elites = [self.population[i] for i in sorted_indices[:num_elites]]

        # Añade los élites a la nueva población
        new_population.extend(elites)

        # Recorta la nueva población si excede el tamaño deseado
        new_population = new_population[:self.population_size]

        # Convierte la nueva población a un array de NumPy
        return np.array(new_population)


    def run(self):

        if  not self.population:
            self.initialize_population()

        # Para llevar el conteo de generaciones sin mejora
        generations_without_improvement = 0
        previous_best_fitness = None  # Variable para almacenar el mejor fitness de la generación anterior

        for generation in range(self.max_generations):
            
            print(f"Generación {generation + 1} de {self.max_generations}...")
            self.evaluate_fitness()
            
            # Verificar si el fitness máximo es igual al de la generación anterior
            best_fitness = np.max(self.fitness_values)
            if previous_best_fitness is not None and best_fitness == previous_best_fitness:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0  # Reiniciar el contador si hubo una mejora

            previous_best_fitness = best_fitness  # Actualizar el mejor fitness de la generación actual

            # Si el fitness no mejora durante 15 generaciones, detener el algoritmo
            if generations_without_improvement >= 15:
                print("Condición de parada alcanzada: El mejor fitness no ha cambiado durante 15 generaciones.")
                break

            new_population = []

            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                child1, child2 = self.apply_crossover(parent1, parent2)
                new_population.extend(
                    [self.apply_mutation(child1), self.apply_mutation(child2)])

            # Verificación de consistencia en new_population
            assert all(len(ind) == self.chromosome_length for ind in new_population), \
                "Error: Los individuos en new_population no tienen la longitud esperada antes de aplicar elitismo."

            self.population = self.apply_elitism(new_population)
            self.generation += 1
            self.calculate_statistics(self.generation)
            
        # Exportar estadísticas a Excel
        self.export_statistics_to_excel("genetic_algorithm_statistics.xlsx")
        
        # Guardar las gráficas en archivos en vez de mostrarlas
        self.save_fitness_over_generations("fitness_over_generations.png")
        self.save_boxplot_fitness("fitness_boxplot.png")
