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
        self.fitness_values_per_generation = []


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
        
        # Guardar los valores de fitness para cada generación
        self.fitness_values_per_generation.append(self.fitness_values)
        
        # Almacena los valores para graficar después
        self.best_fitness_per_generation.append(max_fitness)
        self.worst_fitness_per_generation.append(min_fitness)
        self.average_fitness_per_generation.append(avg_fitness)


    def export_statistics_to_excel(self, filename):
        df = pd.DataFrame(self.statistics)
        df.to_csv(filename, index=False)


    def save_fitness_over_generations(self, filename="fitness_over_generations.png"):
        """ Guarda una gráfica del fitness promedio, mejor y peor a lo largo de las generaciones. """
        plt.figure(figsize=(10, 6))
        generations = np.arange(1, len(self.best_fitness_per_generation) + 1)  # Ajusta la longitud de generaciones

        plt.plot(generations, self.best_fitness_per_generation, label="Mejor Fitness", color="green")
        plt.plot(generations, self.worst_fitness_per_generation, label="Peor Fitness", color="red")
        plt.plot(generations, self.average_fitness_per_generation, label="Fitness Promedio", color="blue")

        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Evolución del Fitness a lo largo de las generaciones")
        plt.legend()
        plt.grid(True)

        plt.savefig(filename)
        plt.close()


    def save_boxplot_fitness(self, filename="fitness_boxplot.png"):
        """ Guarda un diagrama de cajas del fitness para cada generación individual. """
        # Usar la cantidad de generaciones completadas
        completed_generations = len(self.fitness_values_per_generation)
    
        # Revisar que haya datos para cada generación
        assert completed_generations > 0, "No hay datos para generar el diagrama de cajas."
    
        # Preparar los datos para el boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(self.fitness_values_per_generation[:completed_generations], vert=True, patch_artist=True)
    
        # Etiquetas y configuración del gráfico
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Diagrama de Cajas del Fitness por Generación")
        plt.xticks(range(1, completed_generations + 1))  # Asegurarse de que haya una etiqueta por generación
        plt.grid(True)
    
        # Guardar la imagen en un archivo
        plt.savefig(filename)
        plt.close()


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

        for generation in range(self.max_generations):
            
            print(f"Generación {generation + 1} de {self.max_generations}...")
            self.evaluate_fitness()

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
            
            # Condición de parada si el promedio de fitness se estabiliza en las últimas 15 generaciones
            if len(self.average_fitness_per_generation) >= 15:
                recent_averages = self.average_fitness_per_generation[-15:]
                if np.all(np.isclose(recent_averages, recent_averages[0], atol=1e-5)):
                    print("Condición de parada alcanzada: El promedio de fitness se ha estabilizado en las últimas 15 generaciones.")
                    break
            
        # Exportar estadísticas a Excel
        self.export_statistics_to_excel("genetic_algorithm_statistics.xlsx")
        
        # Guardar las gráficas en archivos en vez de mostrarlas
        self.save_fitness_over_generations("fitness_over_generations.png")
        self.save_boxplot_fitness("fitness_boxplot.png")
