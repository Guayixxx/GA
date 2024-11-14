import numpy as np
import random
from numba import jit
from PIL import Image
import os

from .selection import Selection
from .crossover import Crossover
from .mutation import Mutation  

@jit(nopython=True)
def evaluate_fitness_hamming(population, target_matrix):
         # Calcula la distancia de Hamming de cada individuo con respecto a la matriz objetivo
        fitness = np.zeros(population.shape[0])
        for i in range(population.shape[0]):
            # Calcula la distancia de Hamming manualmente
            fitness[i] = 1 - np.sum(population[i] != target_matrix) / len(target_matrix)
        return fitness

class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, mutation_rate, crossover_rate, elitism_rate, chromosome_length, save_interval = 1):
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
        
        self.target_matrix = np.array([[0, 0, 0],
                                       [1, 0, 1],
                                       [1, 0, 1],
                                       [0, 0, 1]]).flatten()
        
        self.save_interval = save_interval


    def initialize_population(self):
        # Inicializa la población con genes aleatorios
        self.population = np.random.randint(0, 2, (self.population_size, self.chromosome_length))
        # Verificación
        assert all(len(ind) == self.chromosome_length for ind in self.population), \
            "Error: Hay individuos con longitud de cromosoma inconsistente en la población inicial."


    def evaluate_fitness(self):
        # Calcula el fitness de cada individuo y lo guarda en una lista separada
        self.fitness_values = evaluate_fitness_hamming(self.population, self.target_matrix)


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


    def individual_to_image(self, individual):
        """Convierte un cromosoma (individuo) a una imagen 2D de tamaño 4x3 (en este caso)."""
        # Convertir el individuo (un array 1D) en una matriz 2D
        matrix = np.array(individual).reshape(4, 3)  # Asumiendo que la matriz es 4x3
        # Convertir la matriz binaria a una imagen en escala de grises
        img = Image.fromarray(np.uint8(matrix * 255))  # Multiplicamos por 255 para que sea en blanco y negro
        
        # Redimensionar manteniendo la relación de aspecto
        width, height = img.size
        new_width = 400  # Nuevo ancho
        new_height = int((new_width / width) * height)  # Calculamos la altura proporcional
        img = img.resize((new_width, new_height), Image.NEAREST)  # Redimensionamos sin distorsionar
        
        return img


    def save_individual_image(self, individual, generation):
        """Convierte un cromosoma en imagen y guarda la imagen en un archivo en la carpeta 'images'."""
        # Crear la carpeta si no existe
        if not os.path.exists('images'):
            os.makedirs('images')

        # Convertir el individuo en una imagen
        img = self.individual_to_image(individual)

        # Crear un nombre de archivo único basado en la generación
        filename = f'images/generation_{generation}_best_individual.png'

        # Guardar la imagen
        img.save(filename)
        print(f"Imagen guardada como {filename}")


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
            
            if np.any(self.fitness_values >= 1.0):  # Cuando la distancia de Hamming sea 0
                print("Solución alcanzada con la matriz objetivo.")
                
                # Imprimir la matriz objetivo y la solución encontrada
                print("Matriz Objetivo:")
                print(self.target_matrix.reshape(4, 3))  # Reshape para mostrarla como matriz 4x3

                best_individual = self.population[0]  # Suponiendo que este es el mejor individuo
                print("Solución Encontrada:")
                print(best_individual.reshape(4, 3))  # Reshape para mostrarla como matriz 4x3
                
                break
            
            # Guardar la imagen del mejor individuo solo cada 10 generaciones
            if generation % self.save_interval == 0:  # Guarda la imagen solo cada 10 generaciones
                best_individual = self.population[0]  # Suponiendo que este es el mejor individuo
                self.save_individual_image(best_individual, generation)
