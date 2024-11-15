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


def create_gif(image_folder, output_filename, duration=500):
        # Obtener todos los archivos de imagen en la carpeta
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith('.png')],
            key=lambda x: int(x.split('_')[1])  # Ordena por número de generación en el nombre
        )

        # Cargar todas las imágenes
        images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

        # Crear y guardar el GIF
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # 0 significa que el GIF se repetirá indefinidamente
        )

        print(f"GIF guardado como {output_filename}")


def load_image_as_matrix(image_path, threshold=128):
    """
    Carga una imagen en escala de grises y la convierte a una matriz binaria usando un umbral.

    :param image_path: Ruta de la imagen.
    :param threshold: Umbral para clasificar los píxeles en 0 (negro) y 1 (blanco).
    :return: binary_matrix (matriz binaria 1D), img_shape (dimensiones originales de la imagen)
    """
    # Cargar la imagen en modo escala de grises
    img = Image.open(image_path).convert("L")
    
    # Convertir la imagen a una matriz numpy
    img_array = np.array(img)
    
    # Convertir la imagen a binaria según el umbral
    binary_matrix = (img_array >= threshold).astype(int)
    
    # Devolver la matriz binaria aplanada y las dimensiones de la imagen
    return binary_matrix.flatten(), img_array.shape


class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, mutation_rate, crossover_rate, elitism_rate, image_path=None, threshold=128, save_interval = 10):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.population = []  # Almacena la población actual de individuos
        
        self.selection = Selection()  # Instancia de la clase Selection
        self.crossover = Crossover()  # Instancia de la clase Crossover
        self.mutation = Mutation(self.mutation_rate)  # Instancia de la clase Mutation
        
        self.generation = 0  # Contador de generaciones actuales
        
        # Cargar la imagen si se proporciona
        if image_path:
            self.target_matrix, self.img_shape = load_image_as_matrix(image_path, threshold)
            self.chromosome_length = len(self.target_matrix)  # Longitud del cromosoma según la imagen
        
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
        """Convierte un cromosoma en una imagen usando las dimensiones originales de la imagen."""
        # Convertir el individuo en una matriz con las dimensiones de la imagen original
        matrix = np.array(individual).reshape(self.img_shape)
        
        # Convertir la matriz binaria a una imagen en escala de grises
        img = Image.fromarray(np.uint8(matrix * 255))  # Multiplica por 255 para blanco y negro
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
            
            best_individual = self.population[np.argmax(self.fitness_values)]
            best_fitness = np.max(self.fitness_values)

            # Mostrar el fitness del mejor individuo
            print(f"Generación {generation + 1} - Mejor Fitness: {best_fitness}")
            
            if np.array_equal(best_individual, self.target_matrix):

                print("Solución Encontrada:")
                
                print(f"Fitness máximo actual: {np.max(self.fitness_values)}")
                self.save_individual_image(best_individual, generation)
                
                break
            
            # Guardar la imagen del mejor individuo solo cada 10 generaciones
            if generation % self.save_interval == 0:  # Guarda la imagen solo cada 10 generaciones
                best_individual = self.population[0]  # Suponiendo que este es el mejor individuo
                self.save_individual_image(best_individual, generation)