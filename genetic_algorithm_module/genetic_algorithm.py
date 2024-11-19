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


def load_image_as_rgb_matrices(image_path):
    """
    Carga una imagen como tres matrices independientes para los canales R, G y B.
    
    Args:
        image_path (str): Ruta de la imagen a cargar.

    Returns:
        tuple: Tres matrices NumPy (R, G, B) y las dimensiones de la imagen.
    """
    # Abrir la imagen
    img = Image.open(image_path).convert("RGB")
    
    # Convertir la imagen a una matriz NumPy
    img_array = np.array(img)  # Dimensiones: (alto, ancho, 3)
    
    # Separar los canales R, G y B
    r_channel = img_array[:, :, 0]  # Canal Rojo
    g_channel = img_array[:, :, 1]  # Canal Verde
    b_channel = img_array[:, :, 2]  # Canal Azul

    # Calcular el tamaño del cromosoma (número de píxeles)
    num_pixels = img_array.shape[0] * img_array.shape[1]

    # Retornar las matrices y el tamaño del cromosoma
    return r_channel.flatten(), g_channel.flatten(), b_channel.flatten(), num_pixels, img_array.shape[:2]

def create_gif(image_folder, output_filename, duration=500):
    """Crea un GIF a partir de imágenes en una carpeta con formato `gen_[número].png`."""
    # Obtener todos los archivos que terminan en .png
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.startswith('gen_') and f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0])  # Extraer el número después de 'gen_'
    )

    # Verificar si hay imágenes válidas
    if not image_files:
        raise ValueError("No se encontraron archivos de imagen válidos en la carpeta.")

    # Cargar todas las imágenes
    images = [Image.open(os.path.join(image_folder, f)) for f in image_files]

    # Crear y guardar el GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF guardado como {output_filename}")

class GeneticAlgorithm:
    def __init__(self, population_size, max_generations, mutation_rate, crossover_rate, elitism_rate, image_path, threshold=128, save_interval=10):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.save_interval = save_interval

        self.r_target, self.g_target, self.b_target, self.chromosome_length, self.img_shape = load_image_as_rgb_matrices(image_path)
        self.population = np.random.randint(0, 256, (population_size, self.chromosome_length))

        self.crossover = Crossover()
        self.mutation = Mutation(mutation_rate)
        self.fitness_values = np.zeros(self.population_size)  # Inicializamos los valores de fitness
        
        self.generation = 0

    def run(self):
        """Ejecuta el algoritmo genético."""
        # Inicializar la población si no está inicializada
        if self.population.size == 0:  # Cambié la condición aquí
            self.initialize_population()

        # Crear/limpiar el archivo de log al inicio
        with open("fitness_log.txt", "w") as f:
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

            # Verificación de consistencia en new_population
            assert all(len(ind) == self.chromosome_length for ind in new_population), \
                "Error: Los individuos en new_population no tienen la longitud esperada antes de aplicar elitismo."

            # Aplicar elitismo
            self.population = self.apply_elitism(new_population)
            self.generation += 1

            # Obtener el mejor individuo y su fitness
            best_individual = self.population[np.argmax(self.fitness_values)]
            best_fitness = np.max(self.fitness_values)

            # Guardar el fitness en el archivo
            with open("fitness_log.txt", "a") as f:
                f.write(f"{generation + 1},{best_fitness:.4f}\n")

            # Mostrar el fitness del mejor individuo
            print(f"Generación {generation + 1} - Mejor Fitness: {best_fitness}")

            # Comprobar si se alcanzó la solución óptima
            if np.array_equal(best_individual, self.target_matrix):
                print("Solución Encontrada:")
                print(f"Fitness máximo actual: {best_fitness}")
                self.save_image(best_individual, generation)
                break

            # Guardar la imagen del mejor individuo solo cada intervalo
            if generation % self.save_interval == 0:
                self.save_image(best_individual, generation)

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

    def save_image(self, individual, generation):
        """Convierte un individuo en imagen y lo guarda."""
        img_array = individual.reshape(self.img_shape) * 255
        Image.fromarray(np.uint8(img_array)).save(f'images/gen_{generation}.png')

    def evaluate_fitness(self):
        """Evaluar el fitness de la población utilizando la distancia Hamming."""
        self.fitness_values = evaluate_fitness_hamming(self.population, self.target_matrix)

    def select_parents(self):
        """Selecciona dos padres usando la selección por ruleta."""
        parent1, parent2 = roulette_wheel_selection(self.population, self.fitness_values)
        return parent1, parent2