from genetic_algorithm_module.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm_module.ImagesMethods import load_image_as_rgb_matrices, combine_channels_and_save_images

# Parámetros del algoritmo genético
params = {
    "population_size": 20,
    "max_generations": 50000,
    "mutation_rate": 0.7,
    "crossover_rate": 0.4,
    "elitism_rate": 0.2,
}

# Cargar y separar los canales de la imagen
r_target, g_target, b_target, img_shape = load_image_as_rgb_matrices("/home/juan-pablo/Documentos/Artificial/GA/FotoRed.jpg")

# Crear y ejecutar el algoritmo genético para cada canal
gen_algo_r = GeneticAlgorithm(**params, target=r_target, log_filename="fitness_log_r.txt")
gen_algo_r.run()
r_final_solution = gen_algo_r.best_solution  # Última mejor solución del canal R

gen_algo_g = GeneticAlgorithm(**params, target=g_target, log_filename="fitness_log_g.txt")
gen_algo_g.run()
g_final_solution = gen_algo_g.best_solution  # Última mejor solución del canal G

gen_algo_b = GeneticAlgorithm(**params, target=b_target, log_filename="fitness_log_b.txt")
gen_algo_b.run()
b_final_solution = gen_algo_b.best_solution  # Última mejor solución del canal B

# Combinar los canales y guardar la imagen final
combine_channels_and_save_images(
    [r_final_solution], [g_final_solution], [b_final_solution], img_shape, output_filename='resultado_final.png'
)
