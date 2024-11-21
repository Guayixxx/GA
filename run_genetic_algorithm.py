from genetic_algorithm_module.genetic_algorithm import GeneticAlgorithm, create_gif
from genetic_algorithm_module.ImagesMethods import load_image_as_rgb_matrices, combine_channels_and_save_images

# Parámetros del algoritmo genético
params = {
    "population_size": 20,
    "max_generations": 5000,
    "mutation_rate": 0.7,
    "crossover_rate": 0.4,
    "elitism_rate": 0.2,
}

# Cargar y separar los canales de la imagen
r_target, g_target, b_target, img_shape = load_image_as_rgb_matrices("FotoRed.jpg")

# Crear y ejecutar el algoritmo genético para cada canal
gen_algo_r = GeneticAlgorithm(**params, target=r_target)
gen_algo_r.run()
r_solutions = gen_algo_r.best_solutions

gen_algo_g = GeneticAlgorithm(**params, target=g_target)
gen_algo_g.run()
g_solutions = gen_algo_g.best_solutions

gen_algo_b = GeneticAlgorithm(**params, target=b_target)
gen_algo_b.run()
b_solutions = gen_algo_b.best_solutions

# Combinar los canales y guardar las imágenes
output_folder = "images"
combine_channels_and_save_images(r_solutions, g_solutions, b_solutions, img_shape, output_folder=output_folder)

# Crear el GIF
create_gif(image_folder=output_folder, output_filename='evolucion_generaciones.gif')
