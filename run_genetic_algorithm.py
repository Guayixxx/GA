#run_genetic_algorithm.py

from genetic_algorithm_module.genetic_algorithm import GeneticAlgorithm, create_gif

# Parámetros para el algoritmo genético
population_size = 100
max_generations = 500
mutation_rate = 0.1
crossover_rate = 0.8
elitism_rate = 0.2
chromosome_length = 12

# Crear y ejecutar el algoritmo genético
gen_algo = GeneticAlgorithm(
    population_size=population_size,
    max_generations=max_generations,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    elitism_rate=elitism_rate,
    chromosome_length=chromosome_length
)

gen_algo.run()
print("Ejecución completa.")

#Crear Gif
create_gif(image_folder='images', output_filename='evolucion_generaciones.gif')