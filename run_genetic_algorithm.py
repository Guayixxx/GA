#run_genetic_algorithm.py

from genetic_algorithm_module.genetic_algorithm import GeneticAlgorithm, create_gif

# Parámetros para el algoritmo genético
population_size = 10
max_generations = 2000
mutation_rate = 1
crossover_rate = 0.4
elitism_rate = 0.00000001
image_path = "/home/juan-pablo/Documentos/Artificial/GA/Firma.png"

# Crear y ejecutar el algoritmo genético
gen_algo = GeneticAlgorithm(
    population_size=population_size,
    max_generations=max_generations,
    mutation_rate=mutation_rate,
    crossover_rate=crossover_rate,
    elitism_rate=elitism_rate,
    image_path=image_path,
)

gen_algo.run()
print("Ejecución completa.")

#Crear Gif
create_gif(image_folder='images', output_filename='evolucion_generaciones.gif')