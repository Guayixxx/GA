from genetic_algorithm_module.genetic_algorithm import GeneticAlgorithm, create_gif

# Parámetros del algoritmo genético
params = {
    "population_size": 20,
    "max_generations": 5000,
    "mutation_rate": 0.7,
    "crossover_rate": 0.4,
    "elitism_rate": 0.2,
    "image_path": "FirmaRed.png"   # Cambiar según el SO
    # "image_path": "/home/juan-pablo/Documentos/Artificial/GA/FirmaRed.png"   # Cambiar según el SO
    
}

# Ejecutar el algoritmo genético
gen_algo = GeneticAlgorithm(**params)
gen_algo.run()

# Crear el GIF
create_gif(image_folder='images', output_filename='evolucion_generaciones.gif')