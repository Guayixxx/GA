# ImagesMethods

from PIL import Image
import numpy as np
import os

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
    return r_channel.flatten(), g_channel.flatten(), b_channel.flatten(), img_array.shape[:2]


def combine_channels_and_save_images(r_solutions, g_solutions, b_solutions, img_shape, output_folder="images"):
    """
    Combina las soluciones de los canales R, G, B en cada generación y guarda las imágenes resultantes.

    Args:
        r_solutions (list): Soluciones del canal R por generación.
        g_solutions (list): Soluciones del canal G por generación.
        b_solutions (list): Soluciones del canal B por generación.
        img_shape (tuple): Forma original de la imagen (alto, ancho).
        output_folder (str): Carpeta donde se guardarán las imágenes combinadas.
    """
    # Crear la carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    num_generations = len(r_solutions)
    for i in range(num_generations):
        # Combinar los canales R, G, B
        r_channel = r_solutions[i].reshape(img_shape)
        g_channel = g_solutions[i].reshape(img_shape)
        b_channel = b_solutions[i].reshape(img_shape)

        combined_image = np.stack([r_channel, g_channel, b_channel], axis=-1)

        # Guardar la imagen combinada
        output_path = os.path.join(output_folder, f"gen_{i + 1}.png")
        Image.fromarray(np.uint8(combined_image)).save(output_path)

    print(f"Imágenes de evolución guardadas en la carpeta: {output_folder}")