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


def combine_channels_and_save_images(r_solutions, g_solutions, b_solutions, img_shape, output_filename="evolucion_generaciones.gif"):
    """
    Combina las soluciones de los canales R, G, B y guarda un GIF de la evolución.

    Args:
        r_solutions (list): Soluciones del canal R por generación.
        g_solutions (list): Soluciones del canal G por generación.
        b_solutions (list): Soluciones del canal B por generación.
        img_shape (tuple): Forma original de la imagen (alto, ancho).
        output_filename (str): Nombre del archivo GIF a guardar.
    """
    images = []
    num_generations = len(r_solutions)

    for i in range(num_generations):
        # Combinar los canales R, G, B
        r_channel = r_solutions[i].reshape(img_shape)
        g_channel = g_solutions[i].reshape(img_shape)
        b_channel = b_solutions[i].reshape(img_shape)

        combined_image = np.stack([r_channel, g_channel, b_channel], axis=-1)

        # Convertir la imagen combinada en una imagen PIL
        images.append(Image.fromarray(np.uint8(combined_image)))

    # Crear el GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0
    )
    print(f"GIF guardado como {output_filename}")
