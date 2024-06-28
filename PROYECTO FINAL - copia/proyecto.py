import numpy as np
from PIL import Image, ImageOps
import os
from scipy.signal import convolve2d  

def rgb_a_escala_gris(image):
    image_array = np.array(image)

    if len(image_array.shape) == 3:
        # Separar los canales de color
        r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
        
        grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b
        grayscale_image = Image.fromarray(grayscale_image.astype('uint8'))
    elif len(image_array.shape) == 2:
        grayscale_image = image 
    return grayscale_image

def normalizar_imagen(image):
    image_array = np.array(image)
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_image = (image_array - min_val) * (255.0 / (max_val - min_val))
    
    normalized_image = np.clip(normalized_image, 0, 255)
    normalized_image = normalized_image.astype('uint8')
    normalized_image = Image.fromarray(normalized_image)
    
    return normalized_image

def sobel_filtro(image):

    image_array = np.array(image).astype('int32')  

    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])


    gradient_x = convolve2d(image_array, kernel_x, mode='same', boundary='symm')
    gradient_y = convolve2d(image_array, kernel_y, mode='same', boundary='symm')

  
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    gradient_magnitude = gradient_magnitude.astype('uint8')
    sobel_image = Image.fromarray(gradient_magnitude)

    return sobel_image

def binarize_image(image, threshold):
    
    if image.mode != 'L':
        image = image.convert('L')
    image_array = np.array(image)

    binary_image = np.where(image_array > threshold, 255, 0)

    binary_image = binary_image.astype('uint8')
    binary_image = Image.fromarray(binary_image)

    return binary_image

def adaptive_threshold_segmentation(image, block_size=21, constant=4):
    if image.mode != 'L':
        image = image.convert('L')

    image_array = np.array(image)
    binary_image = np.zeros_like(image_array)
    height, width = image_array.shape
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
           
            block = image_array[y:y+block_size, x:x+block_size]
        
            local_threshold = np.mean(block) - constant
            binary_block = np.where(block > local_threshold, 255, 0)
            binary_image[y:y+block_size, x:x+block_size] = binary_block

    binary_image = binary_image.astype('uint8')
    binary_image = Image.fromarray(binary_image)
    
    return binary_image

def main():
    try:
        # Directorio y nombre de archivo
        directory = 'C:\\Users\\HP\\Documents\\4to Semestre\\PROCESAMIENTO DIGITAL DE IMAGEN\\PROYECTO FINAL'
        filename = 'dedo1.jpg'
        image_path = os.path.join(directory, filename)
        
        # Verificar si el archivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"No se encontró el archivo: {image_path}")
        
        # Cargar la imagen usando PIL
        image = Image.open(image_path)
        
        # Convertir la imagen a escala de grises
        grayscale_image = rgb_a_escala_gris(image)
        
        # Normalizar la imagen para ajustar el contraste
        normalized_image = normalizar_imagen(grayscale_image)
        
        # Aplicar filtro gaussiano para reducir el ruido directamente
        image_array = np.array(normalized_image)
        sigma = 1  # Valor de sigma para el filtro gaussiano
        size = int(2 * np.ceil(3 * sigma) + 1)
        x = np.arange(-size // 2 + 1, size // 2 + 1)
        kernel = np.exp(-0.5 * (x / sigma)**2)
        kernel /= kernel.sum()
        
        blurred_image = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=image_array)
        blurred_image = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=blurred_image)
        
        blurred_image = blurred_image.astype('uint8')
        smoothed_image = Image.fromarray(blurred_image)
        
       
        sobel_image = sobel_filtro(smoothed_image)
        
     
        threshold = 120
        binarized_image = binarize_image(sobel_image, threshold)
        
       
        adaptive_segmented_image = adaptive_threshold_segmentation(smoothed_image)
        
      
        grayscale_image_path = os.path.join(directory, 'dedo1_grayscale.jpg')
        adaptive_segmented_image.save(grayscale_image_path)
        
      
        adaptive_segmented_image.show()
        
        print(f"La imagen en escala de grises con segmentación adaptativa se ha guardado como: {grayscale_image_path}")
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()

