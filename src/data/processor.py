import numpy as np
from PIL import Image

def preprocess_image(image_path: str, target_size: tuple = (64, 64)) -> np.ndarray:
    try:
        with Image.open(image_path) as img:
            gray_img = img.convert('L')
            resized_img = gray_img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(resized_img, dtype = np.float32)
            normalized_array = img_array / 255.0
            flattened_vector = normalized_array.flatten()

            return flattened_vector
    
    except FileNotFoundError:
        raise FileNotFoundError(f"No image found at target path: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")
    
if __name__ == "__main__":
    print("Image processor module initialized successfully.")
