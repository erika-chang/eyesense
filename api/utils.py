import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def prepare_image(image, height, width):
    """
    Prepare image datasets from directories

    Args:
        Image

    Returns:
        Numpy Array
    """

    # Convert the image to RGB
    image = image.convert("RGB")

    # Resize the image
    image = image.resize((width, height))

    # Convert to NumPy array and normalize
    image_array = np.array(image) / 255.0

    # Expand dimensions (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)


    return image_array
