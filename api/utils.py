import tensorflow as tf
from tensorflow.keras import layers

def prepare_image(image):
    """
    Prepare image datasets from directories
    
    Args:
        Image
        
    Returns:
        Tensor
    """
    # Resize image
    resized_image = 
    
    # Normalize
    test_image = resized_image / 255
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    test_image = test_image.prefetch(buffer_size=AUTOTUNE)
    
    return test_image