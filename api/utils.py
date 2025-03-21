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


def load_model():
    # Download model on startup
    MODEL_BUCKET = "eyesense-model1"
    MODEL_PATH = "best_model.h5"
    LOCAL_MODEL_PATH = "/tmp/model.h5"
    
    # Download the model from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    
    # Load the model
    global model
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    return model