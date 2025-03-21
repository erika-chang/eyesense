import tensorflow as tf
from tensorflow.keras import layers

def normalize_dataset(ds):
    """Normalize images in a dataset by rescaling pixel values to [0,1]"""
    normalization_layer = layers.Rescaling(1./255)
    return ds.map(lambda x, y: (normalization_layer(x), y))

def prepare_data(train_ds, val_ds):
    """
    Load and prepare image datasets from directories
    
    Args:
        train_dir: Directory containing training images organized in class folders
        val_dir: Directory containing validation images organized in class folders
        
    Returns:
        Tuple of (train_ds, val_ds)
    """
    # Normalize datasets
    normalized_train_ds = normalize_dataset(train_ds)
    normalized_val_ds = normalize_dataset(val_ds)
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    normalized_train_ds = normalized_train_ds.prefetch(buffer_size=AUTOTUNE)
    normalized_val_ds = normalized_val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return normalized_train_ds, normalized_val_ds