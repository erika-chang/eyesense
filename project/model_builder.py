from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.applications import ResNet50

def build_model(num_classes=7, input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Build and compile a ResNet50-based model for image classification
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for the Adam optimizer
        
    Returns:
        Compiled Keras model
    """
    # Create base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model
    
    # Add custom layers for classification
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)  # Uncomment to add dropout for regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model