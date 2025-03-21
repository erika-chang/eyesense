import os
import tensorflow as tf
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_ds, val_ds, epochs=5, class_weight=None, 
                use_early_stopping=False, patience=5, model_save_path=None):
    """
    Train the model and return training history
    
    Args:
        model: Compiled Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs
        class_weight: Optional dictionary of class weights
        use_early_stopping: Whether to use early stopping
        patience: Patience for early stopping
        model_save_path: Path to save the model
        
    Returns:
        Training history
    """
    # Create callbacks list
    callback_list = []
    
    # Add early stopping if requested
    if use_early_stopping:
        es = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callback_list.append(es)
    
    # Add model checkpoint if save path is provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        checkpoint = callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True
        )
        callback_list.append(checkpoint)
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callback_list if callback_list else None,
        class_weight=class_weight,
        verbose=1
    )
    
    return history

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: History object returned by model.fit()
        save_path: Path to save the plot image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()