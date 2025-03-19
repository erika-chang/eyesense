import argparse
import os

from preprocessing import prepare_data
from model_builder import build_model
from train import train_model, plot_training_history
from data.data_loader import load_dataset

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an image classification model using ResNet50")
        
    # Model parameters
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--img_height", type=int, default=224, help="Image height")
    parser.add_argument("--img_width", type=int, default=224, help="Image width")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--use_early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    train_ds, val_ds, class_names = load_dataset(structured_dir="./data/data_train/")
    normalized_train_ds, normalized_val_ds = prepare_data(train_ds, val_ds)
    
    # Build model
    print("Building model...")
    model = build_model(
        num_classes=args.num_classes,
        input_shape=(args.img_height, args.img_width, 3),
        learning_rate=args.learning_rate
    )
    
    # Model summary
    model.summary()
    
    # Train model
    print("Training model...")
    model_save_path = os.path.join(args.output_dir, "best_model.h5")
    history = train_model(
        model,
        normalized_train_ds,
        normalized_val_ds,
        epochs=args.epochs,
        use_early_stopping=args.use_early_stopping,
        patience=args.patience,
        model_save_path=model_save_path
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.output_dir, "training_history.png"))
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()