"""
Training Script for ISL Detection Model with RESUME SUPPORT
Saves checkpoint after every epoch - can resume from where you stopped!
"""
import os
import sys
import json
import time
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

from src.config import EPOCHS, BATCH_SIZE, MODEL_DIR, MODEL_PATH
from src.preprocess import prepare_training_data
from src.model import create_model, get_callbacks


# Checkpoint paths
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.h5")
PROGRESS_PATH = os.path.join(MODEL_DIR, "training_progress.json")


class EpochCheckpoint(Callback):
    """
    Custom callback to save model and progress after EVERY epoch
    Allows resuming training from any point
    """
    
    def __init__(self, checkpoint_path, progress_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.progress_path = progress_path
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def on_epoch_end(self, epoch, logs=None):
        # Save model checkpoint
        self.model.save(self.checkpoint_path)
        
        # Update history
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        
        # Save progress
        progress = {
            'completed_epochs': epoch + 1,
            'total_epochs': EPOCHS,
            'best_val_accuracy': max(self.history['val_accuracy']),
            'last_val_accuracy': logs.get('val_accuracy'),
            'history': self.history
        }
        
        with open(self.progress_path, 'w') as f:
            json.dump(progress, f, indent=2)
        
        print(f"\n  💾 Checkpoint saved at epoch {epoch + 1}/{EPOCHS}")
        print(f"     Resume anytime with: python src/train.py")


def load_checkpoint():
    """
    Load existing checkpoint and progress if available
    Returns: (model, start_epoch, history) or (None, 0, None)
    """
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(PROGRESS_PATH):
        try:
            # Load progress
            with open(PROGRESS_PATH, 'r') as f:
                progress = json.load(f)
            
            completed = progress['completed_epochs']
            total = progress['total_epochs']
            
            if completed >= total:
                print(f"\n✓ Training already completed ({completed}/{total} epochs)")
                print("  Delete checkpoint files to retrain from scratch:")
                print(f"    - {CHECKPOINT_PATH}")
                print(f"    - {PROGRESS_PATH}")
                return None, completed, progress.get('history')
            
            # Load model
            print(f"\n📂 Found checkpoint at epoch {completed}/{total}")
            model = load_model(CHECKPOINT_PATH)
            print(f"  ✓ Model loaded from checkpoint")
            
            return model, completed, progress.get('history')
            
        except Exception as e:
            print(f"  ⚠ Could not load checkpoint: {e}")
            print("  Starting fresh training...")
    
    return None, 0, None


def clear_checkpoints():
    """Remove checkpoint files to start fresh"""
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
    print("  ✓ Cleared previous checkpoints")


def plot_training_history(history_dict, save_path=None):
    """Plot training history from dictionary"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history_dict['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history_dict['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history_dict['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Training plot saved to: {save_path}")
    
    plt.show()


def train(resume=True):
    """
    Main training function with resume support
    
    Args:
        resume: If True, resume from checkpoint. If False, start fresh.
    """
    print("=" * 60)
    print("   ISL Detection - Training with Resume Support")
    print("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check for existing checkpoint
    model = None
    start_epoch = 0
    prev_history = None
    
    if resume:
        model, start_epoch, prev_history = load_checkpoint()
        
        if start_epoch >= EPOCHS:
            # Training already complete
            if prev_history:
                plot_path = os.path.join(MODEL_DIR, "training_history.png")
                plot_training_history(prev_history, save_path=plot_path)
            return None
    else:
        clear_checkpoints()
    
    # Step 1: Load data
    print("\n[Step 1/3] Loading dataset...")
    try:
        X_train, X_val, y_train, y_val, data_gen = prepare_training_data()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return None
    
    print(f"  ✓ Training: {len(X_train)} samples")
    print(f"  ✓ Validation: {len(X_val)} samples")
    
    # Step 2: Create or load model
    print("\n[Step 2/3] Preparing model...")
    if model is None:
        model = create_model()
        print(f"  ✓ New model created ({model.count_params():,} params)")
    else:
        print(f"  ✓ Resuming from epoch {start_epoch}")
    
    # Step 3: Train
    remaining_epochs = EPOCHS - start_epoch
    print(f"\n[Step 3/3] Training for {remaining_epochs} more epochs...")
    print(f"  • Starting from epoch: {start_epoch + 1}")
    print(f"  • Target epochs: {EPOCHS}")
    print(f"  • Batch size: {BATCH_SIZE}")
    print("-" * 60)
    
    # Create checkpoint callback
    checkpoint_callback = EpochCheckpoint(CHECKPOINT_PATH, PROGRESS_PATH)
    
    # Restore previous history if resuming
    if prev_history:
        checkpoint_callback.history = prev_history
    
    start_time = time.time()
    
    # Train
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback] + get_callbacks(),
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Save final model
    model.save(MODEL_PATH)
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print("\n" + "-" * 60)
    print(f"  ✓ Training completed in {training_time/60:.1f} minutes")
    print(f"  ✓ Final validation accuracy: {val_accuracy*100:.2f}%")
    print(f"  ✓ Model saved to: {MODEL_PATH}")
    
    # Plot history
    plot_path = os.path.join(MODEL_DIR, "training_history.png")
    plot_training_history(checkpoint_callback.history, save_path=plot_path)
    
    # Clean up checkpoints after successful completion
    print("\n  🧹 Cleaning up checkpoints...")
    clear_checkpoints()
    
    print("\n" + "=" * 60)
    print("   ✅ Training Complete!")
    print("   Run: python src/predict.py")
    print("=" * 60)
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ISL Detection Model')
    parser.add_argument('--fresh', action='store_true', 
                       help='Start fresh training (ignore checkpoints)')
    args = parser.parse_args()
    
    train(resume=not args.fresh)
