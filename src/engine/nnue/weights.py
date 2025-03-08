import os
import torch
from datetime import datetime

def save_weights(model, name=None):
    """
    Save model weights to a file.
    
    Args:
        model: NNUE model to save
        name: Optional name for the weights file
        
    Returns:
        Path to the saved weights file
    """
    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Generate filename
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"nnue_weights_{timestamp}"
    
    # Add .pt extension if not present
    if not name.endswith(".pt"):
        name += ".pt"
        
    # Create full path
    path = os.path.join("saved_models", name)
    
    # Save model state
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")
    
    return path

def load_weights(model, path):
    """
    Load model weights from a file.
    
    Args:
        model: NNUE model to load weights into
        path: Path to weights file
        
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}")
    
    # Load state dictionary
    state_dict = torch.load(path)
    
    # Load weights into model
    model.load_state_dict(state_dict)
    print(f"Model weights loaded from {path}")
    
    return model

def get_latest_weights():
    """
    Find the most recently saved weights file.
    
    Returns:
        Path to the latest weights file, or None if no weights found
    """
    if not os.path.exists("saved_models"):
        return None
    
    weight_files = [f for f in os.listdir("saved_models") if f.startswith("nnue_weights_") and f.endswith(".pt")]
    
    if not weight_files:
        return None
    
    # Sort by timestamp (newest first)
    weight_files.sort(reverse=True)
    
    return os.path.join("saved_models", weight_files[0])
