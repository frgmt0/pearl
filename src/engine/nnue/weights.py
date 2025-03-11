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
    
    # Set model-specific filename if not specified
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if the model is enhanced and XL for automatic naming
        is_enhanced = getattr(model, 'use_enhanced', False)
        is_xl = getattr(model, 'use_xl', False)
        
        if is_enhanced and is_xl:
            name = f"pearlXL_{timestamp}"
        elif is_enhanced:
            name = f"pearl_{timestamp}"
        else:
            name = f"nnue_weights_{timestamp}"
    
    # For primary models, use consistent names
    if name == "base":
        name = "pearl"  # Default model is now pearl.pt
    
    # Add .pt extension if not present
    if not name.endswith(".pt"):
        name += ".pt"
        
    # Create full path
    path = os.path.join("saved_models", name)
    
    # Save model configuration along with weights
    model_config = {
        'state_dict': model.state_dict(),
        'metadata': {
            'use_enhanced': getattr(model, 'use_enhanced', False),
            'use_xl': getattr(model, 'use_xl', False),
            'timestamp': datetime.now().isoformat(),
            'version': '2.0'
        }
    }
    
    # Save model with config
    torch.save(model_config, path)
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
    
    # Load checkpoint
    checkpoint = torch.load(path)
    
    # Check if we have a new format checkpoint (with metadata)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # New format - extract metadata
        metadata = checkpoint.get('metadata', {})
        saved_enhanced = metadata.get('use_enhanced', False)
        saved_xl = metadata.get('use_xl', False)
        
        # Check model compatibility
        model_enhanced = getattr(model, 'use_enhanced', False)
        model_xl = getattr(model, 'use_xl', False)
        
        if saved_enhanced != model_enhanced or saved_xl != model_xl:
            # Architecture mismatch - this will raise an error
            print(f"Warning: Model architecture mismatch. Saved model uses enhanced={saved_enhanced}, xl={saved_xl}.")
            print(f"Current model uses enhanced={model_enhanced}, xl={model_xl}.")
            raise ValueError("Model architecture mismatch")
        
        # Load state dict
        state_dict = checkpoint['state_dict']
    else:
        # Old format - direct state dict
        state_dict = checkpoint
        
        # Check model compatibility with direct inspection
        model_enhanced = getattr(model, 'use_enhanced', False)
        model_xl = getattr(model, 'use_xl', False)
        
        # Check keys to detect model type
        has_enhanced_keys = any(key.startswith('enhanced_model.') for key in state_dict.keys())
        
        # Check for XL-specific keys (if model is XL)
        has_xl_keys = False
        if model_xl:
            has_xl_keys = any('hidden3' in key for key in state_dict.keys())
            
            # If model is XL but state_dict is not, we have a mismatch
            if not has_xl_keys:
                print(f"Warning: Model architecture mismatch. Loaded model does not have XL architecture.")
                print(f"Current model uses enhanced={model_enhanced}, xl={model_xl}.")
                raise ValueError("Model architecture mismatch")
        
        # If model is enhanced but state_dict is standard (or vice versa), we have a mismatch
        if model_enhanced != has_enhanced_keys:
            print(f"Warning: Model architecture mismatch. Loaded model is {'standard' if not has_enhanced_keys else 'enhanced'} format.")
            print(f"Current model uses enhanced={model_enhanced}, xl={model_xl}.")
            raise ValueError("Model architecture mismatch")
    
    # Try to load weights into model
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise ValueError("Failed to load model weights")
    print(f"Model weights loaded from {path}")
    
    return model

def detect_model_architecture(path):
    """
    Detect the architecture type from a saved model file.
    
    Args:
        path: Path to the model file
        
    Returns:
        Tuple of (use_enhanced, use_xl) or None if detection fails
    """
    try:
        if not os.path.exists(path):
            return None
            
        # Load checkpoint
        checkpoint = torch.load(path)
        
        # Check if we have a new format checkpoint (with metadata)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # New format - extract metadata
            metadata = checkpoint.get('metadata', {})
            saved_enhanced = metadata.get('use_enhanced', False)
            saved_xl = metadata.get('use_xl', False)
            return (saved_enhanced, saved_xl)
        else:
            # Old format - check state dict keys
            state_dict = checkpoint
            has_enhanced_keys = any(key.startswith('enhanced_model.') for key in state_dict.keys())
            
            if has_enhanced_keys:
                # Check for XL-specific keys
                has_xl_keys = any('hidden3' in key for key in state_dict.keys())
                return (True, has_xl_keys)
            else:
                return (False, False)
    except Exception:
        return None

def get_latest_weights():
    """
    Find the most recently saved weights file.
    Prioritizes base.pt if it exists.
    
    Returns:
        Path to the latest weights file, or None if no weights found
    """
    if not os.path.exists("saved_models"):
        return None
    
    # First check for model files with standardized names
    model_files = [
        os.path.join("saved_models", "standard.pt"),
        os.path.join("saved_models", "pearl.pt"),
        os.path.join("saved_models", "pearlXL.pt"),
        os.path.join("saved_models", "base.pt")  # Legacy name
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            return model_path
    
    # Otherwise look for timestamped weights
    weight_files = [f for f in os.listdir("saved_models") 
                   if (f.startswith("nnue_weights_") or f.startswith("pearl_") or f.startswith("pearlXL_")) 
                   and f.endswith(".pt")]
    
    if not weight_files:
        return None
    
    # Sort by timestamp (newest first)
    weight_files.sort(reverse=True)
    
    return os.path.join("saved_models", weight_files[0])
    
def get_best_matching_weights(use_enhanced=True, use_xl=False):
    """
    Find the best matching weights file for a given architecture.
    
    Args:
        use_enhanced: Whether to find an enhanced model
        use_xl: Whether to find an XL model
        
    Returns:
        Path to the best matching weights file, or None if no match found
    """
    if not os.path.exists("saved_models"):
        return None
    
    # Look for the specific named file first
    if use_enhanced and use_xl:
        pearlxl_path = os.path.join("saved_models", "pearlXL.pt")
        if os.path.exists(pearlxl_path):
            return pearlxl_path
    
    if use_enhanced and not use_xl:
        pearl_path = os.path.join("saved_models", "pearl.pt")
        if os.path.exists(pearl_path):
            return pearl_path
    
    if not use_enhanced:
        standard_path = os.path.join("saved_models", "standard.pt")
        if os.path.exists(standard_path):
            return standard_path
    
    # No exact match found, try detecting model types
    # Get all weight files
    files = []
    for filename in os.listdir("saved_models"):
        if filename.endswith(".pt"):
            path = os.path.join("saved_models", filename)
            files.append(path)
    
    if not files:
        return None
    
    # First try to find a model that exactly matches the desired architecture
    for path in files:
        arch = detect_model_architecture(path)
        if arch == (use_enhanced, use_xl):
            return path
    
    # Fall back to any model file
    return get_latest_weights()
