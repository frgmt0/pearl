import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import chess
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import json
import requests
import tempfile
import shutil
from neural_network import ChessNet, board_to_input, move_to_index

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ChessParquetDataset(Dataset):
    """Chess dataset for training the neural network on Stockfish games from parquet files."""
    
    def __init__(self, file_path, max_samples=None):
        self.samples = []
        self.process_file(file_path, max_samples)
        
    def process_file(self, file_path, max_samples):
        """Process chess games from a parquet file."""
        print(f"Processing chess games from {file_path}...")
        
        # Read the parquet file
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Limit to max_samples if specified
        if max_samples and max_samples < len(df):
            df = df.sample(max_samples)
        
        count = 0
        for _, item in tqdm(df.iterrows(), total=len(df)):
            # Skip invalid entries
            if 'Moves' not in item or 'Termination' not in item or 'Result' not in item:
                continue
                
            try:
                # The 'Moves' column is a NumPy array of strings with UCI format moves (e.g., "d2d4")
                moves = item['Moves']
                
                # Handle different possible formats
                if isinstance(moves, np.ndarray):
                    # It's already a NumPy array
                    pass
                elif isinstance(moves, list):
                    # It's a list, convert to array
                    moves = np.array(moves)
                elif isinstance(moves, str):
                    # It's a string, try to parse it
                    if moves.startswith('[') and moves.endswith(']'):
                        # It's a string representation of a list
                        moves_list = moves.strip('[]').replace('"', '').replace("'", "").split(',')
                        moves = np.array([move.strip() for move in moves_list])
                    else:
                        # It's a space-separated string
                        moves = np.array(moves.split())
                
                # Create a board to replay the game
                board = chess.Board()
                positions = []
                
                # Record positions and outcomes during the game
                for uci_move in moves:
                    try:
                        # Parse UCI move (e.g., "d2d4")
                        move = chess.Move.from_uci(uci_move.strip())
                        
                        # Check if the move is legal
                        if move in board.legal_moves:
                            # Record position before the move is made
                            positions.append((board.copy(), move))
                            # Make the move
                            board.push(move)
                        else:
                            print(f"Illegal move {uci_move} in position {board.fen()}")
                            break
                    except ValueError as e:
                        print(f"Invalid UCI move: {uci_move} - {e}")
                        continue
                        
                # Determine the game result
                if item['Result'] == '1-0':
                    outcome = 1.0  # White wins
                elif item['Result'] == '0-1':
                    outcome = -1.0  # Black wins
                else:  # '1/2-1/2'
                    outcome = 0.0  # Draw
                
                # Add positions to samples with outcome as value target
                for pos, move in positions:
                    # Convert position to neural network input
                    input_tensor = torch.FloatTensor(board_to_input(pos))
                    
                    # Create policy target (one-hot encoded move)
                    move_idx = move_to_index(move)
                    
                    # Value target depends on whose turn it was and the outcome
                    value = outcome if pos.turn == chess.WHITE else -outcome
                    
                    self.samples.append((input_tensor, move_idx, value))
                
                count += 1
                    
            except Exception as e:
                # Skip problematic games
                print(f"Error processing game: {e}")
                continue
                
        print(f"Processed {len(self.samples)} positions from {count} games in {file_path}")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        input_tensor, move_idx, value = self.samples[idx]
        return input_tensor, move_idx, value

def download_parquet_file(file_index, temp_dir, args=None):
    """Download or locate a parquet file.
    
    If args.dataset_path is provided, this will use the local file.
    Otherwise, it will download from Hugging Face.
    """
    # If using local dataset
    if args and args.dataset_path:
        parquet_files = glob.glob(os.path.join(args.dataset_path, args.file_pattern))
        parquet_files.sort()
        
        if file_index < len(parquet_files):
            return parquet_files[file_index]
        else:
            print(f"Error: File index {file_index} is out of range. Only {len(parquet_files)} files available.")
            return None
    
    # Otherwise download from Hugging Face
    # The files are named chess_game_0001.parquet, chess_game_0002.parquet, etc.
    file_name = f"chess_game_{file_index:04d}.parquet"
    local_path = os.path.join(temp_dir, file_name)
    
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"File {file_name} already exists in {temp_dir}")
        return local_path
    
    # URL for the file on Hugging Face
    url = f"https://huggingface.co/datasets/laion/strategic_game_chess/resolve/main/{file_name}"
    
    try:
        print(f"Downloading {file_name} from Hugging Face dataset laion/strategic_game_chess...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {file_name} to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        return None

def train_model_streaming(model, file_indices, val_indices, args):
    """Train the neural network model by streaming parquet files one at a time."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training metrics
    metrics = {
        'train_policy_loss': [],
        'train_value_loss': [],
        'train_total_loss': [],
        'train_policy_accuracy': [],
        'val_policy_loss': [],
        'val_value_loss': [],
        'val_total_loss': [],
        'val_policy_accuracy': []
    }
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Load checkpoint if resuming training
    if args.resume and os.path.exists(args.output_model):
        print(f"Loading checkpoint from {args.output_model}")
        checkpoint = torch.load(args.output_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Load metrics if available
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
        
        print(f"Resuming from epoch {start_epoch}")
    
    # Training progress file
    progress_file = os.path.join(os.path.dirname(args.output_model), "training_progress.json")
    progress = {
        'processed_files': [],
        'current_epoch': start_epoch,
        'best_val_loss': best_val_loss
    }
    
    # Load progress if exists
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    
    # Create a temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory for downloads: {temp_dir}")
        
        # Main training loop
        for epoch in range(start_epoch, args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            progress['current_epoch'] = epoch
            
            # Process each training file
            for file_idx in file_indices:
                # Skip already processed files in this epoch
                file_key = f"epoch_{epoch}_file_{file_idx}"
                if file_key in progress['processed_files']:
                    print(f"Skipping already processed file index: {file_idx}")
                    continue
                
                # Get the file (download or local)
                file_path = download_parquet_file(file_idx, temp_dir, args)
                if not file_path:
                    print(f"Skipping file index {file_idx} due to error")
                    continue
                
                print(f"Training on file: {file_path}")
                
                # Create dataset and dataloader for this file
                train_dataset = ChessParquetDataset(file_path, args.samples_per_file)
                
                # Skip if no samples were processed
                if len(train_dataset) == 0:
                    print(f"No valid samples found in {file_path}, skipping to next file")
                    progress['processed_files'].append(file_key)
                    
                    # Save progress
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f)
                    
                    # Delete the file to save space if requested
                    if args.delete_after_processing and not args.dataset_path:
                        try:
                            os.remove(file_path)
                            print(f"Deleted file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting file: {e}")
                    
                    continue
                
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
                
                # Training phase
                model.train()
                train_policy_loss = 0
                train_value_loss = 0
                train_total_loss = 0
                correct_moves = 0
                total_moves = 0
                
                for inputs, move_idxs, values in tqdm(train_loader):
                    inputs = inputs.to(device)
                    # Convert move_idxs to tensor
                    move_idxs = torch.tensor(move_idxs, dtype=torch.long).to(device)
                    values = torch.tensor(values, dtype=torch.float).view(-1, 1).to(device)
                    
                    # Forward pass
                    policy_output, value_output = model(inputs)
                    
                    # Calculate policy loss (move prediction)
                    policy_loss = F.cross_entropy(policy_output, move_idxs)
                    
                    # Calculate value loss (position evaluation)
                    value_loss = F.mse_loss(value_output, values)
                    
                    # Combined loss
                    total_loss = policy_loss + value_loss
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    train_policy_loss += policy_loss.item()
                    train_value_loss += value_loss.item()
                    train_total_loss += total_loss.item()
                    
                    # Calculate accuracy
                    _, predicted_moves = torch.max(policy_output, 1)
                    correct_moves += (predicted_moves == move_idxs).sum().item()
                    total_moves += move_idxs.size(0)
                
                # Average training metrics for this file
                avg_train_policy_loss = train_policy_loss / len(train_loader)
                avg_train_value_loss = train_value_loss / len(train_loader)
                avg_train_total_loss = train_total_loss / len(train_loader)
                train_accuracy = 100 * correct_moves / total_moves
                
                metrics['train_policy_loss'].append(avg_train_policy_loss)
                metrics['train_value_loss'].append(avg_train_value_loss)
                metrics['train_total_loss'].append(avg_train_total_loss)
                metrics['train_policy_accuracy'].append(train_accuracy)
                
                print(f"Training - Policy Loss: {avg_train_policy_loss:.4f}, Value Loss: {avg_train_value_loss:.4f}, "
                      f"Total Loss: {avg_train_total_loss:.4f}, Move Accuracy: {train_accuracy:.2f}%")
                
                # Mark file as processed
                progress['processed_files'].append(file_key)
                
                # Save progress
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
                
                # Free memory
                del train_dataset
                del train_loader
                import gc
                gc.collect()
                
                # Delete the file to save space
                if args.delete_after_processing and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted file {file_path} to save space")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
                
                # Save checkpoint after each file
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'val_loss': best_val_loss
                }
                checkpoint_path = os.path.join(os.path.dirname(args.output_model), 
                                             f"checkpoint_epoch_{epoch+1}_file_{file_idx}.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Validation phase
            if val_indices:
                print("Running validation...")
                model.eval()
                val_policy_loss = 0
                val_value_loss = 0
                val_total_loss = 0
                val_correct_moves = 0
                val_total_moves = 0
                num_val_batches = 0
                
                with torch.no_grad():
                    for val_idx in val_indices:
                        # Get validation file
                        val_file = download_parquet_file(val_idx, temp_dir, args)
                        if not val_file:
                            print(f"Skipping validation file index {val_idx} due to error")
                            continue
                        
                        print(f"Validating on file: {val_file}")
                        
                        # Create validation dataset and dataloader
                        val_dataset = ChessParquetDataset(val_file, args.samples_per_file)
                        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
                        
                        for inputs, move_idxs, values in tqdm(val_loader):
                            inputs = inputs.to(device)
                            move_idxs = torch.tensor(move_idxs, dtype=torch.long).to(device)
                            values = torch.tensor(values, dtype=torch.float).to(device).view(-1, 1)
                            
                            # Forward pass
                            policy_output, value_output = model(inputs)
                            
                            # Calculate losses
                            policy_loss = F.cross_entropy(policy_output, move_idxs)
                            value_loss = F.mse_loss(value_output, values)
                            total_loss = policy_loss + value_loss
                            
                            # Accumulate losses
                            val_policy_loss += policy_loss.item() * inputs.size(0)
                            val_value_loss += value_loss.item() * inputs.size(0)
                            val_total_loss += total_loss.item() * inputs.size(0)
                            
                            # Calculate accuracy
                            _, predicted_moves = torch.max(policy_output, 1)
                            val_correct_moves += (predicted_moves == move_idxs).sum().item()
                            val_total_moves += move_idxs.size(0)
                            num_val_batches += 1
                        
                        # Free memory
                        if args.delete_after_processing and not args.dataset_path:
                            try:
                                os.remove(val_file)
                                print(f"Deleted validation file: {val_file}")
                            except Exception as e:
                                print(f"Error deleting validation file: {e}")
                
                # Average validation metrics
                if val_total_moves > 0:
                    avg_val_policy_loss = val_policy_loss / val_total_moves
                    avg_val_value_loss = val_value_loss / val_total_moves
                    avg_val_total_loss = val_total_loss / val_total_moves
                    val_accuracy = 100 * val_correct_moves / val_total_moves
                    
                    metrics['val_policy_loss'].append(avg_val_policy_loss)
                    metrics['val_value_loss'].append(avg_val_value_loss)
                    metrics['val_total_loss'].append(avg_val_total_loss)
                    metrics['val_policy_accuracy'].append(val_accuracy)
                    
                    print(f"Validation - Policy Loss: {avg_val_policy_loss:.4f}, Value Loss: {avg_val_value_loss:.4f}, "
                          f"Total Loss: {avg_val_total_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
                    
                    # Update learning rate scheduler
                    scheduler.step(avg_val_total_loss)
                    
                    # Save best model
                    if avg_val_total_loss < best_val_loss:
                        best_val_loss = avg_val_total_loss
                        progress['best_val_loss'] = best_val_loss
                        
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'val_loss': best_val_loss,
                            'metrics': metrics
                        }
                        
                        torch.save(checkpoint, args.output_model)
                        print(f"New best model saved to {args.output_model}")
                else:
                    print("No validation data was processed")
                
            # Save progress
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
                
    # Plot and save metrics
    plot_metrics(metrics, os.path.join(os.path.dirname(args.output_model), "training_metrics.png"))
    
    return model, metrics

def plot_metrics(metrics, save_path):
    """Plot and save training metrics."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot losses
    steps = list(range(1, len(metrics['train_policy_loss']) + 1))
    ax1.plot(steps, metrics['train_total_loss'], 'b-', label='Training Loss')
    ax1.plot(steps, metrics['val_total_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot policy and value losses
    ax2.plot(steps, metrics['train_policy_loss'], 'g-', label='Train Policy Loss')
    ax2.plot(steps, metrics['val_policy_loss'], 'y-', label='Val Policy Loss')
    ax2.plot(steps, metrics['train_value_loss'], 'c-', label='Train Value Loss')
    ax2.plot(steps, metrics['val_value_loss'], 'm-', label='Val Value Loss')
    ax2.set_title('Policy and Value Losses')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot accuracy
    ax3.plot(steps, metrics['train_policy_accuracy'], 'b-', label='Train Accuracy')
    ax3.plot(steps, metrics['val_policy_accuracy'], 'r-', label='Val Accuracy')
    ax3.set_title('Move Prediction Accuracy')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train a chess neural network on Stockfish games from parquet files')
    parser.add_argument('--dataset-path', type=str, default=None, 
                        help='Path to the directory containing parquet files (if None, files will be downloaded from Hugging Face)')
    parser.add_argument('--file-pattern', type=str, default='*.parquet', 
                        help='Pattern to match parquet files (only used if dataset-path is provided)')
    parser.add_argument('--start-file', type=int, default=1,
                        help='First file index to process when downloading from Hugging Face (1-indexed)')
    parser.add_argument('--end-file', type=int, default=20,
                        help='Last file index to process when downloading from Hugging Face (1-indexed)')
    parser.add_argument('--streaming', action='store_true', default=True,
                        help='Process files one at a time to save memory (default: True)')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Training batch size')
    parser.add_argument('--samples-per-file', type=int, default=100000, 
                        help='Maximum number of samples to process per file')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--model-name', type=str, default='base_model.pt',
                        help='Name of the model file to save (will be saved in saved_models directory)')
    parser.add_argument('--val-split', type=float, default=0.1, 
                        help='Fraction of files to use for validation')
    parser.add_argument('--residual-blocks', type=int, default=10, 
                        help='Number of residual blocks in the neural network')
    parser.add_argument('--channels', type=int, default=128, 
                        help='Number of channels in convolutional layers')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from checkpoint')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of worker processes for data loading')
    parser.add_argument('--delete-after-processing', action='store_true',
                        help='Delete parquet files after processing to save space')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='Save a checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set output model path
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
    os.makedirs(output_dir, exist_ok=True)
    args.output_model = os.path.join(output_dir, args.model_name)
    
    # Determine file indices to process
    if args.dataset_path:
        # Use local files
        parquet_files = glob.glob(os.path.join(args.dataset_path, args.file_pattern))
        parquet_files.sort()
        
        if not parquet_files:
            print(f"No parquet files found matching pattern {args.file_pattern} in {args.dataset_path}")
            return
        
        print(f"Found {len(parquet_files)} parquet files")
        file_indices = list(range(len(parquet_files)))
    else:
        # Use streaming mode with file indices from Hugging Face
        # Note: The files are 1-indexed (chess_game_0001.parquet, chess_game_0002.parquet, etc.)
        file_indices = list(range(args.start_file, args.end_file + 1))
        print(f"Using Hugging Face dataset with file indices {args.start_file} to {args.end_file}")
    
    # Split into train and validation
    num_val_files = max(1, int(len(file_indices) * args.val_split))
    val_indices = file_indices[:num_val_files]
    train_indices = file_indices[num_val_files:]
    
    print(f"Using {len(train_indices)} files for training and {len(val_indices)} files for validation")
    
    # Initialize the model
    model = ChessNet(residual_blocks=args.residual_blocks, channels=args.channels)
    
    # Train the model
    model, metrics = train_model_streaming(model, train_indices, val_indices, args)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 