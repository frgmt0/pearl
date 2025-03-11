#!/usr/bin/env python3
import os
import csv
import chess.pgn
import io

def extract_move_sequence(game):
    """Extract the move sequence from a chess game."""
    moves = []
    board = game.board()
    
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    
    return " ".join(moves)

def extract_result(game):
    """Extract the result from a chess game."""
    result = game.headers.get("Result", "*")
    return result

def process_pgn_file(pgn_file_path, output_csv_path, append=True):
    """Process a PGN file and extract move sequences and results."""
    # Check if output file exists and determine mode
    file_exists = os.path.isfile(output_csv_path)
    mode = 'a' if append and file_exists else 'w'
    
    print(f"Opening PGN file: {pgn_file_path}")
    
    try:
        # Open the PGN file
        with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
            pgn_content = pgn_file.read()
            
            # Check if the file is empty or has invalid content
            if not pgn_content.strip():
                print(f"Warning: {pgn_file_path} is empty or contains only whitespace.")
                return 0
            
            # Create a StringIO object to parse the PGN content
            pgn_io = io.StringIO(pgn_content)
            
            # Open the CSV file
            with open(output_csv_path, mode, newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # Write header if file is new or not appending
                if not (append and file_exists):
                    csv_writer.writerow(['move_sequence', 'result'])
                
                # Process each game in the PGN file
                games_processed = 0
                
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_io)
                        if game is None:
                            break
                        
                        # Extract move sequence and result
                        move_sequence = extract_move_sequence(game)
                        result = extract_result(game)
                        
                        print(f"Game {games_processed + 1}: Move sequence length: {len(move_sequence)}, Result: {result}")
                        
                        # Skip games with empty move sequences
                        if not move_sequence:
                            print(f"Warning: Game {games_processed + 1} has an empty move sequence. Skipping.")
                            continue
                        
                        # Write to CSV
                        csv_writer.writerow([move_sequence, result])
                        
                        games_processed += 1
                        
                        # Print progress every 10 games
                        if games_processed % 10 == 0:
                            print(f"Processed {games_processed} games from {os.path.basename(pgn_file_path)}")
                    
                    except Exception as e:
                        print(f"Error processing game: {e}")
                        # Try to continue with the next game
                        continue
                
                print(f"Finished processing {games_processed} games from {os.path.basename(pgn_file_path)}")
        
        return games_processed
    
    except Exception as e:
        print(f"Error opening or processing file {pgn_file_path}: {e}")
        return 0

def main():
    # Define paths
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    pgn_dir = os.path.join(dataset_dir, 'mastergames')
    output_csv_path = os.path.join(dataset_dir, 'dataset.csv')
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"PGN directory: {pgn_dir}")
    print(f"Output CSV path: {output_csv_path}")
    
    # Get all PGN files in the directory
    pgn_files = [os.path.join(pgn_dir, f) for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        print("No PGN files found in the directory.")
        return
    
    print(f"Found {len(pgn_files)} PGN files: {[os.path.basename(f) for f in pgn_files]}")
    
    # Check if dataset.csv exists and has a header
    file_exists = os.path.isfile(output_csv_path)
    has_header = False
    
    if file_exists:
        try:
            with open(output_csv_path, 'r') as f:
                first_line = f.readline().strip()
                has_header = first_line == 'move_sequence,result'
                
            if has_header:
                print(f"Existing dataset.csv found with header. Will append to it.")
            else:
                print(f"Existing dataset.csv found but without proper header. Will create header.")
        except Exception as e:
            print(f"Error checking existing file: {e}")
    else:
        print(f"No existing dataset.csv found. Will create new file with header.")
    
    total_games = 0
    
    # Process all files in append mode
    for i, pgn_file in enumerate(pgn_files):
        print(f"Processing {os.path.basename(pgn_file)}...")
        
        # For the first file, we need to create a header if the file doesn't exist or doesn't have a header
        if i == 0 and (not file_exists or not has_header):
            games = process_pgn_file(pgn_file, output_csv_path, append=False)
        else:
            games = process_pgn_file(pgn_file, output_csv_path, append=True)
            
        total_games += games
    
    print(f"Total games processed: {total_games}")
    print(f"Data has been saved to {output_csv_path}")
    
    # Verify the output file
    try:
        with open(output_csv_path, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"The output file contains {line_count} lines (including header).")
    except Exception as e:
        print(f"Error verifying output file: {e}")

if __name__ == "__main__":
    main() 