import requests
import json
import time
import chess
from urllib.parse import urlencode
import re

class StockfishAPI:
    """
    Client for interacting with the Stockfish online API.
    """
    def __init__(self, base_url="https://stockfish.online/api/s/v2.php", timeout=10, max_retries=10):
        """
        Initialize the Stockfish API client.
        
        Args:
            base_url: Stockfish API URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for API calls
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
    def analyze_position(self, fen, depth=15):
        """
        Analyze a chess position using the Stockfish API.
        
        Args:
            fen: FEN string of the position
            depth: Search depth (1-20)
            
        Returns:
            Dictionary with analysis results or None if failed
        """
        # Validate depth
        if depth < 1:
            depth = 1
        elif depth > 20:
            depth = 20
            
        # Prepare parameters
        params = {
            'fen': fen,
            'depth': depth
        }
        
        # Initialize retry parameters
        retry_count = 0
        retry_delay = 2  # Start with 2 seconds
        max_delay = 30   # Maximum delay of 30 seconds
        
        while retry_count < self.max_retries:
            try:
                # Make request
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    return data
                elif response.status_code == 429:  # Too Many Requests
                    print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay = min(retry_delay + 1, max_delay)
                    retry_count += 1
                else:
                    print(f"API request failed with status code {response.status_code}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay = min(retry_delay + 1, max_delay)
                    retry_count += 1
            except requests.exceptions.Timeout:
                print(f"Request timed out. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay = min(retry_delay + 1, max_delay)
                retry_count += 1
            except requests.exceptions.ConnectionError:
                print(f"Connection error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay = min(retry_delay + 1, max_delay)
                retry_count += 1
            except Exception as e:
                print(f"Error connecting to Stockfish API: {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay = min(retry_delay + 1, max_delay)
                retry_count += 1
        
        print(f"Failed to connect to Stockfish API after {self.max_retries} retries.")
        return None
            
    def get_best_move(self, fen, depth=15):
        """
        Get the best move for a position.
        
        Args:
            fen: FEN string of the position
            depth: Search depth (1-20)
            
        Returns:
            Tuple of (best_move, score) or (None, None) if failed
            where best_move is a UCI string and score is in centipawns
        """
        # Analyze position
        analysis = self.analyze_position(fen, depth)
        
        if analysis and 'bestmove' in analysis:
            # Extract just the move part from the bestmove string
            # The API might return something like "bestmove e2e4 ponder d7d5"
            move_match = re.search(r'bestmove\s+(\w+)', analysis['bestmove'])
            if move_match:
                move = move_match.group(1)
            else:
                # Fallback: try to get the first word if regex fails
                move = analysis['bestmove'].split()[0]
                if move == 'bestmove':
                    # If the first word is 'bestmove', take the second word
                    parts = analysis['bestmove'].split()
                    move = parts[1] if len(parts) > 1 else None
            
            # Extract score if available
            score = None
            if 'info' in analysis and 'score' in analysis['info']:
                score_info = analysis['info']['score']
                
                # Handle different score types
                if 'cp' in score_info:
                    score = int(score_info['cp'])
                elif 'mate' in score_info:
                    mate_in = int(score_info['mate'])
                    if mate_in > 0:
                        score = 30000 - mate_in * 100  # Position close to mate
                    else:
                        score = -30000 - mate_in * 100  # Being mated
            
            return move, score
        
        return None, None
        
    def play_move(self, fen, depth=15):
        """
        Get Stockfish to play a move from the given position.
        
        Args:
            fen: FEN string of the position
            depth: Search depth (1-20)
            
        Returns:
            Tuple of (new_fen, move, score) or (None, None, None) if failed
        """
        # Get best move
        move_uci, score = self.get_best_move(fen, depth)
        
        if move_uci:
            try:
                # Create board from FEN
                board = chess.Board(fen)
                
                # Parse move
                move = chess.Move.from_uci(move_uci)
                
                # Make the move
                board.push(move)
                
                # Return new position
                return board.fen(), move_uci, score
            except Exception as e:
                print(f"Error applying move: {e}")
        
        return None, None, None
    
    def play_multiple_moves(self, fen, num_moves=5, depth=15):
        """
        Play a sequence of moves from the given position.
        
        Args:
            fen: Starting FEN string
            num_moves: Number of moves to play
            depth: Search depth
            
        Returns:
            List of (fen, move, score) tuples
        """
        results = []
        current_fen = fen
        
        for _ in range(num_moves):
            # Play a move
            new_fen, move, score = self.play_move(current_fen, depth)
            
            if new_fen:
                # Add to results
                results.append((current_fen, move, score))
                
                # Update current position
                current_fen = new_fen
                
                # Check if game is over
                board = chess.Board(new_fen)
                if board.is_game_over():
                    break
            else:
                # Stop if move failed
                break
        
        return results

class MockStockfishAPI:
    """
    Mock implementation of StockfishAPI for testing without network access.
    Uses a local chess engine when available.
    """
    def __init__(self, max_retries=10):
        """Initialize the mock Stockfish API client."""
        self.max_retries = max_retries
        try:
            from chess.engine import SimpleEngine
            import subprocess
            
            # Try to locate Stockfish executable
            stockfish_path = self._find_stockfish()
            
            if stockfish_path:
                # Initialize engine
                self.engine = SimpleEngine.popen_uci(stockfish_path)
                self.has_engine = True
            else:
                self.has_engine = False
        except:
            self.has_engine = False
    
    def _find_stockfish(self):
        """Try to find Stockfish executable on the system."""
        try:
            # Try to locate Stockfish using 'which' command on Unix
            import subprocess
            result = subprocess.run(['which', 'stockfish'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback to common paths
        import os
        paths = [
            '/usr/bin/stockfish',
            '/usr/local/bin/stockfish',
            'C:\\Program Files\\Stockfish\\stockfish.exe',
            os.path.expanduser('~/Stockfish/stockfish')
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def analyze_position(self, fen, depth=15):
        """Mock implementation of analyze_position."""
        if self.has_engine:
            try:
                # Create board from FEN
                board = chess.Board(fen)
                
                # Set depth limit
                limit = chess.engine.Limit(depth=depth)
                
                # Get analysis
                info = self.engine.analyse(board, limit)
                
                # Format result similarly to the API
                result = {
                    'bestmove': info['pv'][0].uci(),
                    'info': {
                        'depth': info['depth'],
                        'nodes': info.get('nodes', 0),
                        'time': info.get('time', 0)
                    }
                }
                
                # Add score information
                if 'score' in info:
                    score = info['score']
                    result['info']['score'] = {}
                    
                    if score.is_mate():
                        result['info']['score']['mate'] = score.mate()
                    else:
                        result['info']['score']['cp'] = score.cp
                
                return result
            except Exception as e:
                print(f"Error in mock analysis: {e}")
                
        # If no engine or analysis failed, return a simple random move
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        if legal_moves:
            import random
            move = random.choice(legal_moves)
            
            return {
                'bestmove': move.uci(),
                'info': {
                    'depth': 1,
                    'score': {'cp': 0}
                }
            }
        
        return None
    
    def get_best_move(self, fen, depth=15):
        """Mock implementation of get_best_move."""
        analysis = self.analyze_position(fen, depth)
        
        if analysis and 'bestmove' in analysis:
            move = analysis['bestmove']
            
            # Extract score if available
            score = 0
            if 'info' in analysis and 'score' in analysis['info']:
                score_info = analysis['info']['score']
                
                if 'cp' in score_info:
                    score = score_info['cp']
                elif 'mate' in score_info:
                    mate_in = score_info['mate']
                    if mate_in > 0:
                        score = 30000 - mate_in * 100
                    else:
                        score = -30000 - mate_in * 100
            
            return move, score
        
        return None, None
    
    def play_move(self, fen, depth=15):
        """Mock implementation of play_move."""
        # Get best move
        move_uci, score = self.get_best_move(fen, depth)
        
        if move_uci:
            try:
                # Create board from FEN
                board = chess.Board(fen)
                
                # Parse move
                move = chess.Move.from_uci(move_uci)
                
                # Make the move
                board.push(move)
                
                # Return new position
                return board.fen(), move_uci, score
            except Exception as e:
                print(f"Error applying move: {e}")
        
        return None, None, None
    
    def play_multiple_moves(self, fen, num_moves=5, depth=15):
        """Mock implementation of play_multiple_moves."""
        results = []
        current_fen = fen
        
        for _ in range(num_moves):
            # Play a move
            new_fen, move, score = self.play_move(current_fen, depth)
            
            if new_fen:
                # Add to results
                results.append((current_fen, move, score))
                
                # Update current position
                current_fen = new_fen
                
                # Check if game is over
                board = chess.Board(new_fen)
                if board.is_game_over():
                    break
            else:
                # Stop if move failed
                break
        
        return results
    
    def __del__(self):
        """Clean up engine when object is destroyed."""
        if hasattr(self, 'engine') and self.has_engine:
            try:
                self.engine.quit()
            except:
                pass
