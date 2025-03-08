import time
import sys
from collections import OrderedDict

class TranspositionTable:
    """
    Memory-efficient transposition table with LRU replacement strategy.
    """
    def __init__(self, max_size_mb=64):
        """
        Initialize a new transposition table.
        
        Args:
            max_size_mb: Maximum size in megabytes
        """
        self.max_entries = int((max_size_mb * 1024 * 1024) / 64)  # Rough estimate of entry size
        self.table = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def store(self, key, depth, score, move, flag, age):
        """
        Store a position in the transposition table.
        
        Args:
            key: Zobrist hash of the position
            depth: Search depth
            score: Evaluation score
            move: Best move
            flag: Node type (exact, alpha, beta)
            age: Current search age for replacement
        """
        # Check if table is full
        if len(self.table) >= self.max_entries:
            # Remove oldest entry (LRU strategy)
            self.table.popitem(last=False)
        
        # Store the entry
        self.table[key] = {
            'depth': depth,
            'score': score,
            'move': move,
            'flag': flag,
            'age': age
        }
        
        # Move to the end (most recently used)
        self.table.move_to_end(key)
    
    def probe(self, key):
        """
        Probe the transposition table for a position.
        
        Args:
            key: Zobrist hash of the position
            
        Returns:
            Entry if found, None otherwise
        """
        entry = self.table.get(key)
        
        if entry is not None:
            # Update statistics
            self.hits += 1
            # Move to the end (most recently used)
            self.table.move_to_end(key)
            return entry
        else:
            # Update statistics
            self.misses += 1
            return None
    
    def get_move(self, key):
        """
        Get the best move for a position.
        
        Args:
            key: Zobrist hash of the position
            
        Returns:
            Best move if found, None otherwise
        """
        entry = self.table.get(key)
        if entry is not None:
            return entry.get('move')
        return None
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
    
    def get_stats(self):
        """
        Get statistics about the transposition table.
        
        Returns:
            Dictionary with statistics
        """
        total_lookups = self.hits + self.misses
        hit_rate = (self.hits / total_lookups) * 100 if total_lookups > 0 else 0
        
        return {
            'entries': len(self.table),
            'max_entries': self.max_entries,
            'usage': f"{len(self.table) / self.max_entries * 100:.1f}%",
            'hits': self.hits,
            'misses': self.misses, 
            'hit_rate': f"{hit_rate:.1f}%",
            'collisions': self.collisions
        }

class PositionCache:
    """
    Cache for storing preprocessed positions.
    """
    def __init__(self, max_size=1000):
        """
        Initialize a position cache.
        
        Args:
            max_size: Maximum number of positions to store
        """
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def store(self, fen, features):
        """
        Store a position's features.
        
        Args:
            fen: FEN string of the position
            features: Preprocessed features
        """
        # Check if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
        
        # Store the entry
        self.cache[fen] = features
        
        # Move to the end (most recently used)
        self.cache.move_to_end(fen)
    
    def get(self, fen):
        """
        Get a position's features from the cache.
        
        Args:
            fen: FEN string of the position
            
        Returns:
            Features if found, None otherwise
        """
        features = self.cache.get(fen)
        
        if features is not None:
            # Move to the end (most recently used)
            self.cache.move_to_end(fen)
            
        return features
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()

class MemoryManager:
    """
    Manage memory usage for the chess engine.
    """
    def __init__(self, tt_size_mb=64, position_cache_size=1000):
        """
        Initialize memory manager.
        
        Args:
            tt_size_mb: Transposition table size in megabytes
            position_cache_size: Number of positions to cache
        """
        self.transposition_table = TranspositionTable(tt_size_mb)
        self.position_cache = PositionCache(position_cache_size)
        self.start_time = time.time()
    
    def clear_all(self):
        """Clear all memory structures."""
        self.transposition_table.clear()
        self.position_cache.clear()
    
    def get_memory_usage(self):
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        tt_stats = self.transposition_table.get_stats()
        
        return {
            'tt_entries': tt_stats['entries'],
            'tt_usage': tt_stats['usage'],
            'hit_rate': tt_stats['hit_rate'],
            'cache_size': len(self.position_cache.cache),
            'uptime': int(time.time() - self.start_time)
        }
