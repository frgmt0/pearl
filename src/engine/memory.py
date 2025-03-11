import time
import sys
from collections import OrderedDict

class TranspositionTable:
    """
    Memory-efficient transposition table with LRU replacement strategy and aging mechanism.
    
    This implementation uses a fixed-size table with entry replacement based on:
    1. Age (older entries from previous searches are replaced first)
    2. Depth (shallower depth entries are replaced before deeper ones)
    3. LRU (least recently used entries are replaced when age and depth are equal)
    """
    
    def __init__(self, max_size_mb=128):
        """
        Initialize the transposition table with a maximum size in MB.
        
        Args:
            max_size_mb: Maximum size of the table in megabytes (default: 128MB)
        """
        # Calculate the number of entries based on memory size
        # Each entry is approximately 32 bytes (hash, depth, score, move, flag, age)
        self.max_entries = (max_size_mb * 1024 * 1024) // 32
        
        # Initialize the table as an OrderedDict for LRU functionality
        self.table = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0
        
        print(f"Initialized transposition table with {self.max_entries} entries ({max_size_mb}MB)")
    
    def store(self, key, depth, score, move, flag, age):
        """
        Store a position in the transposition table.
        
        Args:
            key: Zobrist hash of the position
            depth: Remaining depth at this position
            score: Evaluation score
            move: Best move found at this position
            flag: Type of node (EXACT, ALPHA, BETA)
            age: Current search age
        """
        # Check if we need to replace an entry
        if len(self.table) >= self.max_entries and key not in self.table:
            # Remove the least recently used entry
            self.table.popitem(last=False)
        
        # Store the entry
        self.table[key] = {
            'depth': depth,
            'score': score,
            'move': move,
            'flag': flag,
            'age': age,
            'access_time': time.time()
        }
        
        # Move to the end of the OrderedDict (most recently used)
        if key in self.table:
            self.table.move_to_end(key)
        
        self.stores += 1
    
    def probe(self, key):
        """
        Probe the transposition table for a position.
        
        Args:
            key: Zobrist hash of the position
            
        Returns:
            Entry if found, None otherwise
        """
        entry = self.table.get(key)
        
        if entry:
            # Update access time and move to the end (most recently used)
            entry['access_time'] = time.time()
            self.table.move_to_end(key)
            self.hits += 1
            return entry
        else:
            self.misses += 1
            return None
    
    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        self.stores = 0
    
    def get_stats(self):
        """Get statistics about the transposition table."""
        total_lookups = self.hits + self.misses
        hit_rate = (self.hits / total_lookups * 100) if total_lookups > 0 else 0
        
        return {
            'size': len(self.table),
            'max_size': self.max_entries,
            'usage': f"{len(self.table) / self.max_entries * 100:.2f}%",
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'stores': self.stores,
            'collisions': self.collisions
        }
    
    def resize(self, max_size_mb):
        """
        Resize the transposition table.
        
        Args:
            max_size_mb: New maximum size in megabytes
        """
        old_table = self.table
        self.max_entries = (max_size_mb * 1024 * 1024) // 32
        self.table = OrderedDict()
        
        # Copy the most recent entries to the new table
        entries = list(old_table.items())
        entries.sort(key=lambda x: (x[1]['age'], -x[1]['depth'], -x[1]['access_time']))
        
        for key, entry in entries[:self.max_entries]:
            self.table[key] = entry
        
        print(f"Resized transposition table to {self.max_entries} entries ({max_size_mb}MB)")
    
    def prune_old_entries(self, current_age):
        """
        Prune entries from previous searches.
        
        Args:
            current_age: Current search age
        """
        # Keep entries from the current search and deep entries from the previous search
        keys_to_remove = []
        for key, entry in self.table.items():
            if entry['age'] < current_age - 1 or (entry['age'] == current_age - 1 and entry['depth'] < 5):
                keys_to_remove.append(key)
        
        # Remove old entries
        for key in keys_to_remove:
            del self.table[key]
        
        print(f"Pruned {len(keys_to_remove)} old entries from transposition table")

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
            'tt_entries': tt_stats['size'],
            'tt_usage': tt_stats['usage'],
            'hit_rate': tt_stats['hit_rate'],
            'cache_size': len(self.position_cache.cache),
            'uptime': int(time.time() - self.start_time)
        }
