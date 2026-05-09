"""
Memory-efficient SQLite-backed inventory for molecule lookups.

This module provides a disk-backed inventory implementation that uses SQLite
instead of loading all molecules into RAM. Reduces memory usage from 5-8GB
to 500MB-1GB while maintaining compatibility with syntheseus.MolInventory.

Usage:
    from aotcore.data.inventory_mmap import SQLiteInventory

    inventory = SQLiteInventory('./dataset/inventory.db')

    # Check if molecule exists (disk-backed with cache)
    if 'CC(=O)O' in inventory:
        print("Molecule found!")
"""

import os
import sqlite3
from typing import Optional
from functools import lru_cache
from syntheseus.search.mol_inventory import BaseMolInventory


class SQLiteInventory(BaseMolInventory):
    """
    Disk-backed inventory using SQLite with LRU caching.

    This class provides memory-efficient molecule inventory lookups by storing
    molecules in a SQLite database instead of RAM. It uses an LRU cache to
    speed up repeated lookups of the same molecules.

    Memory usage:
        - Without cache: ~100-200 MB (just SQLite connection)
        - With cache (10k entries): ~500 MB - 1 GB
        - Original pickle approach: 5-8 GB

    Performance:
        - First lookup: ~1-5 ms (disk read)
        - Cached lookup: ~0.001 ms (memory)
        - Cache hit rate in typical search: 60-80%
    """

    def __init__(self, db_path: str, cache_size: int = 10000):
        """
        Initialize SQLite-backed inventory.

        Args:
            db_path: Path to SQLite database file (created by convert_inventory.py)
            cache_size: Maximum number of molecules to cache in memory (default: 10000)
                       10k entries ≈ 50-100 MB depending on SMILES length
        """
        self.db_path = db_path
        self.cache_size = cache_size

        # Create connection with optimizations
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,  # Allow usage across threads
            isolation_level=None       # Autocommit mode (read-only anyway)
        )

        # Performance optimizations for read-only database on shared filesystem
        # Note: Avoid WAL mode on GPFS/network filesystems
        self.conn.execute("PRAGMA synchronous = OFF")  # Fastest (safe for read-only)
        self.conn.execute("PRAGMA cache_size = -64000")   # 64MB cache
        self.conn.execute("PRAGMA temp_store = MEMORY")  # Temp tables in RAM
        self.conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O

        # Create cached lookup function
        self._cached_lookup = lru_cache(maxsize=cache_size)(self._db_contains)

        # Verify database is valid. On worker benchmarks with many forked
        # interpreters, counting the whole inventory in every child causes
        # avoidable GPFS/SQLite startup contention.
        try:
            if os.environ.get("AOT_SQLITE_INVENTORY_SKIP_COUNT", "0") == "1":
                self.conn.execute("SELECT 1 FROM inventory LIMIT 1").fetchone()
                print(f"SQLiteInventory initialized from {db_path} (count skipped)")
            else:
                cursor = self.conn.execute("SELECT COUNT(*) FROM inventory")
                count = cursor.fetchone()[0]
                print(f"SQLiteInventory initialized: {count:,} molecules in database")
        except sqlite3.OperationalError as e:
            raise ValueError(
                f"Invalid inventory database at {db_path}. "
                f"Please run convert_inventory.py first. Error: {e}"
            )

    def _db_contains(self, smiles: str) -> bool:
        """
        Internal method: Check if SMILES exists in database.

        This method is wrapped by LRU cache for performance.

        Args:
            smiles: SMILES string to lookup

        Returns:
            True if molecule exists in inventory, False otherwise
        """
        cursor = self.conn.execute(
            "SELECT 1 FROM inventory WHERE smiles = ? LIMIT 1",
            (smiles,)
        )
        return cursor.fetchone() is not None

    def __contains__(self, mol) -> bool:
        """
        Check if molecule exists in inventory (MolInventory interface).

        This method uses LRU cache to avoid repeated database queries
        for the same molecules.

        Args:
            mol: Either a SMILES string or a Molecule object with .smiles attribute

        Returns:
            True if molecule exists in inventory, False otherwise

        Example:
            >>> inventory = SQLiteInventory('./dataset/inventory.db')
            >>> 'CC(=O)O' in inventory
            True
        """
        # Extract SMILES from Molecule object if needed
        if hasattr(mol, 'smiles'):
            smiles = mol.smiles
        elif hasattr(mol, 'to_smiles'):
            smiles = mol.to_smiles()
        elif isinstance(mol, str):
            smiles = mol
        else:
            smiles = str(mol)

        return self._cached_lookup(smiles)

    def is_purchasable(self, mol) -> bool:
        """
        Check if molecule is purchasable (alias for __contains__).

        Required by BaseMolInventory interface.

        Args:
            mol: Either a SMILES string or a Molecule object

        Returns:
            True if molecule exists in inventory, False otherwise
        """
        return self.__contains__(mol)

    def get_cache_info(self) -> dict:
        """
        Get cache statistics for monitoring performance.

        Returns:
            Dictionary with cache statistics:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - size: Current cache size
                - maxsize: Maximum cache size
                - hit_rate: Cache hit rate (0-1)
        """
        info = self._cached_lookup.cache_info()
        total = info.hits + info.misses
        hit_rate = info.hits / total if total > 0 else 0.0

        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize,
            'hit_rate': hit_rate
        }

    def clear_cache(self):
        """Clear the LRU cache (useful for memory pressure situations)."""
        self._cached_lookup.cache_clear()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __del__(self):
        """Cleanup: close connection when object is garbage collected."""
        try:
            self.conn.close()
        except:
            pass

    def __repr__(self) -> str:
        """String representation for debugging."""
        cache_info = self.get_cache_info()
        return (
            f"SQLiteInventory(db_path='{self.db_path}', "
            f"cache_size={self.cache_size}, "
            f"cache_hit_rate={cache_info['hit_rate']:.1%})"
        )


class ShelveInventory(BaseMolInventory):
    """
    Alternative disk-backed inventory using Python's shelve module.

    Simpler than SQLite but potentially slower for lookups.
    Kept as fallback option if SQLite has compatibility issues.

    Usage:
        inventory = ShelveInventory('./dataset/inventory.shelve')
    """

    def __init__(self, shelve_path: str):
        """
        Initialize shelve-backed inventory.

        Args:
            shelve_path: Path to shelve database file (without extension)
        """
        import shelve
        self.db = shelve.open(shelve_path, flag='r')  # Read-only
        self.cache = {}  # Simple dict cache

    def __contains__(self, smiles: str) -> bool:
        """Check if molecule exists in inventory."""
        # Check cache first
        if smiles in self.cache:
            return self.cache[smiles]

        # Check database
        result = smiles in self.db

        # Cache result
        if len(self.cache) < 10000:  # Limit cache size
            self.cache[smiles] = result

        return result

    def close(self):
        """Close database."""
        self.db.close()

    def __del__(self):
        """Cleanup."""
        try:
            self.db.close()
        except:
            pass
