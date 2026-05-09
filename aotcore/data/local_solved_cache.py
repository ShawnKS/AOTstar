"""
Run-local cache for solved one-step retrosynthesis decompositions.

The cache is stored in a SQLite file under the current run's result directory
and mirrored into RAM for fast lookup during search. It is shared only by the
workers and budget rounds belonging to that run.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional


class LocalSolvedRouteCache:
    """SQLite-backed per-run solved cache with RAM mirror for runtime lookups."""

    def __init__(self, db_path: str, preload_to_ram: bool = True):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, List[str]] = {}
        self._row_count = 0
        self._max_updated_at = 0.0

        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA cache_size = -16384")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS solved_routes (
                product_smiles TEXT PRIMARY KEY,
                reactants_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )

        if preload_to_ram:
            self._reload_from_disk_locked()

    def _db_state_locked(self) -> tuple:
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(updated_at), 0) FROM solved_routes"
        ).fetchone()
        return int(row[0]), float(row[1])

    def _reload_from_disk_locked(self):
        rows = self.conn.execute(
            "SELECT product_smiles, reactants_json, updated_at FROM solved_routes"
        ).fetchall()

        memory_cache: Dict[str, List[str]] = {}
        max_updated_at = 0.0

        for product_smiles, reactants_json, updated_at in rows:
            reactants = json.loads(reactants_json)
            if isinstance(reactants, list) and reactants:
                memory_cache[product_smiles] = reactants
            if updated_at > max_updated_at:
                max_updated_at = float(updated_at)

        self._memory_cache = memory_cache
        self._row_count = len(memory_cache)
        self._max_updated_at = max_updated_at

    def refresh_if_changed(self):
        """Reload RAM mirror only when the SQLite content changed."""
        with self._lock:
            row_count, max_updated_at = self._db_state_locked()
            if row_count != self._row_count or max_updated_at > self._max_updated_at:
                self._reload_from_disk_locked()

    def get(self, product_smiles: str) -> Optional[List[str]]:
        """Return cached reactants list for product_smiles, if present."""
        with self._lock:
            reactants = self._memory_cache.get(product_smiles)
            if reactants is None:
                return None
            return list(reactants)

    def put(self, product_smiles: str, reactants: List[str]):
        """Upsert one solved one-step decomposition."""
        if not product_smiles or not reactants:
            return

        updated_at = time.time()
        reactants_json = json.dumps(list(reactants), separators=(",", ":"))

        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO solved_routes
                (product_smiles, reactants_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (product_smiles, reactants_json, updated_at),
            )
            self._memory_cache[product_smiles] = list(reactants)
            self._row_count = len(self._memory_cache)
            if updated_at > self._max_updated_at:
                self._max_updated_at = updated_at

    def close(self):
        with self._lock:
            self.conn.close()

    def __len__(self) -> int:
        with self._lock:
            return len(self._memory_cache)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
