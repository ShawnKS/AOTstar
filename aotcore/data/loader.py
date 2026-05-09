"""
Data Loading Utilities for Retrosynthesis Search

Runtime default:
- Load inventory.pkl into RAM for fast purchasability checks.
- Uses shelve for template_dict unless RAM templates are explicitly requested.
- Removes unused test data loading.
- Implements aggressive garbage collection.
"""

import pickle
import shelve
import ast
import gc
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_dir() -> Path:
    """Return the dataset directory used by the current repository."""
    return Path(os.environ.get("AOT_DATASET_DIR", REPO_ROOT / "dataset")).expanduser()


class LegacyMolSetInventory:
    """Compatibility wrapper for old pickled SmilesListInventory objects.

    Older inventory.pkl files store a set of Molecule objects in ``_mol_set``.
    Newer syntheseus versions expect ``_smiles_set``.  Wrapping avoids building
    another 20M+ string set while preserving the old RAM lookup semantics.
    """

    def __init__(self, mol_set):
        self._mol_set = mol_set

    def is_purchasable(self, mol) -> bool:
        from syntheseus import Molecule

        if isinstance(mol, str):
            mol = Molecule(mol)
        return mol in self._mol_set

    def __contains__(self, mol) -> bool:
        return self.is_purchasable(mol)

    def __len__(self) -> int:
        return len(self._mol_set)

    def to_purchasable_mols(self):
        return self._mol_set


def _repair_legacy_ram_inventory(inventory):
    if hasattr(inventory, "_mol_set") and not hasattr(inventory, "_smiles_set"):
        return LegacyMolSetInventory(inventory._mol_set)
    return inventory


def load_inventory(inventory_path: str):
    """
    Load molecule inventory.

    Args:
        inventory_path: Path to inventory pickle or, explicitly, SQLite .db.

    Returns:
        In-memory pickle inventory for .pkl paths. SQLiteInventory is used only
        when the caller explicitly passes a .db path.
    """
    if inventory_path.endswith('.db'):
        from aotcore.data.inventory_mmap import SQLiteInventory

        inventory = SQLiteInventory(inventory_path)
        print(f"Loaded SQLite inventory from {inventory_path}")
        return inventory

    print(f"Loading RAM inventory from {inventory_path}...")
    with open(inventory_path, 'rb') as handle:
        inventory = pickle.load(handle)
    inventory = _repair_legacy_ram_inventory(inventory)
    try:
        count_text = f" ({len(inventory):,} entries)"
    except Exception:
        count_text = ""
    print(f"Loaded RAM inventory from {inventory_path}{count_text}")
    return inventory


def load_local_solved_cache(cache_path: Optional[str] = None,
                             preload_to_ram: bool = True):
    """Load a run-local solved cache with RAM mirroring."""
    from aotcore.data.local_solved_cache import LocalSolvedRouteCache

    if cache_path is None:
        raise ValueError("local solved cache path must be set by the current run")

    cache = LocalSolvedRouteCache(cache_path, preload_to_ram=preload_to_ram)
    print(f"Loaded local solved cache from {cache_path} ({len(cache):,} entries)")
    return cache


def load_dataset_targets(dataset_name: str) -> List[str]:
    """Load target molecules for specified dataset"""
    base_path = dataset_dir()

    dataset_paths = {
        'pistachio_hard': base_path / "pistachio_hard_targets.txt",
        'pis_hard': base_path / "pis_hard_remaining.txt",
        'pistachio_reachable': base_path / "pistachio_reachable_targets.txt",
        'pistachio_reachable_hard': base_path / "pistachio_reachable_hard.txt",
        'uspto_190': base_path / "uspto_190_targets.txt",
        'pistachio_easy': base_path / "uspto_easy.txt",
    }
    
    if dataset_name not in dataset_paths:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    file_path = dataset_paths[dataset_name]
    targets = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                tuple_data = ast.literal_eval(line.strip())
                targets.append(tuple_data[0])
    
    return targets


def load_training_data(use_ram_templates: bool = False) -> tuple:
    """
    Load training data from pickle files with memory optimizations.

    Args:
        use_ram_templates: If True, load templates to RAM (fast but uses ~4GB)
                          If False, use shelve (slow but memory-efficient)

    Memory optimizations:
    - Removed test data loading (saves ~300MB)
    - Uses shelve for template_dict by default (saves 2-4GB)
    - Optional RAM mode for 5-10x faster template lookups
    - Explicit file closing and garbage collection (saves 500MB-1GB)

    Returns:
        Tuple containing:
            - route_list: Combined train+val routes for RAG examples
            - all_fps: Fingerprints for vectorized similarity (required)
            - reaction_list: Reaction templates for validation
            - all_reaction_fps: Reaction fingerprints for similarity
            - datasub: Substrate data for fallback
            - template_dict: Template dictionary (dict if RAM mode, shelve if disk mode)

    Memory usage:
        - Shelve mode: ~2-4GB (default)
        - RAM mode: ~6-8GB (5-10x faster)
    """
    base_path = dataset_dir()

    # Load route data (train + val only, NO test data)
    train_data = base_path / 'routes_train.pkl'
    val_data = base_path / 'routes_val.pkl'

    print("Loading training routes...")
    with open(train_data, 'rb') as f:
        train_routes = pickle.load(f)

    print("Loading validation routes...")
    with open(val_data, 'rb') as f:
        val_routes = pickle.load(f)

    # Combine training and validation routes
    print("Combining train+val routes...")
    total_routes = train_routes + val_routes

    # Clear intermediate references and force garbage collection
    del train_routes
    del val_routes
    gc.collect()

    # Extract route list
    route_list = []
    for route in total_routes:
        route_list.append(route)

    # Clear total_routes after extraction
    del total_routes
    gc.collect()

    # Load fingerprints (required for vectorized operations, cannot be optimized)
    print("Loading fingerprints...")
    with open(base_path / 'all_fps.pkl', 'rb') as f:
        all_fps = pickle.load(f)

    with open(base_path / 'all_reaction_fps.pkl', 'rb') as f:
        all_reaction_fps = pickle.load(f)

    # Load reaction list
    print("Loading reaction list...")
    with open(base_path / 'reaction_list.pkl', 'rb') as f:
        reaction_list = pickle.load(f)

    # Load datasub
    print("Loading datasub...")
    with open(base_path / 'datasub.pkl', 'rb') as f:
        datasub = pickle.load(f)

    # Load template_dict based on flag
    print("Loading template dictionary...")
    template_dict_path = base_path / 'template_dict.shelve'
    template_dict_pkl = base_path / 'template_dict.pkl'

    if use_ram_templates:
        # RAM mode: Fast lookups, higher memory
        print("  Mode: RAM-based (fast, ~4GB memory)")
        try:
            with open(template_dict_pkl, 'rb') as f:
                template_dict = pickle.load(f)
            print(f"  ✓ Loaded template_dict to RAM from {template_dict_pkl}")
        except Exception as e:
            print(f"  ⚠️  Failed to load pickle: {e}")
            print("  Falling back to shelve...")
            template_dict = shelve.open(str(template_dict_path), flag='r')
    else:
        # Shelve mode: Slow lookups, lower memory (current default)
        print("  Mode: Shelve-backed (memory-efficient, slower)")
        try:
            template_dict = shelve.open(str(template_dict_path), flag='r')
            print(f"  ✓ Loaded shelve-backed template_dict from {template_dict_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to open shelve: {e}")
            print("  Falling back to pickle...")
            with open(template_dict_pkl, 'rb') as f:
                template_dict = pickle.load(f)

    # Final garbage collection
    gc.collect()

    print("Data loading complete!")
    print(f"  - Routes: {len(route_list):,}")
    print(f"  - Fingerprints: {len(all_fps):,}")
    print(f"  - Reactions: {len(reaction_list):,}")
    print(f"  - Template dict type: {type(template_dict).__name__}")

    return (route_list, all_fps, reaction_list, all_reaction_fps,
            datasub, template_dict)


def initialize_test_inventory():
    """Initialize a test inventory for debugging"""
    from syntheseus.search.mol_inventory import SmilesListInventory
    
    test_smiles = [
        'CC', 'CCO', 'c1ccccc1', 'CC(=O)O', 'CN', 'C=O',
        'c1ccc(Cl)cc1', 'c1ccc(Br)cc1', 'c1ccc(O)cc1', 'c1cccnc1',
        'CCC', 'CCCO', 'c1ccc(N)cc1', 'c1ccc(C=O)cc1', 'CCN',
        'c1cnccn1', 'C1CCNCC1', 'CS(=O)(=O)Cl', 'CC(=O)Cl'
    ]
    
    return SmilesListInventory(test_smiles)
