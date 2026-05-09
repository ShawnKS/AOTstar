"""
AND-OR Tree Node Definitions for LLM-Guided Retrosynthesis
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import math


@dataclass(eq=False)
class ORNode:
    """OR Node - Represents a molecule in the synthesis tree"""
    molecule: str
    smiles: str
    is_solved: bool = False
    
    # Child and parent nodes
    child_reactions: List['ANDNode'] = field(default_factory=list)
    parent_reactions: List['ANDNode'] = field(default_factory=list)
    
    # Reaction caching mechanism
    cached_reactions: List[any] = field(default_factory=list)
    used_reaction_indices: Set[int] = field(default_factory=set)
    reactions_initialized: bool = False
    
    # Compound stock check status
    compound_stock_checked: bool = False  # Has compound stock been checked
    last_check_failed: bool = False       # Did last check fail
    
    solving_and_node: Optional['ANDNode'] = None
    
    def get_available_reactions(self) -> List[Tuple[int, any]]:
        """Get still available reactions (returns (index, reaction) pairs)"""
        available = []
        for i, reaction in enumerate(self.cached_reactions):
            if i not in self.used_reaction_indices:
                available.append((i, reaction))
        return available
    
    def mark_reaction_used(self, reaction_index: int):
        """Mark reaction as used"""
        self.used_reaction_indices.add(reaction_index)
    
    def has_available_reactions(self) -> bool:
        """Check if there are still available reactions"""
        return len(self.used_reaction_indices) < len(self.cached_reactions)


@dataclass  
class ANDNode:
    """AND Node - Represents a chemical reaction"""
    reaction_id: str
    reactants: List[str]  # List of reactant SMILES
    product: str  # Product SMILES
    child_molecules: List[ORNode] = field(default_factory=list)  # OR nodes for reactants
    parent_molecule: Optional[ORNode] = None  # OR node for product
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    average_value: float = 0.0
    feasibility_score: float = 0.0
    chemical_score: float = 0.0  # Cached chemical feasibility score
    is_leaf: bool = True  # Is this a leaf node (unexpanded)
    depth: int = 0
    
    def __hash__(self) -> int:
        """Hash based on reaction_id"""
        return hash(self.reaction_id)
    
    def __eq__(self, other) -> bool:
        """Equality comparison based on reaction_id"""
        if not isinstance(other, ANDNode):
            return False
        return self.reaction_id == other.reaction_id
    
    def get_ucb_score(self, c_param: float = 1.414) -> float:
        """Calculate UCB1 score"""
        if self.visit_count == 0:
            return float('inf')
        
        if self.parent_molecule is None:
            return self.average_value
            
        # Calculate parent node's total visits (sum of all sibling AND nodes' visits)
        parent_total_visits = sum(sibling.visit_count for sibling in self.parent_molecule.child_reactions)
        
        exploitation = self.average_value
        exploration = c_param * math.sqrt(math.log(max(parent_total_visits, 1)) / max(self.visit_count, 1))
        return exploitation + exploration
