"""
LLM-Guided Tree Optimizer for Retrosynthesis Planning
"""

import re
import ast
import json
import math
import os
import time
from typing import List, Dict, Optional, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import DataStructs
from syntheseus import Molecule

from aotcore.tree_nodes import ORNode, ANDNode
from aotcore.prompts import construct_full_prompt
from aotcore.utils import sanitize_smiles, process_reaction_routes
from aotcore.optimizer import BaseOptimizer, clear_module_caches, extract_molecules_from_output


class LLMGuidedTreeOptimizer(BaseOptimizer):
    """LLM-guided optimizer based on AND-OR Tree"""
    
    def __init__(self, args=None, inventory=None, template_dict=None,
                 reaction_list=None, all_reaction_fps=None, datasub=None,
                 local_solved_cache=None):
        super().__init__(args, inventory, template_dict, reaction_list,
                        all_reaction_fps, datasub, local_solved_cache)
        self.model_name = "llm_tree_planner"
        
        # API model configuration
        self.api_model = getattr(args, 'api_model', "llm-model")
        self.api_temperature = getattr(args, 'api_temperature', 0.7)
        self.api_max_tokens = getattr(args, 'api_max_tokens', 4096)
        
        # AND-OR Tree state
        self.root_or_node: Optional[ORNode] = None
        self.all_molecules: Dict[str, ORNode] = {}  # smiles -> ORNode
        self.leaf_and_nodes: Set[ANDNode] = set()
        self.total_and_nodes = 0
        
        # Search parameters
        self.max_iterations = 100
        self.max_depth = getattr(args, 'max_depth', 10)
        self.expansion_routes = 1
        self.num_initial_routes = getattr(args, 'num_initial_routes', 3)

        # Selection and backprop strategy
        self.depth_decay = getattr(args, 'depth_decay', 1.0)  # deprecated: path_burden handles depth
        self.backprop_method = getattr(args, 'backprop_method', 'average')
        self.unsolved_penalty = getattr(args, 'unsolved_penalty', 0.1)
        self.c_param = getattr(args, 'c_param', 0.5)
        self.max_reexpansions = getattr(args, 'max_reexpansions', 3)

        print(f"Initialized LLM-Guided Tree Optimizer"
              f" (depth_decay={self.depth_decay},"
              f" backprop={self.backprop_method},"
              f" unsolved_penalty={self.unsolved_penalty},"
              f" max_reexpansions={self.max_reexpansions})")
    
    def clear_cache(self):
        """Clear search cache between molecules to prevent memory accumulation."""
        clear_module_caches()
        self.root_or_node = None
        self.all_molecules.clear()
        self.leaf_and_nodes.clear()
        self.total_and_nodes = 0
        self.visited_molecules = dict()
        self.dead_molecules = dict()
        self.explored_reaction = set()

        # Clear oracle caches that accumulate across molecules
        self.oracle.route_buffer.clear()
        self.oracle.reaction_cache.clear()

        # Clear rule-based search cache (rdchiral objects accumulate)
        self.jx_cache.clear()

        print("Cleared search cache")
    
    def get_or_create_or_node(self, molecule_smiles: str,
                              restore_visited: Optional[Set[str]] = None) -> ORNode:
        """Get or create OR node"""
        molecule_smiles = sanitize_smiles(molecule_smiles)
        if molecule_smiles is None:
            raise ValueError(f"Invalid SMILES: {molecule_smiles}")
        
        if molecule_smiles not in self.all_molecules:
            # Check if molecule is purchasable
            is_solved = self.inventory.is_purchasable(Molecule(molecule_smiles))
            
            or_node = ORNode(
                molecule=molecule_smiles,
                smiles=molecule_smiles,
                is_solved=is_solved
            )
            self.all_molecules[molecule_smiles] = or_node

            if not is_solved:
                self._restore_or_node_from_local_cache(or_node, restore_visited)
        
        return self.all_molecules[molecule_smiles]

    def _restore_or_node_from_local_cache(self, or_node: ORNode,
                                           visited: Optional[Set[str]] = None) -> bool:
        """Materialize a solved subtree from the local cache if available."""
        if self.local_solved_cache is None or or_node.is_solved:
            return or_node.is_solved

        if visited is None:
            visited = set()
        if or_node.smiles in visited:
            return False
        if or_node.solving_and_node is not None:
            return or_node.is_solved

        cached_reactants = self.local_solved_cache.get(or_node.smiles)
        if not cached_reactants:
            return False

        visited.add(or_node.smiles)
        child_or_nodes = []

        try:
            for reactant_smiles in cached_reactants:
                child_or = self.get_or_create_or_node(
                    reactant_smiles, restore_visited=visited
                )
                child_or_nodes.append(child_or)

            if not all(child.is_solved for child in child_or_nodes):
                return False

            reaction_id = f"local_cache::{or_node.smiles}>>{'.'.join(cached_reactants)}"
            cached_and = ANDNode(
                reaction_id=reaction_id,
                reactants=list(cached_reactants),
                product=or_node.smiles,
                parent_molecule=or_node,
                child_molecules=[],
                visit_count=1,
                total_value=1.0,
                average_value=1.0,
                feasibility_score=1.0,
                chemical_score=1.0,
                is_leaf=False,
                depth=0,
            )

            or_node.child_reactions.append(cached_and)
            for child_or in child_or_nodes:
                child_or.parent_reactions.append(cached_and)
                cached_and.child_molecules.append(child_or)

            or_node.is_solved = True
            or_node.solving_and_node = cached_and
            self.total_and_nodes += 1

            mol_str = or_node.smiles[:40] + "..." if len(or_node.smiles) > 40 else or_node.smiles
            print(f"📦 Restored solved node from local cache: {mol_str}")
            return True
        finally:
            visited.discard(or_node.smiles)

    def _store_solved_subtree_to_local_cache(self, or_node: ORNode,
                                              visited: Optional[Set[str]] = None):
        """Persist a solved reaction subtree in post-order for safe reuse."""
        if self.local_solved_cache is None:
            return
        if not or_node.is_solved or or_node.solving_and_node is None:
            return

        if visited is None:
            visited = set()
        if or_node.smiles in visited:
            return
        visited.add(or_node.smiles)

        try:
            for child_or in or_node.solving_and_node.child_molecules:
                self._store_solved_subtree_to_local_cache(child_or, visited)

            if or_node.solving_and_node.depth < 3:
                return

            reactants = [sanitize_smiles(child.smiles) for child in or_node.solving_and_node.child_molecules]
            if all(reactants):
                self.local_solved_cache.put(or_node.smiles, reactants)
        finally:
            visited.discard(or_node.smiles)
    
    def _expand_root_single(self, route_list, all_fps):
        """Expand root node once with retry mechanism.

        Makes up to 8 retry attempts for a single LLM call to produce
        at least one valid AND node under the root OR node.

        Returns:
            Tuple of (list of AND nodes, mapping stats dict)
        """
        retry_count = 0
        while retry_count < 8:
            try:
                retry_count += 1
                if retry_count > 1:
                    print(f"🔄 Retrying root expansion (attempt {retry_count})...")

                and_nodes, mapping_stats = self._expand_or_node_with_llm(self.root_or_node, route_list, all_fps)
                if and_nodes:
                    print(f"✅ Root expansion succeeded on attempt {retry_count}")
                    return and_nodes, mapping_stats
            except Exception as e:
                print(f"Root expansion error (attempt {retry_count}): {e}")
                continue
        print("❌ Root expansion failed after all attempts")
        return [], {}

    def _expand_root(self, route_list, all_fps, num_calls=1):
        """Expand root node with diverse initial routes.

        Makes num_calls independent LLM calls to create diverse AND nodes
        under the root OR node. Each call may propose a different
        decomposition of the target molecule (due to LLM temperature),
        giving UCB multiple directions to explore from the start.

        Args:
            route_list: Reference routes for RAG
            all_fps: Fingerprints for similarity search
            num_calls: Number of independent LLM calls (default 1)

        Returns:
            Tuple of (list of all AND nodes, merged mapping stats dict)
        """
        all_and_nodes = []
        merged_stats = {
            "steps_in_llm_route": 0,
            "valid_steps": 0,
            "mapped_steps": 0,
            "dropped_steps": 0,
            "dropped_step_products": [],
        }

        for call_idx in range(num_calls):
            if num_calls > 1:
                print(f"🌿 Root diversification call {call_idx + 1}/{num_calls}")

            and_nodes, stats = self._expand_root_single(route_list, all_fps)
            all_and_nodes.extend(and_nodes)

            # Merge stats
            for key in ["steps_in_llm_route", "valid_steps", "mapped_steps", "dropped_steps"]:
                merged_stats[key] = merged_stats.get(key, 0) + stats.get(key, 0)
            merged_stats["dropped_step_products"].extend(stats.get("dropped_step_products", []))

        if num_calls > 1:
            print(f"🌿 Root diversification complete: {len(all_and_nodes)} AND nodes "
                  f"from {num_calls} calls")

        return all_and_nodes, merged_stats

    def _build_tree_snapshot(self) -> Dict:
        """Build a snapshot of current tree state for logging."""
        solved_or = sum(1 for n in self.all_molecules.values() if n.is_solved)
        return {
            "total_and_nodes": self.total_and_nodes,
            "total_or_nodes": len(self.all_molecules),
            "leaf_and_nodes_count": len(self.leaf_and_nodes),
            "solved_or_nodes": solved_or,
            "dead_molecules_count": len(self.dead_molecules),
        }

    def _optimize(self, target, route_list, all_fps, config):
        """Main optimization entry point with optional restart strategy.

        When max_restarts > 1, splits the total iteration budget evenly
        across independent runs. Each run starts from a fresh tree.
        If any run solves the target, returns immediately.

        This is effective because the search behaves as a probabilistic
        sampler — two independent 50-iteration runs have higher expected
        success than one 100-iteration run for most molecules.
        """
        max_restarts = config.get("max_restarts", 1)

        if max_restarts <= 1:
            # No restart: use full budget in a single run
            return self._optimize_single_run(target, route_list, all_fps, config)

        # Restart strategy: split budget evenly across runs
        total_iterations = config.get("max_iterations", 100)
        iters_per_run = total_iterations // max_restarts

        # Accumulate search logs and stats across all runs
        all_search_logs = []
        best_partial = None
        best_partial_and_nodes = 0
        cumulative_and_nodes = 0
        global_iter_offset = 0

        for run_idx in range(max_restarts):
            is_last_run = (run_idx == max_restarts - 1)
            # Give remaining budget to the last run (handles rounding)
            run_iters = total_iterations - global_iter_offset if is_last_run else iters_per_run

            run_config = dict(config)
            run_config["max_iterations"] = run_iters
            run_config["max_restarts"] = 1  # prevent recursion

            if run_idx > 0:
                print(f"\n{'='*60}")
                print(f"RESTART {run_idx + 1}/{max_restarts} "
                      f"(iterations {global_iter_offset + 1}-{global_iter_offset + run_iters} "
                      f"of {total_iterations})")
                print(f"{'='*60}")

            result = self._optimize_single_run(target, route_list, all_fps, run_config)

            # Adjust iteration numbers in search log to be globally consistent
            for entry in self.search_log:
                if "iteration" in entry and entry["iteration"] >= 0:
                    entry["iteration"] += global_iter_offset
                entry["restart_run"] = run_idx

            all_search_logs.extend(self.search_log)
            cumulative_and_nodes += self.total_and_nodes

            # Use actual iterations completed in this run (not the budget)
            # Count iterations from search_log: entries with iteration >= 0
            actual_iters_this_run = sum(1 for e in self.search_log
                                        if e.get("iteration", -1) >= 0)
            global_iter_offset += actual_iters_this_run

            # Track iterations completed across all runs
            self.current_iteration = global_iter_offset

            if result is not None and result.get("type") != "partial_solution":
                # Found a complete solution
                print(f"Solved on restart run {run_idx + 1}/{max_restarts}")
                self.search_log = all_search_logs
                self.total_and_nodes = cumulative_and_nodes
                return result

            # Keep the best partial solution (fewest unsolved leaves)
            if result is not None:
                if best_partial is None:
                    best_partial = result
                    best_partial_and_nodes = self.total_and_nodes
                else:
                    cur_unsolved = result.get("statistics", {}).get("unsolved_leaf_count", float('inf'))
                    best_unsolved = best_partial.get("statistics", {}).get("unsolved_leaf_count", float('inf'))
                    if cur_unsolved < best_unsolved:
                        best_partial = result
                        best_partial_and_nodes = self.total_and_nodes

        # All runs exhausted without solving
        self.search_log = all_search_logs
        self.total_and_nodes = cumulative_and_nodes
        return best_partial

    def _optimize_single_run(self, target, route_list, all_fps, config):
        """Single run of the optimization loop (one tree, one budget)."""
        self.clear_cache()
        if self.local_solved_cache is not None:
            self.local_solved_cache.refresh_if_changed()
        self.max_iterations = config.get("max_iterations", 100)
        self.max_depth = config.get("max_depth", 10)
        self.current_iteration = 0
        self.search_log = []

        print(f"\n🔍 Starting LLM-Guided Tree Search:")
        print(f"   Target: {target}")
        print(f"   Budget: {self.max_iterations} iterations")

        # Initialize root node
        self.root_or_node = self.get_or_create_or_node(target)

        if self.root_or_node.is_solved:
            if self.root_or_node.solving_and_node is not None:
                print("✅ Target molecule restored from local solved cache!")
                return self._extract_solution()
            print("✅ Target molecule already purchasable!")
            return self._extract_trivial_solution()

        # Initial expansion with diverse routes
        num_init = self.num_initial_routes
        print(f"🔄 Starting initial expansion ({num_init} diverse calls)...")
        t_init_start = time.time()
        initial_and_nodes, initial_mapping_stats = self._expand_root(
            route_list, all_fps, num_calls=num_init
        )
        t_init_end = time.time()
        print(f"Initially, generated {len(initial_and_nodes)} new AND nodes "
              f"from {num_init} LLM calls "
              f"({t_init_end - t_init_start:.1f}s)")

        # Initial evaluation
        t_init_eval_start = time.time()
        for and_node in initial_and_nodes:
            score = self._evaluate_and_node_chemical(and_node)
            self._initialize_and_node_stats(and_node, score)
        t_init_eval_end = time.time()

        # Log initial expansion timing
        self.search_log.append({
            "iteration": -1,
            "action": "initial_expansion",
            "use_presort": getattr(self, 'use_presort', True),
            "use_tversky_cutoff": getattr(self, 'use_presort', True),
            "blurry_tversky": getattr(self, 'blurry_tversky', True),
            "new_and_nodes_created": len(initial_and_nodes),
            "timing": {
                "expansion_s": round(t_init_end - t_init_start, 2),
                "evaluation_s": round(t_init_eval_end - t_init_eval_start, 2),
                "llm_call_s": initial_mapping_stats.get("llm_call_s", 0),
                "validation_s": initial_mapping_stats.get("validation_s", 0),
            },
        })

        # Main search loop
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            iter_start_time = time.time()
            if iteration % 1 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")

            log_entry = {"iteration": iteration}

            # Update solved status across the current search tree.
            self._update_solved_status()

            if self.root_or_node.is_solved:
                print(f"✅ Found complete solution at iteration {iteration + 1}")
                log_entry["action"] = "solved"
                log_entry.update(self._build_tree_snapshot())
                self.search_log.append(log_entry)
                break

            # Selection: Choose AND node for expansion
            t_sel_start = time.time()
            selected_and_node = self._selection_phase()
            t_sel_end = time.time()

            if selected_and_node is None:
                print("No expandable AND node found")

                # Re-expand root if iterations remain
                if iteration < self.max_iterations - 1:
                    print(f"🔄 Re-expanding root node at iteration {iteration + 1}")
                    new_root_and_nodes, root_mapping_stats = self._expand_root(route_list, all_fps)
                    print(f"   Generated {len(new_root_and_nodes)} new AND nodes from root re-expansion")

                    for and_node in new_root_and_nodes:
                        score = self._evaluate_and_node_chemical(and_node)
                        self._initialize_and_node_stats(and_node, score)

                    log_entry["action"] = "root_reexpansion"
                    log_entry["new_and_nodes_created"] = len(new_root_and_nodes)
                    log_entry.update(root_mapping_stats)
                    log_entry.update(self._build_tree_snapshot())
                    log_entry["timing"] = {
                        "selection_s": round(t_sel_end - t_sel_start, 2),
                        "iteration_total_s": round(time.time() - iter_start_time, 2),
                    }
                    self.search_log.append(log_entry)
                    continue
                else:
                    print("Search terminated: reached maximum iterations")
                    log_entry["action"] = "no_expandable"
                    log_entry.update(self._build_tree_snapshot())
                    self.search_log.append(log_entry)
                    break

            log_entry["action"] = "selection"
            log_entry["selected_and_node_depth"] = selected_and_node.depth
            log_entry["selected_and_node_ucb"] = round(self._get_ucb_score(selected_and_node), 4)
            log_entry["selected_and_node_visits"] = selected_and_node.visit_count

            # Expansion: Generate new routes for selected AND node
            t_exp_start = time.time()
            new_and_nodes, expansion_mapping_stats = self._expansion_phase(selected_and_node, route_list, all_fps)
            t_exp_end = time.time()

            # Evaluation: Evaluate new nodes
            t_eval_start = time.time()
            for new_and_node in new_and_nodes:
                score = self._evaluate_and_node_chemical(new_and_node)
                self._initialize_and_node_stats(new_and_node, score)
                if self.backprop_method == 'andor':
                    self._backpropagate_andor(new_and_node)
                else:
                    self._backpropagate_and_node(new_and_node, score)
            t_eval_end = time.time()

            log_entry["new_and_nodes_created"] = len(new_and_nodes)
            log_entry.update(expansion_mapping_stats)
            log_entry.update(self._build_tree_snapshot())
            log_entry["timing"] = {
                "selection_s": round(t_sel_end - t_sel_start, 2),
                "expansion_s": round(t_exp_end - t_exp_start, 2),
                "evaluation_s": round(t_eval_end - t_eval_start, 2),
                "iteration_total_s": round(time.time() - iter_start_time, 2),
            }
            self.search_log.append(log_entry)

            # Check termination condition
            if len(self.oracle) > config.get("max_oracle_calls", 1000):
                print("Reached maximum oracle calls")
                break

        if self.root_or_node.is_solved:
            return self._extract_solution()
        else:
            return self._extract_partial_solution()
    
    def _selection_phase(self) -> Optional[ANDNode]:
        """Select the most promising AND node to expand next using flat UCB."""
        return self._selection_phase_flat_ucb()

    def _path_unsolved_burden(self, leaf_and_node: ANDNode) -> int:
        """Total unsolved OR nodes on the critical path from this AND to root.

        Walks upward from leaf_and_node to root_or_node, accumulating:
        - own unsolved: this AND's unsolved child OR count
        - at each ancestor AND: unsolved sibling OR count (excluding the path OR)
        - at each OR with multiple parents: pick the path with min sibling burden
          (OR semantics: only one decomposition needs to work)
        """
        burden = sum(1 for c in leaf_and_node.child_molecules if not c.is_solved)

        current_or = leaf_and_node.parent_molecule
        while current_or is not None and current_or != self.root_or_node:
            if not current_or.parent_reactions:
                break

            best_additional = float('inf')
            best_parent = None
            for parent_and in current_or.parent_reactions:
                sibling_unsolved = sum(
                    1 for c in parent_and.child_molecules
                    if c != current_or and not c.is_solved
                )
                if sibling_unsolved < best_additional:
                    best_additional = sibling_unsolved
                    best_parent = parent_and

            if best_parent is None:
                break

            burden += best_additional
            current_or = best_parent.parent_molecule

        return burden

    def _selection_phase_flat_ucb(self) -> Optional[ANDNode]:
        """Select the most promising AND node to expand next (flat UCB).

        Filters leaf AND nodes by: depth < max, not fully solved,
        has at least one unsolved reactant. Picks the one with highest
        penalized UCB score (UCB / (1 + path_burden * gamma)).
        """
        expandable_nodes = [
            node for node in self.leaf_and_nodes
            if (node.depth < self.max_depth and
                not self._is_and_node_fully_solved(node) and
                self._has_expandable_reactants(node))
        ]

        if not expandable_nodes:
            return None

        def _penalized_ucb(and_node):
            ucb = self._get_ucb_score(and_node)
            burden = self._path_unsolved_burden(and_node)
            return ucb / (1.0 + burden * self.unsolved_penalty)

        best_node = max(expandable_nodes, key=_penalized_ucb)

        best_burden = self._path_unsolved_burden(best_node)
        print(f"🎯 Selected AND node at depth {best_node.depth} "
              f"(UCB: {self._get_ucb_score(best_node):.3f}, "
              f"burden: {best_burden}, "
              f"penalized: {_penalized_ucb(best_node):.3f}, "
              f"visits: {best_node.visit_count})")

        return best_node

    def _expansion_phase(self, selected_and_node: ANDNode, route_list: List, all_fps: List) -> Tuple[List[ANDNode], Dict]:
        """Expand a selected AND node by generating routes for its unsolved reactants.

        Picks the most-needing-expansion reactant (prefer never-expanded,
        then least-visited), calls LLM to propose routes, validates them
        through sanitize() pipeline, maps valid steps to new AND nodes.

        5 retries. If all fail, marks the target molecule as dead
        (penalized in future scoring).

        Returns:
            Tuple of (list of created AND nodes, mapping stats dict)
        """
        print(f"🔄 Expanding AND node at depth {selected_and_node.depth}")

        # Find unsolved reactants
        unsolved_reactants = [
            or_node for or_node in selected_and_node.child_molecules
            if not or_node.is_solved
        ]

        if not unsolved_reactants:
            print("   All reactants solved, no expansion needed")
            selected_and_node.is_leaf = False
            self.leaf_and_nodes.discard(selected_and_node)
            return [], {}

        # Select expansion target
        target_reactant = self._select_expansion_target(unsolved_reactants)

        if target_reactant is None:
            print("   All unsolved reactants exhausted (max_reexpansions or dead)")
            selected_and_node.is_leaf = False
            self.leaf_and_nodes.discard(selected_and_node)
            return [], {}

        print(f"   Expanding reactant: {target_reactant.smiles}")

        # Track visited molecules on actual expansion attempts (not during evaluation)
        self.update_visited_molecules([target_reactant.smiles])

        # Retry mechanism
        new_and_nodes = []
        mapping_stats = {}
        max_retries = 5
        total_llm_time = 0.0
        total_validation_time = 0.0
        expansion_retries = 0

        for retry in range(max_retries):
            try:
                if retry > 0:
                    print(f"   🔄 Retrying expansion ({retry + 1}/{max_retries})...")
                    expansion_retries += 1

                # Generate routes with LLM
                t_llm_start = time.time()
                generated_routes = self._generate_routes_with_llm(target_reactant, route_list, all_fps)
                t_llm_end = time.time()
                total_llm_time += t_llm_end - t_llm_start

                if generated_routes:
                    # Map routes to tree structure (includes validation)
                    t_val_start = time.time()
                    new_and_nodes, mapping_stats = self._map_routes_to_tree(
                        target_reactant, generated_routes, selected_and_node.depth + 1
                    )
                    t_val_end = time.time()
                    total_validation_time += t_val_end - t_val_start

                    if new_and_nodes:
                        print(f"   ✅ Expansion succeeded on attempt {retry + 1}")
                        break
            except Exception as e:
                print(f"   Expansion error (attempt {retry + 1}): {e}")
                continue

        # Add timing to mapping_stats for logging
        mapping_stats["llm_call_s"] = round(total_llm_time, 2)
        mapping_stats["validation_s"] = round(total_validation_time, 2)
        mapping_stats["expansion_retries"] = expansion_retries

        if not new_and_nodes:
            print(f"   ❌ Expansion failed after {max_retries} attempts, marking node as exhausted")
            self.update_dead_molecules(target_reactant.smiles)
            selected_and_node.is_leaf = False
            self.leaf_and_nodes.discard(selected_and_node)

        mapping_stats["expansion_target_smiles"] = target_reactant.smiles
        print(f"   Generated {len(new_and_nodes)} new AND nodes")
        return new_and_nodes, mapping_stats
    
    def _generate_routes_with_llm(self, target_or_node: ORNode, route_list: List, all_fps: List) -> List[List[Dict]]:
        """Generate routes for target OR node using LLM"""
        # Get similar routes for reference (RAG)
        similar_routes = self._get_similar_routes(
            target_or_node.smiles, route_list, all_fps, num_examples=3
        )
        
        # Build examples
        examples = ''
        for route in similar_routes:
            examples += '<ROUTE>\n' + str(process_reaction_routes(route)) + '\n</ROUTE>\n'
        
        # Construct prompt
        question = construct_full_prompt(
            target_or_node.smiles, examples, self.expansion_routes
        )
        
        print(f'Query LLM agent using model: {self.api_model}...')
        message, answer = self.query_LLM(
            question, temperature=self.api_temperature, model=self.api_model
        )
        print('response...')
        print(answer)
        
        # Parse multiple routes
        return self._parse_multiple_routes(answer, target_or_node.smiles)
    
    def _parse_multiple_routes(self, llm_response: str, target_smiles: str) -> List[List[Dict]]:
        """Parse LLM response text into structured route step dicts.

        The LLM returns free-form text with <ROUTE>...</ROUTE> blocks.
        Parsing uses 3-tier fallback because LLMs produce inconsistent syntax:
          Tier 1: ast.literal_eval — works when LLM outputs valid Python dicts
          Tier 2: JSON parse (swap ' -> ") — works when LLM outputs JSON-like
          Tier 3: Regex fix for unquoted 'Rational' values — the most common
                  syntax error LLMs make (forgetting to quote string values)
        """
        routes = []

        # Handle None or empty responses
        if llm_response is None or not llm_response:
            print("⚠ LLM response is None or empty")
            return []

        try:
            # Extract ROUTE section
            match = re.search(r'<ROUTE>(.*?)<ROUTE>', llm_response, re.DOTALL)
            if match == None:
                match = re.search(r'<ROUTE>(.*?)</ROUTE>', llm_response, re.DOTALL)
            if not match:
                print("No <ROUTE> section found in LLM response")
                return []

            routes_content = match.group(1).strip()
            parsed_routes = None

            # Strategy 1: Try ast.literal_eval (original, most secure)
            try:
                parsed_routes = ast.literal_eval(routes_content)
                print("✓ ast.literal_eval parsing succeeded")
            except (SyntaxError, ValueError) as e:
                print(f"⚠ ast.literal_eval failed: {e}")

                # Strategy 2: Try JSON parsing as fallback
                try:
                    # Fix common issues: single quotes → double quotes
                    json_content = routes_content.replace("'", '"')
                    parsed_routes = json.loads(json_content)
                    print("✓ Fallback JSON parsing succeeded")
                except json.JSONDecodeError as je:
                    print(f"⚠ JSON fallback also failed: {je}")

                    # Strategy 3: Try more aggressive quote fixing
                    try:
                        # Try to fix unquoted values after colons
                        # This is a simple heuristic - may need refinement
                        fixed_content = routes_content
                        # Replace unquoted Rational values (common issue)
                        fixed_content = re.sub(
                            r"'Rational'\s*:\s*([^'\"{\[\]},][^,}\]]*)",
                            r"'Rational': '\1'",
                            fixed_content
                        )
                        parsed_routes = ast.literal_eval(fixed_content)
                        print("✓ Quote-fixing fallback succeeded")
                    except Exception as e3:
                        print(f"❌ All parsing strategies failed: {e3}")
                        return []

            if parsed_routes is None:
                return []

            # Handle single route vs multiple routes
            if self.expansion_routes == 1 and isinstance(parsed_routes, list) and len(parsed_routes) > 0:
                if isinstance(parsed_routes[0], dict):
                    # It's already a single route (list of steps)
                    parsed_routes = [parsed_routes]

            # Clean each route
            for i, route in enumerate(parsed_routes):
                try:
                    cleaned_route = self._clean_route(route, target_smiles)
                    if cleaned_route:
                        routes.append(cleaned_route)
                except Exception as e:
                    print(f"Failed to clean route {i}: {e}")
                    continue

        except Exception as e:
            print(f"Fatal parsing error: {e}")
            import traceback
            traceback.print_exc()

        print(f"Successfully parsed {len(routes)} routes from LLM response")
        return routes
    
    def _clean_route(self, route: List[Dict], target_smiles: str) -> Optional[List[Dict]]:
        """Clean single route with detailed error reporting"""
        try:
            # Check if last step is valid
            if len(route) >= 2:
                comp1 = ast.literal_eval(route[-1]['Updated molecule set'])
                comp2 = ast.literal_eval(route[-2]['Updated molecule set'])
                last_step_reactants = route[-1]['Reactants']

                if (set(comp1) == set(comp2) or
                    last_step_reactants in ["", "[]", "None", "[None]"]):
                    route = route[:-1]
                    print('Route cleaned - removed redundant last step!')

            # Validate format with detailed error reporting
            for i, step in enumerate(route):
                try:
                    ast.literal_eval(step['Molecule set'])
                except Exception as e:
                    print(f"❌ Step {i+1} 'Molecule set' parsing failed: {e}")
                    print(f"   Value: {str(step.get('Molecule set', 'MISSING'))[:100]}...")
                    raise

                try:
                    ast.literal_eval(step['Reaction'])
                except Exception as e:
                    print(f"❌ Step {i+1} 'Reaction' parsing failed: {e}")
                    print(f"   Value: {str(step.get('Reaction', 'MISSING'))[:100]}...")
                    raise

                try:
                    ast.literal_eval(step['Reactants'])
                except Exception as e:
                    print(f"❌ Step {i+1} 'Reactants' parsing failed: {e}")
                    print(f"   Value: {str(step.get('Reactants', 'MISSING'))[:100]}...")
                    raise

                try:
                    ast.literal_eval(step['Updated molecule set'])
                except Exception as e:
                    print(f"❌ Step {i+1} 'Updated molecule set' parsing failed: {e}")
                    print(f"   Value: {str(step.get('Updated molecule set', 'MISSING'))[:100]}...")
                    raise

            print(f"✓ Route validation passed - {len(route)} steps")
            return route

        except Exception as e:
            print(f"Route cleaning failed: {e}")
            if route:
                print(f"First step keys: {list(route[0].keys()) if route[0] else 'Empty'}")
                print(f"First step sample: {str(route[0])[:200]}...")
            return None
    
    def _map_routes_to_tree(self, target_or_node: ORNode, routes: List[List[Dict]],
                           base_depth: int) -> Tuple[List[ANDNode], Dict]:
        """Map validated route steps into the AND-OR tree structure.

        Takes a list of validated (step, step_info) tuples (invalid steps
        already filtered out) and creates AND nodes connected to OR nodes.
        Step 0 -> AND node under the target OR node.
        Steps 1+ -> flat loop over all OR nodes in this route's subtree
        (handles sibling branches correctly).

        Args:
            target_or_node: The target OR node to expand
            routes: List of route step lists
            base_depth: Base depth for new AND nodes

        Returns:
            Tuple of (list of all created AND nodes, mapping stats dict)
        """
        all_created_nodes = []
        total_steps_in_routes = 0
        total_valid_steps = 0
        total_mapped_steps = 0
        all_dropped_products = []

        # Accumulate per-iteration blurry_search trigger summary across all routes
        iter_blurry_calls = 0
        iter_blurry_total_s = 0.0
        iter_blurry_explored_size = 0
        iter_blurry_by_trigger = {}  # trigger_num -> {calls, elapsed_s}

        # Accumulate per-iteration sub-operation timing across all routes
        iter_lookup_colon_s, iter_lookup_colon_calls = 0.0, 0
        iter_lookup_nocolon_s, iter_lookup_nocolon_calls = 0.0, 0
        iter_product_check_s, iter_product_check_calls = 0.0, 0
        iter_verify_s, iter_verify_calls = 0.0, 0
        iter_reverify_s, iter_reverify_calls = 0.0, 0
        iter_salvage_s, iter_salvage_calls = 0.0, 0
        iter_step_log = []  # per-step: nocolon result + blurry trigger (concatenated across routes)

        for route_idx, route in enumerate(routes):
            try:
                total_steps_in_routes += len(route)

                # Validate route - returns list of (step, step_info) tuples for VALID steps only
                validated_steps = self._validate_route_with_templates(route, target_or_node.smiles)

                # Collect trigger summary written by check_route (even if route invalid)
                ts = getattr(self, '_last_trigger_summary', None)
                if ts:
                    iter_blurry_calls += ts.get("blurry_calls", 0)
                    iter_blurry_total_s += ts.get("blurry_total_s", 0.0)
                    iter_blurry_explored_size = max(iter_blurry_explored_size,
                                                    ts.get("blurry_explored_size", 0))
                    for k, v in ts.items():
                        if k.startswith("blurry_t") and "_" in k[8:]:
                            # key format: blurry_t{N}_{calls|elapsed_s}
                            trig_key = k[:9]  # e.g. "blurry_t1"
                            field = k[10:]    # "calls" or "elapsed_s"
                            iter_blurry_by_trigger.setdefault(trig_key, {"calls": 0, "elapsed_s": 0.0})
                            if field == "calls":
                                iter_blurry_by_trigger[trig_key]["calls"] += v
                            elif field == "elapsed_s":
                                iter_blurry_by_trigger[trig_key]["elapsed_s"] = round(
                                    iter_blurry_by_trigger[trig_key]["elapsed_s"] + v, 3)
                    # Sub-operation timing fields
                    iter_lookup_colon_s += ts.get("lookup_colon_s", 0.0)
                    iter_lookup_colon_calls += ts.get("lookup_colon_calls", 0)
                    iter_lookup_nocolon_s += ts.get("lookup_nocolon_s", 0.0)
                    iter_lookup_nocolon_calls += ts.get("lookup_nocolon_calls", 0)
                    iter_product_check_s += ts.get("product_check_s", 0.0)
                    iter_product_check_calls += ts.get("product_check_calls", 0)
                    iter_verify_s += ts.get("verify_s", 0.0)
                    iter_verify_calls += ts.get("verify_calls", 0)
                    iter_reverify_s += ts.get("reverify_s", 0.0)
                    iter_reverify_calls += ts.get("reverify_calls", 0)
                    iter_salvage_s += ts.get("salvage_s", 0.0)
                    iter_salvage_calls += ts.get("salvage_calls", 0)
                    iter_step_log.extend(ts.get("step_log", []))

                if not validated_steps:
                    continue

                total_valid_steps += len(validated_steps)

                # Create AND node for first valid step
                first_step, first_step_info = validated_steps[0]
                and_node = self._create_and_node_from_step(
                    first_step, target_or_node, route_idx, base_depth
                )

                if and_node:
                    all_created_nodes.append(and_node)
                    total_mapped_steps += 1

                    # Map remaining valid steps using flat sibling-aware mapping
                    remaining_nodes, dropped_products = self._map_validated_steps_to_tree(
                        and_node, validated_steps[1:]
                    )
                    all_created_nodes.extend(remaining_nodes)
                    total_mapped_steps += len(remaining_nodes)
                    all_dropped_products.extend(dropped_products)

            except Exception as e:
                print(f"Failed to map route {route_idx}: {e}")
                continue

        mapping_stats = {
            "steps_in_llm_route": total_steps_in_routes,
            "valid_steps": total_valid_steps,
            "mapped_steps": total_mapped_steps,
            "dropped_steps": len(all_dropped_products),
            "dropped_step_products": all_dropped_products,
            # Per-iteration blurry_search timing summary
            "blurry_calls": iter_blurry_calls,
            "blurry_total_s": round(iter_blurry_total_s, 3),
            "blurry_explored_size": iter_blurry_explored_size,
            **{f"{tk}_{fk}": fv
               for tk, fstats in iter_blurry_by_trigger.items()
               for fk, fv in fstats.items()},
            # Sub-operation timing breakdown
            "lookup_colon_s": round(iter_lookup_colon_s, 3),
            "lookup_colon_calls": iter_lookup_colon_calls,
            "lookup_nocolon_s": round(iter_lookup_nocolon_s, 3),
            "lookup_nocolon_calls": iter_lookup_nocolon_calls,
            "product_check_s": round(iter_product_check_s, 3),
            "product_check_calls": iter_product_check_calls,
            "verify_s": round(iter_verify_s, 3),
            "verify_calls": iter_verify_calls,
            "reverify_s": round(iter_reverify_s, 3),
            "reverify_calls": iter_reverify_calls,
            "salvage_s": round(iter_salvage_s, 3),
            "salvage_calls": iter_salvage_calls,
            "step_log": iter_step_log,
        }

        return all_created_nodes, mapping_stats
    
    def _validate_route_with_templates(self, route: List[Dict], target_smiles: str) -> Optional[List[Tuple[Dict, Dict]]]:
        """Validate route through the sanitize() pipeline and filter to valid steps only.

        Calls sanitize() which runs a single forward pass on the route
        (each step is validated with the previous step's corrected output).
        Returns only the steps that passed validation, paired with their
        evaluation info. Invalid steps are logged and skipped — the tree
        only gets AND nodes for chemistry that actually works.

        Args:
            route: List of route step dictionaries
            target_smiles: Target molecule SMILES

        Returns:
            List of (step_dict, step_info) tuples for VALID steps only,
            or None if no valid steps exist.
        """
        try:
            checked_route, final_evaluation = self.sanitize([target_smiles], route, exploration_signal=True)

            # Filter to only valid steps, pairing route step with evaluation
            valid_steps = []
            for i, (step_idx, is_valid, step_info) in enumerate(final_evaluation):
                if is_valid:
                    valid_steps.append((checked_route[i], step_info))

                    # Show if step was salvaged via skip-and-continue
                    if i > 0:
                        prev_step_id, prev_valid, prev_info = final_evaluation[i-1]
                        if not prev_valid:
                            print(f"✓ Step {step_idx} salvaged (previous step {prev_step_id} was skipped)")
                else:
                    # Enhanced logging: show ALL validation conditions
                    print(f"⏭️  Filtering out invalid step {step_idx}:")
                    print(f"   reaction_existance={step_info.get('reaction_existance', False)}")
                    print(f"   reaction_valid={step_info.get('reaction_valid', False)}")
                    print(f"   updated_set_valid={step_info.get('updated_set_valid', False)}")
                    print(f"   starting_signal={step_info.get('starting_signal', True)}")
                    print(f"   product_inside={step_info.get('product_inside', False)}")
                    invalid_mols = (len(step_info.get('invalid_molset_mol_id', [])) +
                                   len(step_info.get('invalid_updated_mol_id', [])))
                    print(f"   invalid_molecules={invalid_mols}")
                    if step_info.get('auto_fixed', False):
                        print(f"   ✓ (was auto-fixed with generated reactants)")


            if not valid_steps:
                print("No valid steps in route after filtering")
                return None

            print(f"Route validation: {len(valid_steps)}/{len(final_evaluation)} steps valid")
            return valid_steps

        except Exception as e:
            print(f"Route validation error: {e}")
            return None
    
    def _is_route_valid(self, evaluation: List) -> bool:
        """Check if ALL steps in route are valid.

        Args:
            evaluation: List of (step_idx, is_valid, step_info) tuples

        Returns:
            True only if ALL steps are valid
        """
        if not evaluation:
            return False

        for step_idx, is_valid, step_info in evaluation:
            if not is_valid:
                print(f"Step {step_idx} invalid: "
                      f"reaction_existance={step_info.get('reaction_existance', False)}, "
                      f"reaction_valid={step_info.get('reaction_valid', False)}")
                return False

        return True
    
    def _create_and_node_from_step(self, step: Dict, parent_or_node: ORNode, 
                                  route_idx: int, depth: int) -> Optional[ANDNode]:
        """Create AND node from route step"""
        try:
            # Extract reaction info
            reaction = ast.literal_eval(step['Reaction'])[0]
            reactants_smiles = ast.literal_eval(step['Reactants'])
            product_smiles = parent_or_node.smiles
            
            # Create AND node
            reaction_id = f"{product_smiles}_{route_idx}_{depth}_{hash(str(reactants_smiles)) % 10000}"
            and_node = ANDNode(
                reaction_id=reaction_id,
                reactants=reactants_smiles,
                product=product_smiles,
                parent_molecule=parent_or_node,
                depth=depth
            )
            
            # Connect to parent OR node
            parent_or_node.child_reactions.append(and_node)
            
            # Create reactant OR nodes
            available_reactants = []
            for reactant_smiles in reactants_smiles:
                reactant_or_node = self.get_or_create_or_node(reactant_smiles)
                reactant_or_node.parent_reactions.append(and_node)
                and_node.child_molecules.append(reactant_or_node)
                
                if reactant_or_node.is_solved:
                    available_reactants.append("✅")
                else:
                    available_reactants.append("❌")

            # Print reactant availability
            availability_str = " ".join(available_reactants)
            print(f"   Reactants availability: {availability_str} "
                  f"({sum(1 for x in available_reactants if x == '✅')}/{len(available_reactants)})")
            
            self.total_and_nodes += 1
            self.leaf_and_nodes.add(and_node)
            
            # Print reaction info
            reactants_str = " + ".join([r[:30] + "..." if len(r) > 30 else r for r in reactants_smiles])
            print(f"🧪 New reaction: {product_smiles[:30]}... → {reactants_str}")
            
            return and_node
            
        except Exception as e:
            print(f"Failed to create AND node: {e}")
            return None
    
    def _map_validated_steps_to_tree(self, first_and_node: ANDNode,
                                     remaining_steps: List[Tuple[Dict, Dict]]) -> Tuple[List[ANDNode], List[str]]:
        """Map remaining validated steps to OR nodes anywhere in this route's subtree.

        Supports sibling branches in a route: if step 0 creates OR_A and OR_B,
        step 1 decomposes A, and step 2 decomposes B, both steps are mapped.

        Uses the optimizer's OR-node registry to find nodes by SMILES, while
        constraining matches to nodes reachable from first_and_node so steps are
        not mapped to unrelated nodes from previous iterations.

        Args:
            first_and_node: The AND node created from step 0
            remaining_steps: List of (step_dict, step_info) tuples for VALID steps only

        Returns:
            Tuple of (created AND nodes, dropped step product SMILES)
        """
        created_nodes = []
        dropped_products = []

        if not remaining_steps:
            return created_nodes, dropped_products

        # Collect all OR node SMILES reachable from first_and_node
        # (these are the molecules created by this route's step 0)
        reachable_or_smiles = set()
        def collect_reachable(and_node):
            for child_or in and_node.child_molecules:
                reachable_or_smiles.add(child_or.smiles)
        collect_reachable(first_and_node)

        for current_step, current_step_info in remaining_steps:
            try:
                product = extract_molecules_from_output(current_step['Product'])[0]
                product_canonical = sanitize_smiles(product)

                if not product_canonical:
                    continue

                # Find a matching OR node constrained to this route's reachable set.
                target_or_node = None

                # Layer 1: Exact canonical match in the optimizer registry
                if product_canonical in self.all_molecules:
                    candidate = self.all_molecules[product_canonical]
                    if not candidate.is_solved and candidate.smiles in reachable_or_smiles:
                        target_or_node = candidate

                # Layer 2: Try non-canonical match within reachable set
                if target_or_node is None:
                    for smi in reachable_or_smiles:
                        if smi in self.all_molecules:
                            candidate = self.all_molecules[smi]
                            if not candidate.is_solved:
                                cand_canonical = sanitize_smiles(candidate.smiles)
                                if cand_canonical and product_canonical == cand_canonical:
                                    target_or_node = candidate
                                    print(f"  ℹ️  Matched OR node via canonical SMILES")
                                    print(f"     Product: '{product_canonical[:50]}...'")
                                    print(f"     OR node: '{candidate.smiles[:50]}...'")
                                    break

                if target_or_node:
                    # Determine depth from the target OR node's parent AND node
                    depth = first_and_node.depth + 1  # default
                    for parent_and in target_or_node.parent_reactions:
                        if parent_and == first_and_node or parent_and in created_nodes:
                            depth = parent_and.depth + 1
                            break

                    child_and_node = self._create_and_node_from_step(
                        current_step, target_or_node, 0, depth
                    )
                    if child_and_node:
                        created_nodes.append(child_and_node)
                        # Add new child OR nodes to reachable set
                        collect_reachable(child_and_node)
                else:
                    product_str = product_canonical[:40] + "..." if len(product_canonical) > 40 else product_canonical
                    print(f"No matching OR node for product {product_str} - step skipped")
                    dropped_products.append(product_canonical)

            except Exception as e:
                print(f"Failed to map step: {e}")

        return created_nodes, dropped_products
    
    def _evaluate_and_node_chemical(self, and_node: ANDNode) -> float:
        """Score an AND node based on chemistry feasibility only.

        Score = chemistry_score * depth_decay^depth

        Availability (how many children are solved) is NOT included here.
        It is handled exclusively by path_burden in _selection_phase_flat_ucb,
        avoiding double-counting and eliminating the need to rewrite MCTS history
        when children get solved.

        chemistry:    How hard are the unsolved reactants to synthesize?
                      Based on SCScore oracle (lower = easier).
        depth_decay:  Deprecated (default 1.0). Depth is penalized by path_burden.
        """
        try:
            unsolved_count = sum(1 for or_node in and_node.child_molecules
                                if not or_node.is_solved)
            total_reactants = len(and_node.child_molecules)

            # Get unsolved reactants
            unsolved_reactants = [
                or_node.smiles for or_node in and_node.child_molecules
                if not or_node.is_solved
            ]

            if not unsolved_reactants:
                # All reactants are solved
                and_node.feasibility_score = 1.0
                return 1.0

            # NOTE: visited_molecules is updated in _expansion_phase (on actual
            # expansion attempts), not here. Counting evaluation calls would
            # inflate the counter for popular intermediates that appear in many
            # AND nodes, triggering the >15 penalty prematurely.

            # Use Oracle reward to calculate chemical feasibility
            oracle_score = self.oracle.reward(
                self.inventory, unsolved_reactants,
                self.visited_molecules, self.dead_molecules
            )
            print(f'unscaled oracle score: {oracle_score}')

            # Convert oracle score to positive score
            chemistry_score = max(0.0, 1.0 + oracle_score/14)
            depth_penalty_factor = self.depth_decay ** and_node.depth

            # Pure chemistry score with depth penalty (no availability)
            final_score = chemistry_score * depth_penalty_factor
            and_node.feasibility_score = final_score

            solved_count = total_reactants - unsolved_count
            unsolved_penalty_divisor = 1.0 + unsolved_count * self.unsolved_penalty
            print(f"Evaluation - Chemistry: {chemistry_score:.3f}, "
                  f"Depth: {and_node.depth} (penalty: {depth_penalty_factor:.3f}), "
                  f"Unsolved: {unsolved_count}/{total_reactants} "
                  f"(selection penalty: /{unsolved_penalty_divisor:.2f}), "
                  f"Final: {final_score:.3f}")

            return final_score

        except Exception as e:
            print(f"Chemical evaluation failed: {e}")
            return 0.4
    
    def _initialize_and_node_stats(self, and_node: ANDNode, score: float):
        """Initialize AND node statistics"""
        and_node.visit_count = 1
        and_node.total_value = score
        and_node.average_value = score
    
    def _backpropagate_and_node(self, and_node: ANDNode, reward: float, visited_nodes: set = None):
        """Backpropagate AND node statistics"""
        if visited_nodes is None:
            visited_nodes = set()
        
        # Prevent circular visits
        if and_node.reaction_id in visited_nodes:
            return
            
        visited_nodes.add(and_node.reaction_id)
        
        current_or = and_node.parent_molecule
        if current_or is None or current_or == self.root_or_node:
            return

        # Update all parent AND nodes
        for parent_and in current_or.parent_reactions:
            parent_and.visit_count += 1
            parent_and.total_value += reward
            parent_and.average_value = parent_and.total_value / parent_and.visit_count
            # Recursive propagation
            self._backpropagate_and_node(parent_and, reward, visited_nodes)
    
    def _backpropagate_andor(self, start_and_node: ANDNode):
        """Recalculate ancestor values using AND-OR semantics.

        AND node value = min(child OR values)  -- bottleneck child
        OR node value  = max(child AND values) -- best alternative

        Called after creating a new AND node. Walks upward from
        start_and_node's parent OR, recalculating each ancestor.
        """
        visited = set()

        def _or_value(or_node):
            """Value of an OR node = best child AND value."""
            if or_node.is_solved:
                return 1.0
            if not or_node.child_reactions:
                return 0.5  # never expanded = neutral prior
            vals = [a.average_value for a in or_node.child_reactions
                    if self._has_expandable_reactants(a)
                       or self._is_and_node_fully_solved(a)]
            return max(vals) if vals else 0.0

        def _recalc_up(or_node):
            if id(or_node) in visited:
                return
            visited.add(id(or_node))
            if or_node == self.root_or_node:
                return

            for parent_and in or_node.parent_reactions:
                # AND value = min(child OR values) — bottleneck child
                child_vals = [_or_value(c) for c in parent_and.child_molecules]
                new_value = min(child_vals) if child_vals else 0.0

                parent_and.visit_count += 1
                parent_and.average_value = new_value
                parent_and.total_value = new_value * parent_and.visit_count

                if parent_and.parent_molecule:
                    _recalc_up(parent_and.parent_molecule)

        parent_or = start_and_node.parent_molecule
        if parent_or:
            _recalc_up(parent_or)

    def _update_solved_status(self):
        """Update solved status across the current search tree."""
        if self.local_solved_cache is not None:
            self.local_solved_cache.refresh_if_changed()

        changed = True
        
        while changed:
            changed = False
            newly_solved_this_round = set()
            
            # Local-cache restoration can add new OR nodes into self.all_molecules.
            # Iterate over a snapshot so solved propagation itself never mutates
            # the dict being iterated.
            for molecule_smiles, or_node in list(self.all_molecules.items()):
                if or_node.is_solved:
                    continue

                if self._restore_or_node_from_local_cache(or_node):
                    newly_solved_this_round.add(molecule_smiles)
                    changed = True

                    mol_str = or_node.smiles[:30] + "..." if len(or_node.smiles) > 30 else or_node.smiles
                    print(f"✅ Solved from local cache: {mol_str}")
                    continue
                
                for and_node in or_node.child_reactions:
                    if all(child_or.is_solved for child_or in and_node.child_molecules):
                        or_node.is_solved = True
                        or_node.solving_and_node = and_node
                        newly_solved_this_round.add(molecule_smiles)
                        changed = True
                        
                        mol_str = or_node.smiles[:30] + "..." if len(or_node.smiles) > 30 else or_node.smiles
                        print(f"✅ Solved: {mol_str}")
                        break
            
            if newly_solved_this_round:
                for molecule_smiles in newly_solved_this_round:
                    self._store_solved_subtree_to_local_cache(self.all_molecules[molecule_smiles])
                self._reevaluate_affected_and_nodes(newly_solved_this_round)
                self._cleanup_newly_solved_nodes(newly_solved_this_round)

    def _cleanup_newly_solved_nodes(self, newly_solved_molecules: set):
        """Clean up all child reactions of newly solved nodes"""
        cleaned_count = 0
        
        for molecule_smiles in newly_solved_molecules:
            or_node = self.all_molecules[molecule_smiles]
            
            # Clean up all child reactions of this OR node
            for and_node in list(or_node.child_reactions):
                # Remove from leaf_and_nodes if present
                if and_node in self.leaf_and_nodes:
                    self.leaf_and_nodes.discard(and_node)
                    and_node.is_leaf = False
                
                # Recursively clean up subtree
                self._cleanup_subtree(and_node)
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} AND nodes from {len(newly_solved_molecules)} newly solved molecules")

    def _reevaluate_affected_and_nodes(self, newly_solved_molecules):
        """Re-evaluate AND nodes affected by newly solved molecules.

        Since availability is no longer part of evaluation (handled by
        unsolved_penalty at selection time), re-evaluation only updates
        the chemistry component — which changes slightly because
        oracle.reward() now aggregates over fewer unsolved reactants.

        MCTS stats are updated incrementally (one new "visit" with the
        updated chemistry score) to preserve history continuity.
        """
        for molecule_smiles in newly_solved_molecules:
            or_node = self.all_molecules[molecule_smiles]
            for parent_and in or_node.parent_reactions:
                old_score = parent_and.feasibility_score
                new_score = self._evaluate_and_node_chemical(parent_and)
                parent_and.feasibility_score = new_score
                # Incremental MCTS update: treat re-eval as one new visit
                parent_and.visit_count += 1
                parent_and.total_value += new_score
                parent_and.average_value = parent_and.total_value / parent_and.visit_count
                print(f"🔄 Re-eval: {old_score:.3f} -> {new_score:.3f} "
                      f"(avg_value: {parent_and.average_value:.3f}, "
                      f"visits: {parent_and.visit_count})")
                # Propagate improvement to ancestors under andor mode
                if self.backprop_method == 'andor':
                    self._backpropagate_andor(parent_and)
    
    def _cleanup_subtree(self, and_node, visited=None):
        """Remove AND nodes from the search frontier when their parent OR is solved.

        When an OR node gets solved (all children of some AND are solved),
        the other AND nodes under it are no longer useful. This recursively
        removes them from leaf_and_nodes so UCB selection ignores them.

        Key design decisions:
          - Only "live" parents (still in leaf_and_nodes) block cleanup.
            Dead AND nodes that were already removed don't count.
          - Always recurses through non-leaf intermediates to reach
            deeper leaf nodes.
          - Visited-set prevents infinite loops from graph cycles.
        """
        if visited is None:
            visited = set()
        if id(and_node) in visited:
            return
        visited.add(id(and_node))

        for child_or in and_node.child_molecules:
            # Check if child OR node has any LIVE unsolved parent AND nodes.
            # A parent is "live" only if it's still in leaf_and_nodes
            # (i.e., still selectable for expansion). Dead AND nodes that
            # were removed from leaf_and_nodes should not block cleanup.
            live_unsolved_parents = [
                parent for parent in child_or.parent_reactions
                if parent in self.leaf_and_nodes and
                   not all(c.is_solved for c in parent.child_molecules)
            ]

            if not live_unsolved_parents:
                for child_and in list(child_or.child_reactions):
                    if child_and in self.leaf_and_nodes:
                        self.leaf_and_nodes.discard(child_and)
                        child_and.is_leaf = False
                    # Always recurse to reach deeper leaf nodes
                    # (even through non-leaf intermediates)
                    self._cleanup_subtree(child_and, visited)
    
    def _extract_solution(self) -> Optional[Dict]:
        """Extract solution using recorded solving paths"""
        if not self.root_or_node.is_solved:
            print("No complete solution found")
            return None
        
        def build_solution_tree(or_node: ORNode, visited: set = None, max_depth: int = 30) -> Dict:
            # Initialize visited set
            if visited is None:
                visited = set()
            
            # Prevent infinite recursion
            if or_node.smiles in visited:
                return {
                    "type": "circular_reference",
                    "molecule": or_node.smiles,
                    "note": "Circular reference detected"
                }
            
            # Prevent excessive depth
            if max_depth <= 0:
                return {
                    "type": "max_depth_reached",
                    "molecule": or_node.smiles,
                    "note": "Maximum recursion depth reached"
                }
            
            # Distinguish building blocks from reaction-solved molecules
            if or_node.is_solved and not or_node.child_reactions:
                # True building block (inventory molecule)
                return {
                    "type": "building_block",
                    "molecule": or_node.smiles
                }
            
            # Use recorded solving path
            if or_node.solving_and_node is None:
                return {
                    "type": "no_solving_path",
                    "molecule": or_node.smiles,
                    "note": "Node marked as solved but no solving path recorded"
                }
            
            # Add current node to visited set (use copy to avoid affecting siblings)
            visited_copy = visited.copy()
            visited_copy.add(or_node.smiles)
            
            # Use recorded solving path
            solving_and_node = or_node.solving_and_node
            
            return {
                "type": "reaction",
                "molecule": or_node.smiles,
                "reaction_id": solving_and_node.reaction_id,
                "reactants": [build_solution_tree(child, visited_copy, max_depth-1) 
                            for child in solving_and_node.child_molecules],
                "feasibility_score": solving_and_node.feasibility_score,
                "visit_count": solving_and_node.visit_count
            }
        
        solution = build_solution_tree(self.root_or_node)
        print("✅ Extracted complete solution tree using recorded solving paths")
        return solution
    
    def _extract_partial_solution(self) -> Dict:
        """Extract partial solution for failure case analysis"""
        
        def build_partial_tree(or_node: ORNode, visited: set = None, max_depth: int = 30) -> Dict:
            if visited is None:
                visited = set()
            
            if or_node.smiles in visited or max_depth <= 0:
                return {"type": "circular_or_deep", "molecule": or_node.smiles}
            
            # Building block
            if or_node.is_solved and not or_node.child_reactions:
                return {"type": "building_block", "molecule": or_node.smiles}
            
            # Unsolved leaf node
            if not or_node.child_reactions:
                return {
                    "type": "unsolved_leaf", 
                    "molecule": or_node.smiles,
                    "note": "No synthetic routes attempted"
                }
            
            # Select best reaction path
            if or_node.is_solved and or_node.solving_and_node:
                best_and_node = or_node.solving_and_node
                status = "solved"
            else:
                # Select most promising AND node
                best_and_node = max(or_node.child_reactions, 
                                  key=lambda x: (x.feasibility_score, x.visit_count, -x.depth))
                status = "partial"
            
            visited_copy = visited.copy()
            visited_copy.add(or_node.smiles)
            
            return {
                "type": "reaction",
                "status": status,
                "molecule": or_node.smiles,
                "reaction_id": best_and_node.reaction_id,
                "reactants": [build_partial_tree(child, visited_copy, max_depth-1) 
                            for child in best_and_node.child_molecules],
                "feasibility_score": best_and_node.feasibility_score,
                "visit_count": best_and_node.visit_count,
                "depth": best_and_node.depth
            }
        
        # Collect statistics
        unsolved_leaves = []
        depth_stats = {"max_depth": 0, "depth_distribution": {}}
        
        for or_node in self.all_molecules.values():
            if not or_node.is_solved and not or_node.child_reactions:
                unsolved_leaves.append(or_node.smiles)
            
            for and_node in or_node.child_reactions:
                depth = and_node.depth
                depth_stats["max_depth"] = max(depth_stats["max_depth"], depth)
                depth_stats["depth_distribution"][depth] = depth_stats["depth_distribution"].get(depth, 0) + 1
        
        partial_tree = build_partial_tree(self.root_or_node)
        
        return {
            "type": "partial_solution",
            "target_molecule": self.root_or_node.smiles,
            "solution_tree": partial_tree,
            "statistics": {
                "total_and_nodes": self.total_and_nodes,
                "total_or_nodes": len(self.all_molecules),
                "unsolved_leaf_count": len(unsolved_leaves),
                "unsolved_leaves": unsolved_leaves[:10],
                "depth_stats": depth_stats,
                "leaf_and_nodes_remaining": len(self.leaf_and_nodes)
            },
            "analysis_hints": self._generate_failure_analysis_hints(unsolved_leaves, depth_stats)
        }

    def _generate_failure_analysis_hints(self, unsolved_leaves: List[str], depth_stats: Dict) -> List[str]:
        """Generate failure analysis hints"""
        hints = []
        
        if len(unsolved_leaves) > 10:
            hints.append(f"Too many unsolved leaves ({len(unsolved_leaves)}), may need broader search")
        
        if depth_stats["max_depth"] >= self.max_depth - 1:
            hints.append(f"Reached max depth ({self.max_depth}), may need deeper search")
        
        if len(depth_stats["depth_distribution"]) <= 2:
            hints.append("Search too narrow, may need more exploration")
        
        if self.total_and_nodes < 10:
            hints.append("Very few nodes explored, LLM may be generating poor routes")
        
        return hints

    def _extract_trivial_solution(self) -> Dict:
        """Extract trivial solution (target already purchasable)"""
        return {
            "type": "trivial",
            "molecule": self.root_or_node.smiles,
            "message": "Target molecule is already purchasable"
        }
    
    # Helper methods
    def _get_similar_routes(self, target_smiles: str, route_list: List, all_fps: List, 
                           num_examples: int = 3) -> List:
        """Get similar historical routes (RAG)"""
        try:
            getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
            similarity_metric = DataStructs.BulkTanimotoSimilarity
            
            target_fp = getfp(target_smiles)
            sims = similarity_metric(target_fp, all_fps)
            
            # Get most similar routes
            rag_tuples = list(zip(sims, route_list))
            rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:num_examples]
            
            return [t[1] for t in rag_tuples]
            
        except Exception as e:
            print(f"Failed to get similar routes: {e}")
            return route_list[:num_examples]
    
    def _is_and_node_fully_solved(self, and_node: ANDNode) -> bool:
        """Check if AND node is fully solved"""
        return all(child_or.is_solved for child_or in and_node.child_molecules)
    
    def _has_expandable_reactants(self, and_node: ANDNode) -> bool:
        """Check if AND node has expandable (unsolved and not dead) reactants.

        AND semantics: ALL children must be solved. If any unsolved child
        is in dead_molecules, the AND node can never be satisfied — skip it
        to avoid wasting LLM calls on a doomed expansion.
        """
        for or_node in and_node.child_molecules:
            if not or_node.is_solved and or_node.smiles in self.dead_molecules:
                return False
        return any(not or_node.is_solved for or_node in and_node.child_molecules)
    
    def _get_ucb_score(self, and_node: ANDNode) -> float:
        """Calculate UCB1 score.

        Pure exploitation + exploration, no depth weighting.
        Depth penalization is handled by path_burden in selection.
        """
        if and_node.visit_count == 0:
            return float('inf')

        if and_node.parent_molecule is None:
            return and_node.average_value

        parent_total_visits = sum(sibling.visit_count
                                 for sibling in and_node.parent_molecule.child_reactions)

        exploitation = and_node.average_value
        exploration = self.c_param * math.sqrt(
            math.log(max(parent_total_visits, 1)) / max(and_node.visit_count, 1)
        )

        return exploitation + exploration
    
    def _select_expansion_target(self, unsolved_reactants: List[ORNode]) -> Optional[ORNode]:
        """Select reactant most needing expansion.

        Filters out OR nodes that have reached max_reexpansions or are dead.
        Returns None if all unsolved reactants are exhausted.
        """
        # Filter: not dead and under max_reexpansions limit
        expandable = [
            or_node for or_node in unsolved_reactants
            if (len(or_node.child_reactions) < self.max_reexpansions
                and or_node.smiles not in self.dead_molecules)
        ]

        if not expandable:
            return None  # All unsolved children exhausted

        # Prioritize molecules with no child reactions
        def priority_key(or_node):
            if not or_node.child_reactions:
                return (0, 0)  # Highest priority: no child reactions
            else:
                total_visits = sum(child.visit_count for child in or_node.child_reactions)
                return (1, total_visits)  # Secondary priority: by visit count

        return min(expandable, key=priority_key)
    
    def _expand_or_node_with_llm(self, or_node: ORNode, route_list: List, all_fps: List) -> Tuple[List[ANDNode], Dict]:
        """Generate initial AND nodes for root OR node"""
        t_llm = time.time()
        generated_routes = self._generate_routes_with_llm(or_node, route_list, all_fps)
        llm_time = time.time() - t_llm

        t_val = time.time()
        nodes, stats = self._map_routes_to_tree(or_node, generated_routes, 0)
        val_time = time.time() - t_val

        stats["llm_call_s"] = stats.get("llm_call_s", 0) + round(llm_time, 2)
        stats["validation_s"] = stats.get("validation_s", 0) + round(val_time, 2)
        return nodes, stats

    def _find_matching_step(self, or_node: ORNode, steps: List[Dict]) -> Optional[Dict]:
        """Find matching route step for OR node"""
        target_smiles = or_node.smiles
        
        for step in steps:
            try:
                product = extract_molecules_from_output(step['Product'])[0]
                if sanitize_smiles(product) == target_smiles:
                    return step
            except:
                continue
        
        return None
