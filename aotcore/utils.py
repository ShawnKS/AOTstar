import copy
import ast
import os
import threading
from collections import OrderedDict
from itertools import permutations
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions, DataStructs
import numpy as np
from multiprocessing import Pool, cpu_count
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction

def change_to_forward_reaction(reaction:str):
    products, reactants = reaction.split(">>")
    return reactants + ">>" + products

def process_reaction_routes(route:list):
    json_list = []
    for idx,i in enumerate(route):
        products, reactants = i.split(">>")
        reaction = i
        reactants_list = reactants.split(".")
        #forward_reaction = change_to_forward_reaction(reaction)
        if idx == 0:
            step = {
                "Molecule set": str([products]),
                "Product": str([products]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(reactants_list)
            }
            json_list.append(step)
        else:
            original_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            update_set = copy.deepcopy(ast.literal_eval(json_list[idx-1]["Updated molecule set"]))
            try:
                update_set.remove(products)
            except:
                print(update_set)
                print(products)
            update_set = update_set + reactants_list
            step = {
                "Molecule set": str(original_set),
                "Product": str([products]),
                "Reaction": str([reaction]),
                "Reactants": str(reactants_list),
                "Updated molecule set": str(update_set)
            }
            json_list.append(step)
    
    return json_list

def starting_invalid_feedback(evaluation):

    fb = '''\nIn the first step, the molecule in the molecule set should be the target molecule'''.format(step[2]['target_smi'])

    return fb

def molecule_invalid_feedback(evaluation):

    invalid_molecule_id = evaluation['invalid_updated_mol_id']
    updated_molecule_set = evaluation['updated_molecule_set']

    fb = '''\nIn the 'Updated molecule set','''

    for i in range(len(invalid_molecule_id)):
        fb = fb + '''
        the molecule {} is not a valid molecule SMILES. Please make sure all the molecules are in the SMILES format.
        '''.format(updated_molecule_set[invalid_molecule_id[i]])

    return fb

def molecule_unavailable_feedback(evaluation, inventory):
    unavailable_mol_id = evaluation['unavailable_mol_id']
    updated_molecule_set = evaluation['updated_molecule_set']
    unperchasable_molecule = check_availability(updated_molecule_set, inventory)

    fb = '''\nIn the 'Updated molecule set','''

    fb = fb + '''
    the molecule {} cannot be purchased from the market.\n
    '''.format(str(unperchasable_molecule))


    return fb

def reaction_unavailable_feedback(evaluation):
    reaction = evaluation['reaction']
    #forward_reaction = change_to_forward_reaction(reaction)
    fb = '''\nThe reaction {} does not exist in the USPTO dataset. Please make sure all the molecules in the reaction are in SMILES format.\n'''.format(reaction)

    return fb
def product_not_inside_feedback(evaluation):
    product = evaluation['product'][0]

    fb = '''\nThe product molecule {} is not in the molecule set. Please make sure the product molecule is in the molecule set.\n'''.format(product)

    return fb
def check_availability(smi_list, inventory):
    unavailable_list = []
    for smi in smi_list:
        signal = inventory.is_purchasable(Molecule(smi))
        if not signal:
            unavailable_list.append(smi)
    
    return unavailable_list
    
def retrieve_routes(target_smi, all_fps, route_list, number):
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    fp = getfp(target_smi)
    sim_score = similarity_metric(fp, [fp_ for fp_ in all_fps])

    rag_tuples = list(zip(sim_score, route_list))
    rag_tuples = sorted(rag_tuples, key=lambda x: x[0], reverse=True)[:50]

    route_list = [t[1] for t in rag_tuples]
    sims_list = [t[0] for t in rag_tuples]


    sum_scores = sum(sims_list)
    population_probs = [p / sum_scores for p in sims_list]
    sampled_index = np.random.choice(len(route_list), p=population_probs, size=3, replace=False)
    sampled_routes = [route_list[i] for i in sampled_index]

    return sampled_routes
def updated_set_mismatch_feedback(evaluation):
    reaction = evaluation['reaction']
    product = evaluation['product'][0]

    fb = '''\nThe molecule set and the updated molecule set are not aligned. In each step, you need to keep a molecule set in which are the molecules we need. After taking the backward reaction in this step, you need to remove the products from the molecule set and add the reactants to the molecule set and then store
this set as 'Updated molecule set' in this step. In the last step, all the molecules in the 'Updated molecule set' should be purchasable. Please also check whether the product of this reaction is in the molecule set.'''

    return fb

def reaction_can_not_happen(evaluation):
    reaction = evaluation['reaction']
    #forward_reaction = change_to_forward_reaction(reaction)    
    fb = '''\nThe reaction {} cannot happen with the product molecule. \n'''.format(reaction)

    return fb


def _build_ordered_updated_set(molecule_set, product, reactants_generated):
    """Build a stable updated molecule list without changing set semantics."""
    ordered = []
    seen = set()

    products_canonical = {
        sanitize_smiles(smi) for smi in product if sanitize_smiles(smi) is not None
    }

    for smi in molecule_set:
        canonical = sanitize_smiles(smi)
        if canonical is None or canonical in products_canonical or canonical in seen:
            continue
        ordered.append(canonical)
        seen.add(canonical)

    for smi in reactants_generated:
        canonical = sanitize_smiles(smi)
        if canonical is None or canonical in seen:
            continue
        ordered.append(canonical)
        seen.add(canonical)

    return ordered

def verify_reaction_step(molecule_set, updated_molecule_set, reaction, product, reactants, inventory, oracle):
    """
    Core chemistry validation: run a real reaction template and replace LLM's answers.

    Given a step's product and a found reaction template, runs rdchiral to
    generate ground-truth reactants. If the reaction works, auto-fixes the
    molecule set bookkeeping using generated reactants (not LLM-proposed ones).

    This is the central "auto-fix" mechanism — the step is valid because
    the chemistry is real, regardless of what the LLM said.

    Returns:
        tuple: (reaction_valid, updated_set_valid, corrected_updated_set, generated_reactants, auto_fixed, corrections_map)
    """
    results = {
        'reaction_valid': False,
        'updated_set_valid': False,
        'corrected_updated_set': None,
        'generated_reactants': None,
        'auto_fixed': False,
        'corrections_map': {}
    }

    # Parse the reaction, reactants, and products
    try:
        reaction_smiles = reaction
        reactants_llm = [Chem.MolFromSmiles(smi) for smi in reactants]  # LLM's proposal
        products_expected = [Chem.MolFromSmiles(smi) for smi in product]
        updated_molecule_set_mols = [Chem.MolFromSmiles(smi) for smi in updated_molecule_set]
        original_molecule_set = [Chem.MolFromSmiles(smi) for smi in molecule_set]
    except Exception as e:
        print(f"Error parsing molecules: {e}")
        return (results['reaction_valid'], results['updated_set_valid'],
                results['corrected_updated_set'], results['generated_reactants'], results['auto_fixed'],
                results['corrections_map'])

    # Step 1: Check if the reaction template can generate reactants from product.
    # run_retro owns the bounded product/template result cache.
    try:
        target_rd = rdchiralReactants(product[0])
        reaction_outputs = run_retro(target_rd, reaction_smiles)
    except Exception as e:
        print(f"Error running/ranking reaction: {e}")
        return (results['reaction_valid'], results['updated_set_valid'],
                results['corrected_updated_set'], results['generated_reactants'], results['auto_fixed'],
                results['corrections_map'])

    reactants_generated = []
    try:
        if len(reaction_outputs) > 1:
            reaction_outputs = rank_reactants(reaction_outputs, inventory, oracle)

        if len(reaction_outputs) > 0:
            reactants_generated = [reactant for reactant in reaction_outputs[0]]
            reactants_generated = [sanitize_smiles(smi) for smi in reactants_generated]
        else:
            reactants_generated = []

        if None in reactants_generated or reactants_generated == []:
            results['reaction_valid'] = False
            return (results['reaction_valid'], results['updated_set_valid'],
                    results['corrected_updated_set'], results['generated_reactants'], results['auto_fixed'],
                    results['corrections_map'])
        else:
            results['reaction_valid'] = True
            results['generated_reactants'] = reactants_generated  # Save for later use
    except Exception as e:
        print(f"Error running/ranking reaction: {e}")
        return (results['reaction_valid'], results['updated_set_valid'],
                results['corrected_updated_set'], results['generated_reactants'], results['auto_fixed'],
                results['corrections_map'])

    # Step 2: Validate/auto-fix molecule set bookkeeping (only if reaction is valid)
    # KEY FIX: Use GENERATED reactants (correct) not LLM's proposed reactants (may be wrong)
    if results['reaction_valid']:
        try:
            # Parse and log invalid molecules
            updated_smiles = set()
            invalid_updated_mols = []
            for i, (smi, mol) in enumerate(zip(updated_molecule_set, updated_molecule_set_mols)):
                if mol is not None:
                    updated_smiles.add(Chem.MolToSmiles(mol))
                else:
                    invalid_updated_mols.append((i, smi))

            products_smiles = set()
            invalid_products = []
            for i, (smi, mol) in enumerate(zip(product, products_expected)):
                if mol is not None:
                    products_smiles.add(Chem.MolToSmiles(mol))
                else:
                    invalid_products.append((i, smi))

            original_smiles = set()
            invalid_originals = []
            for i, (smi, mol) in enumerate(zip(molecule_set, original_molecule_set)):
                if mol is not None:
                    original_smiles.add(Chem.MolToSmiles(mol))
                else:
                    invalid_originals.append((i, smi))

            # Log invalid molecules
            if invalid_updated_mols:
                print(f"  ⚠️  Invalid molecules in LLM's updated set:")
                for idx, smi in invalid_updated_mols:
                    print(f"     [{idx}] '{smi}' (failed RDKit parsing)")

            if invalid_products:
                print(f"  ⚠️  Invalid product molecules:")
                for idx, smi in invalid_products:
                    print(f"     [{idx}] '{smi}' (failed RDKit parsing)")

            if invalid_originals:
                print(f"  ⚠️  Invalid molecules in original set:")
                for idx, smi in invalid_originals:
                    print(f"     [{idx}] '{smi}' (failed RDKit parsing)")

            # Use GENERATED reactants for bookkeeping (not LLM's proposal!)
            reactants_smiles = set(reactants_generated)

            # The product anchor must belong to the current frontier.
            if not products_smiles.issubset(original_smiles):
                print('  ❌ Product is not in current molecule set; auto-fix not allowed')
                results['updated_set_valid'] = False
                results['auto_fixed'] = False
                return (results['reaction_valid'], results['updated_set_valid'],
                        results['corrected_updated_set'], results['generated_reactants'], results['auto_fixed'],
                        results['corrections_map'])

            # Compute correct updated set using generated reactants
            expected_updated_sets = (original_smiles | reactants_smiles) - products_smiles
            ordered_updated_set = _build_ordered_updated_set(
                molecule_set, product, reactants_generated
            )

            # Check if LLM's proposed set matches the correct set
            if expected_updated_sets == updated_smiles and products_smiles.issubset(original_smiles):
                # LLM got it exactly right
                results['updated_set_valid'] = True
                results['corrected_updated_set'] = ordered_updated_set
                results['auto_fixed'] = False
            else:
                # LLM made a bookkeeping error - auto-fix with correct reactants
                results['updated_set_valid'] = True
                results['corrected_updated_set'] = ordered_updated_set
                results['auto_fixed'] = True

                print(f"  ⚠️  Auto-fixing molecule set with generated reactants:")
                print(f"     LLM proposed set: {sorted(updated_smiles)}")
                print(f"     Corrected set: {sorted(expected_updated_sets)}")
                print(f"     (Using generated reactants: {reactants_generated})")

            # Additional safety checks
            if None in updated_molecule_set_mols:
                print('  ❌ None in updated molecule set')
                results['updated_set_valid'] = False
            elif None in original_molecule_set:
                print('  ❌ None in original molecule set')
                results['updated_set_valid'] = False
            elif None in products_expected:
                print('  ❌ None in products')
                results['updated_set_valid'] = False

            # Check for product/reactant overlap (invalid chemistry)
            common_elements = products_smiles & reactants_smiles
            if common_elements:
                print('  ❌ Product equals reactants')
                results['updated_set_valid'] = False

        except Exception as e:
            print(f"Error validating molecule set: {e}")
            results['updated_set_valid'] = False

    return (results['reaction_valid'], results['updated_set_valid'],
            results['corrected_updated_set'], results['generated_reactants'], results['auto_fixed'],
            results['corrections_map'])


def is_reaction_in_dict(reaction_smiles, preprocessed_dict, ordered_keys=None, tversky_scores=None):
    reaction_key = None
    try:
        products, reactants = reaction_smiles.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p) for p in products.split(".")]
    except Exception as e:
        print(f"Error parsing input reaction: {e}")
        return False, reaction_key

    if None in reactant_mols or None in product_mols:
        return False, reaction_key

    # Use presorted key order if provided, otherwise iterate dict directly
    if ordered_keys is not None:
        key_iter = enumerate(ordered_keys)
    else:
        key_iter = enumerate(preprocessed_dict.keys())

    for i, key in key_iter:
        # F1 Tversky cutoff: templates are sorted by Tversky descending,
        # once score < 1.0, remaining templates cannot match (subset property)
        if tversky_scores is not None and tversky_scores[i] < 1.0:
            break

        smarts_reactant_mols, smarts_product_mols = preprocessed_dict[key]
        try:
            if len(smarts_reactant_mols) != len(reactant_mols):
                continue
            if len(smarts_product_mols) != len(product_mols):
                continue
            # Short-circuit: only check products if reactants match
            if is_one_to_one_match(smarts_reactant_mols, reactant_mols):
                if is_one_to_one_match(smarts_product_mols, product_mols):
                    reaction_key = key
                    return True, reaction_key
        except Exception as e:
            print(f"Error processing SMARTS {smarts_reactant_mols}: {e}")
            continue
    return False, reaction_key

def is_reaction_match(args):
    reactant_mols, product_mols, smarts_reactant_mols, smarts_product_mols = args
    if len(smarts_reactant_mols) != len(reactant_mols):
        return False
    if not is_one_to_one_match(smarts_reactant_mols, reactant_mols):
        return False
    if len(smarts_product_mols) != len(product_mols):
        return False
    if not is_one_to_one_match(smarts_product_mols, product_mols):
        return False
    return True

def _hopcroft_karp(adj, n_left, n_right):
    """Hopcroft-Karp maximum bipartite matching. Returns matching size."""
    match_left = [None] * n_left
    match_right = [None] * n_right

    def bfs():
        queue = []
        dist = [None] * n_left
        for u in range(n_left):
            if match_left[u] is None:
                dist[u] = 0
                queue.append(u)
        found = False
        qi = 0
        while qi < len(queue):
            u = queue[qi]; qi += 1
            for v in adj.get(u, []):
                w = match_right[v]
                if w is None:
                    found = True
                elif dist[w] is None:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found, dist

    def dfs(u, dist):
        for v in adj.get(u, []):
            w = match_right[v]
            if w is None or (dist[w] is not None and dist[w] == dist[u] + 1 and dfs(w, dist)):
                match_left[u] = v
                match_right[v] = u
                return True
        dist[u] = None
        return False

    matching = 0
    while True:
        found, dist = bfs()
        if not found:
            break
        for u in range(n_left):
            if match_left[u] is None:
                if dfs(u, dist):
                    matching += 1
    return matching


def is_one_to_one_match(smarts_mols, target_mols):
    """Check if there exists a one-to-one assignment where each target matches its smarts.
    Uses bipartite matching (Hopcroft-Karp) instead of brute-force permutations."""
    n = len(smarts_mols)
    adj = {}
    for i, smarts in enumerate(smarts_mols):
        neighbors = []
        for j, target in enumerate(target_mols):
            if target.HasSubstructMatch(smarts):
                neighbors.append(j)
        if not neighbors:
            return False  # early exit: this smarts matches nothing
        adj[i] = neighbors
    return _hopcroft_karp(adj, n, len(target_mols)) == n

def check_validity(mol_list: list):
    validity_signal = [False] * len(mol_list)
    for idx, smi in enumerate(mol_list):
        signal = sanitize_smiles(smi)
        if signal != None:
            validity_signal[idx] = True
    
    return validity_signal

def check_purchasable(mol_list: list, validity_signals, inventory):
    availability_signals = [False] * len(mol_list)
    for idx, smi in enumerate(mol_list):
        if validity_signals[idx] == True: 
            signal = inventory.is_purchasable(Molecule(smi))
            availability_signals[idx] = signal
    
    return availability_signals


def sanitize_smiles(smi):
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, canonical=True)
        return smi_canon
    except:
        return None

def _positive_int_from_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(0, value)


_RETRO_REACTION_CACHE_LIMIT = _positive_int_from_env("AOT_RETRO_REACTION_CACHE_SIZE", 20000)
_RETRO_RESULT_CACHE_LIMIT = _positive_int_from_env("AOT_RETRO_RESULT_CACHE_SIZE", 100000)
_retro_cache_lock = threading.RLock()
_rdchiral_reaction_cache = OrderedDict()  # template_str -> compiled rdchiralReaction
_retro_result_cache = OrderedDict()       # (product_smiles, template_str) -> tuple(tuple(reactants))


def _lru_get(cache, key):
    with _retro_cache_lock:
        if key not in cache:
            return False, None
        value = cache[key]
        cache.move_to_end(key)
        return True, value


def _lru_set(cache, key, value, limit):
    if limit <= 0:
        return
    with _retro_cache_lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)


def _normalize_retro_template(template):
    reactants = template.split(">>")[0].split(".")
    if len(reactants) > 1:
        return "(" + template.replace(">>", ")>>")
    return template


def _retro_product_cache_key(product):
    return getattr(product, "reactant_smiles", None) or str(product)


def _copy_retro_outputs(outputs):
    return [list(output) for output in outputs]


def clear_retro_caches(clear_compiled=True, clear_results=True):
    """Clear retro caches explicitly.

    The search runner keeps these caches across targets because they depend only
    on product/template chemistry, not on tree-search state.
    """
    with _retro_cache_lock:
        if clear_compiled:
            _rdchiral_reaction_cache.clear()
        if clear_results:
            _retro_result_cache.clear()


def retro_cache_info():
    with _retro_cache_lock:
        return {
            "compiled": len(_rdchiral_reaction_cache),
            "compiled_limit": _RETRO_REACTION_CACHE_LIMIT,
            "results": len(_retro_result_cache),
            "results_limit": _RETRO_RESULT_CACHE_LIMIT,
        }


def run_retro(product, template):
    """
    Run a reaction given the product and the template.
    Args:
        product (str): product
        template (str): template
    Returns:
        str: reactant SMILES string
    """
    template = _normalize_retro_template(template)
    cache_key = (_retro_product_cache_key(product), template)
    hit, cached_outputs = _lru_get(_retro_result_cache, cache_key)
    if hit:
        return _copy_retro_outputs(cached_outputs)

    # Use cached compiled reaction if available
    hit, compiled = _lru_get(_rdchiral_reaction_cache, template)
    if not hit:
        compiled = rdchiralReaction(template)
        _lru_set(_rdchiral_reaction_cache, template, compiled, _RETRO_REACTION_CACHE_LIMIT)

    try:
        outputs = rdchiralRun(compiled, product)
    except Exception as e:
        print(f"Error {e} running retro reaction {template} on product {product}")
        _lru_set(_retro_result_cache, cache_key, tuple(), _RETRO_RESULT_CACHE_LIMIT)
        return []
    result = tuple(tuple(output.split(".")) for output in outputs)
    _lru_set(_retro_result_cache, cache_key, result, _RETRO_RESULT_CACHE_LIMIT)
    return _copy_retro_outputs(result)

def smiles_to_reaction(smiles):
    try:
        reactants, products = smiles.split(">>")
        reactant_list = reactants.split(".")
        product_list = products.split(".")
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_list]
        product_mols = [Chem.MolFromSmiles(p) for p in product_list]
        reaction_smarts = f"{'.'.join([Chem.MolToSmarts(mol) for mol in reactant_mols])}>>{'.'.join([Chem.MolToSmarts(mol) for mol in product_mols])}"
        return rdChemReactions.ReactionFromSmarts(reaction_smarts)
    except Exception as e:
        print(f"Failed to convert SMILES to reaction: {e}")
        return None

def rank_reactants(reactants_list, inventory, oracle):
    """
    Rank reactants based on the number of products generated
    """
    dead_molecules = dict()
    visited_molecules = dict()
    non_empty_reactant_list = [item for item in reactants_list if item != []]
    scores = [oracle.reward(inventory, reactant, visited_molecules, dead_molecules) for reactant in non_empty_reactant_list]
    sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), key=lambda pair: pair[0], reverse=True)]
    return sorted_list
