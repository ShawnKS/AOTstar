"""
Base Optimizer and Oracle Classes for Tree Search
"""

import os
import time
import yaml
import random
# import torch
import numpy as np
from rdkit import Chem
from openai import OpenAI
import httpx
import tdc
import copy
import json
import heapq
from rdkit import Chem
from syntheseus import Molecule
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions, DataStructs
from itertools import permutations
from aotcore.utils import *
from scscore.scscore.standalone_model_numpy import *
import pickle
import openai
import json
from concurrent.futures import ThreadPoolExecutor
from rdchiral.main import rdchiralRun
from rdchiral.initialization import rdchiralReactants, rdchiralReaction
from aotcore.sim_based_rag import get_data_df, split_data_df, do_one
from tqdm import tqdm
from contextlib import contextmanager
import signal
import ast
import warnings
import threading
from collections import OrderedDict

# Suppress optional warnings from user-provided HTTPS endpoints with disabled
# certificate verification.
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_ROOT)
_LOCAL_SC_SCORE_WEIGHT_PATH = os.path.join(
    _REPO_ROOT,
    'scscore', 'models', 'full_reaxys_model_2048bool',
    'model.ckpt-10654.as_numpy.json.gz',
)
_SC_SCORE_WEIGHT_PATH = (
    os.environ.get("AOT_SC_SCORE_WEIGHT_PATH")
    or _LOCAL_SC_SCORE_WEIGHT_PATH
)
_shared_oracle_model_lock = threading.RLock()
_shared_sa_scorer = None
_shared_sc_oracle = None


def _share_oracle_models_enabled():
    return os.environ.get("AOT_SHARE_ORACLE_MODELS", "0") == "1"


def _get_shared_sa_scorer():
    global _shared_sa_scorer
    with _shared_oracle_model_lock:
        if _shared_sa_scorer is None:
            _shared_sa_scorer = tdc.Oracle(name='SA')
        return _shared_sa_scorer


def _get_shared_sc_oracle():
    global _shared_sc_oracle
    with _shared_oracle_model_lock:
        if _shared_sc_oracle is None:
            scorer = SCScorer()
            scorer.restore(_SC_SCORE_WEIGHT_PATH, FP_len=2048)
            print('restored...')
            _shared_sc_oracle = scorer
        return _shared_sc_oracle


def preload_oracle_models():
    """Preload read-only oracle scorer models before fork for COW sharing."""
    _get_shared_sa_scorer()
    _get_shared_sc_oracle()

# =============================================================================
# API Configuration for OpenAI-compatible LLM endpoints
# =============================================================================
def _split_api_keys(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(k).strip() for k in value if str(k).strip()]
    return [k.strip() for k in str(value).split(",") if k.strip()]


DEFAULT_API_CONFIG = {
    "timeout": 1200,
    "max_tokens": 4096,
}


def get_api_key_pool(model_name: str = None, args=None):
    """Get API keys from explicit args."""
    keys = []
    if args is not None:
        keys.extend(_split_api_keys(getattr(args, "llm_api_keys", None)))
        keys.extend(_split_api_keys(getattr(args, "llm_api_key", None)))
    return keys


def get_api_key(model_name: str = None, args=None):
    """Get API key from explicit args."""
    pool = get_api_key_pool(model_name, args)
    if pool:
        return pool[0]

    raise RuntimeError(
        "Pass --llm-api-key or --llm-api-keys before querying an LLM"
    )


def get_llm_config(model_name: str = None, args=None) -> dict:
    """Get API configuration for specified model."""
    if model_name is None:
        model_name = getattr(args, "api_model", None) if args is not None else None
        model_name = model_name or "llm-model"

    config = dict(DEFAULT_API_CONFIG)
    if args is not None:
        timeout_arg = getattr(args, "llm_timeout", None)
        max_tokens_arg = getattr(args, "api_max_tokens", None)
        if timeout_arg is not None:
            config["timeout"] = int(timeout_arg)
        if max_tokens_arg is not None:
            config["max_tokens"] = int(max_tokens_arg)

    base_url = (
        (getattr(args, "llm_base_url", None) if args is not None else None)
    )
    if not base_url:
        raise RuntimeError(
            "Pass --llm-base-url before querying an LLM"
        )
    config["base_url"] = base_url

    is_http = base_url.startswith("http://")
    verify_ssl_arg = getattr(args, "llm_verify_ssl", None) if args is not None else None
    if verify_ssl_arg is None:
        verify_ssl = True
    else:
        verify_ssl = bool(verify_ssl_arg)
    config["verify_ssl"] = verify_ssl and not is_http

    trust_env_arg = getattr(args, "llm_trust_env", None) if args is not None else None
    if trust_env_arg is None:
        trust_env = False
    else:
        trust_env = bool(trust_env_arg)
    config["trust_env"] = trust_env

    return config


def create_llm_client(model_name: str = None, args=None):
    """Create OpenAI client configured for LLM API

    LLM API settings are passed explicitly from the search entrypoint.
    """
    config = get_llm_config(model_name, args)

    http_client = httpx.Client(
        verify=config["verify_ssl"],
        timeout=config["timeout"],
        trust_env=config["trust_env"],
    )

    client = OpenAI(
        api_key=get_api_key(model_name, args),
        base_url=config["base_url"],
        http_client=http_client
    )

    return client, config


@contextmanager
def timeout(duration):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Search exceeded {duration} seconds")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def preprocess_reaction_dict(reaction_dict):
    """Preprocess reaction_dict to compile SMARTS into RDKit Mol objects."""
    preprocessed_dict = {}
    for key, smarts in reaction_dict.items():
        try:
            products, reactants = smarts.split(">>")
            reactant_mols = [Chem.MolFromSmarts(r) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmarts(p) for p in products.split(".")]
            preprocessed_dict[key] = (reactant_mols, product_mols)
        except Exception as e:
            print(f"Error preprocessing SMARTS {smarts}: {e}")
    return preprocessed_dict


def process_reaction_routes(route):
    """Process reaction routes for display/comparison"""
    # Implementation depends on your specific needs
    return route


class Oracle:
    """Oracle class for scoring routes"""
    
    def __init__(self, args=None, route_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.route_buffer = {} if route_buffer is None else route_buffer
        self.reaction_cache = dict()  # mol_smiles: [reaction]
        try:
            self.static_score_cache_size = int(os.environ.get("AOT_ORACLE_SCORE_CACHE_SIZE", 50000))
        except (TypeError, ValueError):
            self.static_score_cache_size = 50000
        self.static_score_cache_size = max(0, self.static_score_cache_size)
        self.static_score_cache = OrderedDict()  # mol_smiles -> (SC_score_or_None, sa_score_or_None)

        self.last_log = 0
        if _share_oracle_models_enabled():
            self.sa_scorer = _get_shared_sa_scorer()
            self.sc_Oracle = _get_shared_sc_oracle()
        else:
            self.sa_scorer = tdc.Oracle(name='SA')
            self.sc_Oracle = SCScorer()
            self.sc_Oracle.restore(_SC_SCORE_WEIGHT_PATH, FP_len=2048)
            print('restored...')

    def store_cache(self, mol_smiles, reaction):
        if mol_smiles in self.reaction_cache:
            if reaction not in self.reaction_cache[mol_smiles]:
                self.reaction_cache[mol_smiles].append(reaction)
        else:
            self.reaction_cache[mol_smiles] = [reaction]

    def _get_static_scores(self, mol_smiles, need_sc=True, need_sa=True):
        SC_score = None
        sa_score = None
        if self.static_score_cache_size > 0 and mol_smiles in self.static_score_cache:
            SC_score, sa_score = self.static_score_cache[mol_smiles]
            self.static_score_cache.move_to_end(mol_smiles)
            if (not need_sc or SC_score is not None) and (not need_sa or sa_score is not None):
                return SC_score, sa_score

        if need_sc and SC_score is None:
            _, SC_score = self.sc_Oracle.get_score_from_smi(mol_smiles)
            SC_score = float(np.asarray(SC_score).reshape(-1)[0])
        if need_sa and sa_score is None:
            sa_score = float(self.sa_scorer(mol_smiles))

        if self.static_score_cache_size > 0:
            self.static_score_cache[mol_smiles] = (SC_score, sa_score)
            self.static_score_cache.move_to_end(mol_smiles)
            while len(self.static_score_cache) > self.static_score_cache_size:
                self.static_score_cache.popitem(last=False)
        return SC_score, sa_score

    def get_static_scores(self, mol_smiles):
        return self._get_static_scores(mol_smiles, need_sc=True, need_sa=True)

    def get_oracle_score(self, mol_smiles):
        length = len(self.route_buffer)
        if os.environ.get("AOT_LAZY_STATIC_SCORES", "1") == "0":
            SC_score, sa_score = self.get_static_scores(mol_smiles)
        elif length <= 150:
            SC_score, _ = self._get_static_scores(mol_smiles, need_sc=True, need_sa=False)
            return SC_score
        elif length < 220:
            SC_score, sa_score = self._get_static_scores(mol_smiles, need_sc=True, need_sa=True)
        else:
            _, sa_score = self._get_static_scores(mol_smiles, need_sc=False, need_sa=True)
            return 0.5 * sa_score

        if length <= 150:
            overall_score = SC_score
        elif length > 150 and length < 220:
            alpha = (length/self.max_oracle_calls)
            overall_score = (1 - alpha) * SC_score + 0.5 * alpha * sa_score            
        else:
            overall_score = 0.5 * sa_score 
        return overall_score
    
    @property
    def budget(self):
        return self.max_oracle_calls
    
    def reward(self, inventory, updated_molecule_set: list, visited_molecules, dead_molecules):
        score_list = []
        for smi in updated_molecule_set:
            if smi in dead_molecules:
                if dead_molecules[smi] >= 1:
                    print('dead molecules!')
                    score_list.append(100)
                    continue
            try:
                signal = inventory.is_purchasable(Molecule(smi))
                if not signal:
                    score = self.get_oracle_score(smi)
                    if smi in visited_molecules:
                        print(f"Visited times: {visited_molecules[smi]}")
                        if visited_molecules[smi] > 15:
                            score = (visited_molecules[smi]/15) * score
                            print(f"Visited times adjust score")
                    score_list.append(score)
            except Exception as e:
                print(f"Error: {e}")
                score_list.append(5)
        if len(score_list) != 0:
            score_max = np.max(score_list)
        else:
            score_max = 0
        if len(score_list) != 0:
            score_mean = sum(score_list) / len(score_list) 
        else:
            score_mean = 0
        combined_score = score_mean + sum(score_list)
        final_score = -combined_score
        return final_score
    
    def evaluate(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        for idx, step in enumerate(route_evaluation):
            if step[1] == False:
                score = self.reward(inventory, step[2]['molecule_set'], visited_molecules, dead_molecules)
                return score
            elif step[1] == True:
                continue
        
        # Last step
        print(route_evaluation[-1])
        if route_evaluation[-1][2]['check_availability'] == True and len(route_evaluation[-1][2]['unavailable_mol_id']) == 0:
            score = 0
            return score
        else:
            score = self.reward(inventory, route_evaluation[-1][2]['updated_molecule_set'], visited_molecules, dead_molecules)
            return score

    def sort_buffer(self):
        self.route_buffer = dict(sorted(self.route_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        results_with_stats = {
            'oracle_calls': len(self.route_buffer),
            'routes': self.route_buffer
        }
        with open(output_file_path, 'w') as f:
            yaml.dump(results_with_stats, f, sort_keys=False)

    def log_intermediate(self, finish=False):
        if finish:
            n_calls = self.max_oracle_calls
            self.save_result(self.task_label)

    def __len__(self):
        return len(self.route_buffer) 

    def score_route(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        """Score one route"""
        if len(self.route_buffer) > self.max_oracle_calls:
            return -15
        if route_evaluation is None:
            return -15
        dict_key = json.dumps(route_evaluation)
        if dict_key in self.route_buffer:
            pass
        else:
            self.route_buffer[dict_key] = [
                float(self.evaluate(inventory, route_evaluation, visited_molecules, dead_molecules)), 
                len(self.route_buffer)+1
            ]
        return self.route_buffer[dict_key][0]
    
    def __call__(self, inventory, route_evaluation, visited_molecules, dead_molecules):
        """Score route and handle logging"""
        score_list = self.score_route(inventory, route_evaluation, visited_molecules, dead_molecules)
        if len(self.route_buffer) % self.freq_log == 0 and len(self.route_buffer) > self.last_log:
            self.sort_buffer()
            self.last_log = len(self.route_buffer)
            self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.route_buffer) >= self.max_oracle_calls


class BaseOptimizer:
    """Base optimizer class with common functionality"""
    
    def __init__(self, args=None, inventory=None, template_dict=None,
                 reaction_list=None, all_reaction_fps=None, datasub=None,
                 local_solved_cache=None):
        self.args = args
        if args and hasattr(args, 'api_model'):
            self.model_name = args.api_model

        self.oracle = Oracle(args=self.args)
        self.local_solved_cache = local_solved_cache

        # Load inventory
        if inventory is None:
            print('inventory not provided!!!')
            inventory_path = './dataset/inventory.pkl'
            self.inventory = self.load_inventory(inventory_path)
        else:
            self.inventory = inventory
            print('inventory loaded!')
        
        # Set paths
        if args:
            args.template_path = getattr(args, 'template_path', './dataset/idx2template_retro.json')
            args.rule_based_set_path = getattr(
                args,
                'rule_based_set_path',
                os.environ.get(
                    "AOT_RULE_BASED_SET_PATH",
                    os.path.join(_REPO_ROOT, "scscore", "data", "data_processed.csv"),
                ),
            )
        
        # Load templates
        self.original_template_dict = self.load_template(args.template_path if args else './dataset/idx2template_retro.json')
        self.template_dict = template_dict
        self.reaction_list, self.all_reaction_fps = reaction_list, all_reaction_fps
        self.datasub = datasub
        
        if template_dict is None:
            self.template_dict = preprocess_reaction_dict(self.original_template_dict)
        if reaction_list is None or all_reaction_fps is None:
            self.reaction_list, self.all_reaction_fps = self.get_reaction_fps(self.original_template_dict)
        if datasub is None and args:
            self.datasub = self.load_rule_based_set(args.rule_based_set_path)
        
        # Presort: pre-compute Pattern FPs for template products
        self.use_presort = getattr(args, 'use_presort', True) if args else True
        self.blurry_tversky = getattr(args, 'blurry_tversky', True) if args else True
        self.template_keys = list(self.template_dict.keys())
        self.template_pattern_fps = None
        if self.use_presort or self.blurry_tversky:
            import time as _t
            _t0 = _t.time()
            self.template_pattern_fps = self._build_template_pattern_fps()
            print(f"  PatternFPs: built {len(self.template_pattern_fps)} in {_t.time()-_t0:.1f}s")

        # Initialize search state
        self.explored_reaction = set()
        self.visited_molecules = dict()  # smiles: visit number
        self.dead_molecules = dict()
        self.jx_cache = {}
        self.template_to_key = {v: k for k, v in self.original_template_dict.items()}
        self.fast_template_lookup = os.environ.get("AOT_FAST_TEMPLATE_LOOKUP", "1") != "0"

    def _build_template_pattern_fps(self):
        """Pre-compute Pattern Fingerprints for all template product SMARTS."""
        fps = []
        for key in self.template_keys:
            smarts = self.original_template_dict[key]
            try:
                prod_str = smarts.split(">>")[0]
                combined = None
                for p in prod_str.split("."):
                    mol = Chem.MolFromSmarts(p)
                    if mol is None:
                        continue
                    fp = Chem.PatternFingerprint(mol)
                    combined = fp if combined is None else (combined | fp)
                fps.append(combined if combined else DataStructs.ExplicitBitVect(2048))
            except Exception:
                fps.append(DataStructs.ExplicitBitVect(2048))
        return fps

    def _build_template_reactant_pattern_fps(self):
        """Pre-compute PatternFPs for template reactant sides."""
        from aotcore.oracle_rerank import build_template_reactant_pattern_fps

        return build_template_reactant_pattern_fps(self)

    def presort_keys(self, reaction_smiles):
        """Sort template keys by Pattern FP + Tversky(0,1) similarity to product.
        Returns (sorted_keys, sorted_tversky_scores) for F1 cutoff."""
        if self.template_pattern_fps is None:
            return self.template_keys, None
        try:
            prod_smiles = reaction_smiles.split(">>")[0]
            combined = None
            for p in prod_smiles.split("."):
                mol = Chem.MolFromSmiles(p)
                if mol is None:
                    continue
                fp = Chem.PatternFingerprint(mol)
                combined = fp if combined is None else (combined | fp)
            if combined is None:
                return self.template_keys, None
            sims = DataStructs.BulkTverskySimilarity(combined, self.template_pattern_fps, 0, 1)
            if self.fast_template_lookup:
                try:
                    candidate_indices = [
                        i for i, score in enumerate(sims)
                        if score >= 1.0
                    ]
                    candidate_indices.sort(key=lambda i: sims[i], reverse=True)
                    sorted_keys = [self.template_keys[i] for i in candidate_indices]
                    sorted_scores = [sims[i] for i in candidate_indices]
                    return sorted_keys, sorted_scores
                except Exception:
                    pass
            sorted_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
            sorted_keys = [self.template_keys[i] for i in sorted_indices]
            sorted_scores = [sims[i] for i in sorted_indices]
            return sorted_keys, sorted_scores
        except Exception:
            return self.template_keys, None

    def load_template(self, template_path):
        with open(template_path, "r") as f:
            template_dict = json.load(f)
        return template_dict

    def load_inventory(self, inventory_path):
        with open(inventory_path, 'rb') as file:
            inventory = pickle.load(file)
        return inventory
    
    def load_rule_based_set(self, rule_based_set_path):
        data = get_data_df(rule_based_set_path)
        split_data_df(data)
        
        getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=False)
        
        all_fps = []
        for smi in tqdm(data['prod_smiles']):
            all_fps.append(getfp(smi))
        data['prod_fp'] = all_fps

        datasub = data.loc[data['dataset'] == 'train']
        fps = list(datasub['prod_fp'])
        print('Size of knowledge base: {}'.format(len(fps)))
        
        return datasub

    def update_visited_molecules(self, updated_molecule_set):
        for smi in updated_molecule_set:
            if smi in self.visited_molecules:
                self.visited_molecules[smi] += 1
            else:
                self.visited_molecules[smi] = 1

    def update_dead_molecules(self, dead_molecule):
        smi = dead_molecule
        assert type(smi) == str
        print('update dead molecules!')
        print(smi)
        if smi in self.dead_molecules:
            self.dead_molecules[smi] += 1
        else:
            self.dead_molecules[smi] = 1
                
    def get_reaction_fps(self, template_dict):
        reaction_list = list(template_dict.values())
        getreactionfp = lambda smart_reaction: rdChemReactions.CreateDifferenceFingerprintForReaction(
            rdChemReactions.ReactionFromSmarts(smart_reaction)
        )
        all_reaction_fps = []
        for reaction in reaction_list:
            all_reaction_fps.append(getreactionfp(reaction))
        
        return reaction_list, all_reaction_fps

    def rule_based_search(self, product_smiles, reaction_smiles):
        """
        Last-resort reaction finding: ignore LLM's reaction, search by product similarity.

        Finds molecules in training data similar to our product, retrieves their
        known reaction templates, and tries each. This abandons the LLM's chemistry
        suggestion entirely — pure database lookup.

        Used as fallback when blurry_search() exhausts all fingerprint-similar templates.

        Returns:
            (found, template_key, template_smarts)
        """
        template_list, self.jx_cache = do_one(product_smiles, self.datasub, self.jx_cache)

        if template_list == []:
            print('sim cannot found')
            return False, None, reaction_smiles
        else:
            templates = [t[0] for t in template_list]
            scores = [t[1] for t in template_list]
            weights = np.array(scores) 
            probabilities = weights / weights.sum()
            sampled_index = np.random.choice(len(templates), p=probabilities, size=len(templates), replace=False)
            sorted_templates = [templates[i] for i in sampled_index]
            for template in sorted_templates:
                raw_template = template[1:].replace(')>>', '>>')
                if (product_smiles, raw_template) in self.explored_reaction:
                    continue
                else:
                    key = '99999999999'
                    print('sim based reaction found')
                    return True, key, raw_template
            print('sim cannot found')
            return False, None, reaction_smiles

    def sanitize_smiles(self, smiles):
        """Check if a SMILES string is valid and return the sanitized molecule."""
        if smiles == '':
            return None
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            smi_canon = Chem.MolToSmiles(mol, canonical=True)
            return smi_canon
        except:
            return None

    def sanitize_reaction(self, reaction_smiles):
        """Process a reaction SMILES, removing invalid molecules."""
        try:
            reactants, products = reaction_smiles.split('>>')
        
            reactants_list = reactants.split('.')
            products_list = products.split('.')

            sanitized_reactants = [self.sanitize_smiles(smiles) for smiles in reactants_list]
            sanitized_products = [self.sanitize_smiles(smiles) for smiles in products_list]

            sanitized_reactants = [s for s in sanitized_reactants if s is not None]
            sanitized_products = [s for s in sanitized_products if s is not None]

            sanitized_reaction = ".".join(sanitized_reactants) + ">>" + ".".join(sanitized_products)

            return sanitized_reaction

        except Exception as e:
            print(f"Invalid reaction SMILES format: {reaction_smiles}. Error: {e}")
            return reaction_smiles

    def blurry_search(self, reaction_smiles, product_smiles, exploration_signal):
        """
        Find a similar reaction template when the LLM's proposed reaction doesn't exist.

        How it works:
          1. Convert LLM's reaction to a fingerprint
          2. Compare against all ~100K template fingerprints (Tanimoto similarity)
          3. Take top-N candidates (1000 in exploration mode, 100 otherwise)
          4. Try running each on the product via rdchiral
          5. First that produces valid reactants wins

        Why it works: the LLM often gets the reaction TYPE right (e.g., "break
        an ether bond") but the specific SMARTS pattern wrong. Similar templates
        often apply to the same product.

        Fallback: if all candidates fail, calls rule_based_search() which
        ignores the LLM's reaction entirely and searches by product similarity.

        Returns:
            (found, template_key, template_smarts)
        """
        if getattr(self.args, "blurry_oracle_rerank", False):
            from aotcore.oracle_rerank import blurry_search_oracle_rerank

            return blurry_search_oracle_rerank(
                self,
                reaction_smiles,
                product_smiles,
                exploration_signal,
            )

        if exploration_signal == True:
            reaction_number = 1000
        else:
            reaction_number = 100

        try:
            if self.blurry_tversky and self.template_pattern_fps is not None:
                # PatternFP(product) + Tversky(0,1): rank by product structural match
                prod_mol = Chem.MolFromSmiles(product_smiles)
                if prod_mol is None:
                    raise ValueError(f"Invalid product SMILES: {product_smiles}")
                prod_fp = Chem.PatternFingerprint(prod_mol)
                sims = DataStructs.BulkTverskySimilarity(prod_fp, self.template_pattern_fps, 0, 1)
                # alpha=0,beta=1: score 1.0 means the template product PatternFP
                # is contained by the target product PatternFP; score <1 cannot match.
                candidate_indices = [
                    i for i, score in enumerate(sims)
                    if float(score) >= 1.0
                ]
                top_indices = heapq.nlargest(
                    min(reaction_number, len(candidate_indices)),
                    candidate_indices,
                    key=lambda i: sims[i],
                )
                sorted_reaction_list = [self.reaction_list[i] for i in top_indices]
            else:
                # Original: DifferenceFP(reaction) + Tanimoto
                sanitized_reaction = self.sanitize_reaction(reaction_smiles)
                rxn_obj = smiles_to_reaction(sanitized_reaction)
                fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_obj)
                sims = DataStructs.BulkTanimotoSimilarity(fp_re, self.all_reaction_fps)
                top_indices = heapq.nlargest(reaction_number, range(len(sims)), key=lambda i: sims[i])
                if self.template_pattern_fps is not None:
                    try:
                        prod_mol = Chem.MolFromSmiles(product_smiles)
                        if prod_mol is not None:
                            prod_fp = Chem.PatternFingerprint(prod_mol)
                            product_scores = DataStructs.BulkTverskySimilarity(
                                prod_fp,
                                [self.template_pattern_fps[i] for i in top_indices],
                                0,
                                1,
                            )
                            top_indices = [
                                i for i, score in zip(top_indices, product_scores)
                                if float(score) >= 1.0
                            ]
                    except Exception as filter_error:
                        print(f"Error {filter_error} during product PatternFP hard-filter for {product_smiles[:50]}")
                sorted_reaction_list = [self.reaction_list[i] for i in top_indices]

        except Exception as e:
            print(f"Error {e} in blurry search ranking for {product_smiles[:50]}!")
            return self.rule_based_search(product_smiles, reaction_smiles)

        try:
            target_rd = rdchiralReactants(product_smiles)
        except Exception as e:
            print(f"Error {e} initializing rdchiralReactants for product {product_smiles}")
            return False, None, reaction_smiles

        for reaction_smarts in sorted_reaction_list:
            try:
                reaction_outputs = run_retro(target_rd, reaction_smarts)
                if len(reaction_outputs) > 1:
                    reaction_outputs = self.rank_reactants(reaction_outputs)
                if len(reaction_outputs) == 0:
                    continue
                reactants_generated = [reactant for reactant in reaction_outputs[0]]
                if reactants_generated == []:
                    continue
                elif len(reactants_generated) > 0:
                    key = self.template_to_key.get(reaction_smarts)
                    if (product_smiles, reaction_smarts) in self.explored_reaction:
                        print('redundant!')
                        continue
                    return True, key, reaction_smarts
            except Exception as e:
                print(f"Error {e} testing reaction {reaction_smarts} on product {product_smiles}")
                continue
        return self.rule_based_search(product_smiles, reaction_smiles)
    
    def sanitize(self, starting_list, route, exploration_signal):
        """
        Validation pipeline for LLM-proposed multi-step synthesis routes.

        Single forward pass: each step is validated with the previous step's
        corrected output as input (implicit cascade). Combines the logic that
        was previously split across Pass 1, cascade propagation, and Pass 2.

        Returns:
          (route, evaluation): Validated route and list of
            (step_index, is_valid, step_info) tuples
        """
        route[0]['Molecule set'] = str(starting_list)
        evaluation = self.check_route(starting_list, route, exploration_signal)
        return route, evaluation

    def check_route(self, target_smi, route, exploration_signal):
        """
        Unified single-pass route validation.

        Merges the old Pass 1 (validate with LLM's original sets), cascade
        (propagate corrections forward), and Pass 2 (re-validate with corrected
        sets) into one forward loop where each step sees the previous step's
        corrected output before being validated.

        For each step:
          (0) Fix input — cascade from previous step's corrected output
          (a) Molecule validity (parse SMILES, check starting_signal)
          (b) Reaction lookup (tagged / pattern match / skip if prev invalid)
          (c) Product matching (3 layers: exact / canonical / similarity+rdchiral)
          (d) Chemistry validation (verify_reaction_step if reaction found)
          (e) Aggressive salvaging (Steps 1+, if product missing AND prev auto-fixed)
          (f) Blurry search — Step 0: target=product[0], exploration_signal as-is
                              Steps 1+: target=resolved step product anchor, always exploration mode
          (g) Re-verify if salvaging or blurry search changed the step
          (h) Acceptance gate
          (i) Commit corrected bookkeeping for accepted steps
          (j) Update last_valid_molecule_set for next iteration (implicit cascade)
        """
        results = []
        last_valid_molecule_set = target_smi  # Cascade tracker

        # Per-trigger timing accumulators (for validation bottleneck analysis)
        import time as _time
        _trigger_log = []   # list of dicts, one per blurry_search call

        # Sub-operation timing accumulators
        _lookup_colon_s, _lookup_colon_calls = 0.0, 0      # [A1] ':' branch
        _lookup_nocolon_s, _lookup_nocolon_calls = 0.0, 0  # [A2] is_reaction_in_dict
        _product_check_s, _product_check_calls = 0.0, 0    # [B]  check_product_in_molecule_set
        _verify_s, _verify_calls = 0.0, 0                  # [C]  verify_reaction_step (initial)
        _reverify_s, _reverify_calls = 0.0, 0              # [D]  verify_reaction_step (re-verify)
        _salvage_s, _salvage_calls = 0.0, 0                # [E]  find_best_product_match_aggressive
        _step_log = []                                       # [F]  per-step: nocolon result + blurry trigger

        for i in range(len(route)):
            current_step = route[i]

            # --- (0) Fix input: cascade from previous step's corrected output ---
            if i == 0:
                current_step['Molecule set'] = str(target_smi)
            else:
                # Log cascade propagation
                _, prev_valid, prev_info = results[i - 1]
                if prev_valid:
                    if prev_info.get('auto_fixed', False):
                        print(f"✓ Step {i-1} accepted with auto-fixed molecule set (cascade)")
                        print(f"  Propagating to Step {i}: {last_valid_molecule_set}")
                else:
                    print(f"⏭️  Step {i-1} invalid, skipping (will try Step {i} with last valid set)")
                current_step['Molecule set'] = str(last_valid_molecule_set)

            # Parse fields from route
            step_validity = False
            molecule_set = ast.literal_eval(current_step['Molecule set'])
            updated_molecule_set = ast.literal_eval(current_step['Updated molecule set'])
            reaction = ast.literal_eval(current_step['Reaction'])[0]
            product = extract_molecules_from_output(current_step['Product'])
            reactants = ast.literal_eval(current_step['Reactants'])
            original_llm_product = product[0] if product else None
            anchor_product = None
            anchor_source = "none"
            salvage_attempted = False
            salvage_succeeded = False

            # --- (a) Molecule validity ---
            starting_signal = True
            if i == 0:
                if not set(molecule_set).issubset(set(target_smi)):
                    starting_signal = False

            invalid_molset_mol_id = []
            invalid_updated_mol_id = []

            updated_set_signals = check_validity(updated_molecule_set)
            if False in updated_set_signals:
                invalid_updated_mol_id = [index for index, value in enumerate(updated_set_signals) if not value]

            mol_set_signals = check_validity(molecule_set)
            if False in mol_set_signals:
                invalid_molset_mol_id = [index for index, value in enumerate(mol_set_signals) if not value]

            # Check purchasability (last step only)
            check_availability = False
            unavailable_mol_id = []
            if i == len(route) - 1:
                availibities = check_purchasable(updated_molecule_set, updated_set_signals, self.inventory)
                check_availability = True
            if check_availability == True:
                if False in availibities:
                    unavailable_mol_id = [index for index, value in enumerate(availibities) if not value]

            # --- (b) Reaction lookup ---
            # Snapshot accumulators before this step for per-step delta
            _nc_s_before = _lookup_nocolon_s
            _nc_calls_before = _lookup_nocolon_calls
            _tlog_len_before = len(_trigger_log)

            reaction_valid, updated_set_valid, reaction_existance = False, False, False
            corrected_updated_set = None
            generated_reactants = None
            auto_fixed = False
            corrections_map = {}

            if ':' in reaction:
                _t0 = _time.time()
                keys = [key for key, value in self.original_template_dict.items() if value == reaction]
                _lookup_colon_s += _time.time() - _t0; _lookup_colon_calls += 1
                if len(keys) == 1:
                    reaction_existance = True
                    reaction_key = keys[0]
                else:
                    reaction_existance = False
                    reaction_key = None
            else:
                # Pre-sort template keys if presort is enabled
                _sorted_keys, _tversky_scores = self.presort_keys(reaction) if self.use_presort else (None, None)
                if i == 0:
                    _t0 = _time.time()
                    reaction_existance, reaction_key = is_reaction_in_dict(reaction, self.template_dict, ordered_keys=_sorted_keys, tversky_scores=_tversky_scores)
                    _lookup_nocolon_s += _time.time() - _t0; _lookup_nocolon_calls += 1
                elif results[-1][1] == True:  # Conditional lookup: only if previous step valid
                    _t0 = _time.time()
                    reaction_existance, reaction_key = is_reaction_in_dict(reaction, self.template_dict, ordered_keys=_sorted_keys, tversky_scores=_tversky_scores)
                    _lookup_nocolon_s += _time.time() - _t0; _lookup_nocolon_calls += 1
                else:
                    reaction_existance = False  # Previous step invalid, skip lookup
                    reaction_key = None

            if reaction_key == None:
                new_reaction = reaction
            else:
                new_reaction = self.original_template_dict[reaction_key]

            # --- (c) Product matching (3 layers: exact / canonical / similarity+rdchiral) ---
            _t0 = _time.time()
            product_inside, match_type, matched_mol = check_product_in_molecule_set(
                product[0],
                molecule_set,
                i,
                reaction_template=new_reaction,
                inventory=self.inventory,
                oracle=self.oracle
            )
            _product_check_s += _time.time() - _t0; _product_check_calls += 1

            # Auto-correct Product field for Layer 2/3 matches
            if match_type in ["canonical", "reaction_validated", "reaction_validated_best"] and matched_mol:
                product = [matched_mol]
                current_step['Product'] = str(product)
                anchor_product = matched_mol
                anchor_source = match_type
                print(f"  ✓ Auto-corrected Product field via {match_type} matching")
            elif product_inside and product:
                anchor_product = product[0]
                anchor_source = match_type or "exact"

            # --- (d) Chemistry validation ---
            if reaction_existance == True:
                _t0 = _time.time()
                (reaction_valid, updated_set_valid, corrected_updated_set,
                generated_reactants, auto_fixed, corrections_map) = verify_reaction_step(
                    molecule_set, updated_molecule_set, new_reaction,
                    product, reactants, self.inventory, self.oracle
                )
                _verify_s += _time.time() - _t0; _verify_calls += 1

            # Track if we need re-verification after blurry search changes.
            needs_reverification = False

            # --- (e) Aggressive salvaging (Steps 1+, if product missing AND prev auto-fixed) ---
            if not product_inside and i > 0:
                if i - 1 < len(results):
                    _, _, prev_step_info = results[i - 1]
                    prev_auto_fixed = prev_step_info.get('auto_fixed', False)

                    if prev_auto_fixed and new_reaction:
                        print(f"  🔄 Previous step auto-fixed, trying aggressive salvaging...")
                        salvage_attempted = True
                        _t0 = _time.time()
                        salvaged_mol, salvage_match_type, salvaged_reactants = find_best_product_match_aggressive(
                            product[0], molecule_set, new_reaction,
                            self.inventory, self.oracle, i,
                            use_string_similarity=True,
                            string_similarity_weight=0.3
                        )
                        _salvage_s += _time.time() - _t0; _salvage_calls += 1

                        if salvaged_mol:
                            product_inside = True
                            match_type = salvage_match_type
                            matched_mol = salvaged_mol
                            product = [salvaged_mol]
                            current_step['Product'] = str(product)
                            anchor_product = salvaged_mol
                            anchor_source = salvage_match_type
                            salvage_succeeded = True
                            if salvaged_reactants:
                                reactants = salvaged_reactants
                                current_step['Reactants'] = str(reactants)
                            print(f"  ✅ Step salvaged via aggressive matching!")

                            if reaction_existance == True:
                                print(f"  🔄 Re-verifying reaction on salvaged product anchor...")
                                _t0 = _time.time()
                                (reaction_valid, updated_set_valid, corrected_updated_set,
                                 generated_reactants, auto_fixed, corrections_map) = verify_reaction_step(
                                    molecule_set, updated_molecule_set, new_reaction,
                                    product, reactants, self.inventory, self.oracle
                                )
                                _reverify_s += _time.time() - _t0; _reverify_calls += 1

            # --- (f) Blurry search ---
            # Metadata trackers
            blurry_search_fallback = False
            blurry_search_at_step = None
            blurry_search_molecule_used = None
            cascade_breakage_risk = False
            blurry_trigger_reason = None
            blurry_skipped_no_anchor = False

            if i == 0:
                # Step 0: search target = product[0], uses exploration_signal as-is
                step0_verified_reaction = new_reaction if reaction_existance == True else None
                # Trigger 1: Reaction not found
                if reaction_key == None:
                    _t0 = _time.time()
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(
                        reaction, product[0], exploration_signal
                    )
                    _trigger_log.append({"step": i, "trigger": 1, "reason": "reaction_not_found",
                                         "found": reaction_existance, "elapsed_s": round(_time.time() - _t0, 3),
                                         "explored_reaction_size": len(self.explored_reaction)})
                # Trigger 2: Already explored (avoid redundancy)
                elif (product[0], new_reaction) in self.explored_reaction:
                    _t0 = _time.time()
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(
                        reaction, product[0], exploration_signal
                    )
                    _trigger_log.append({"step": i, "trigger": 2, "reason": "already_explored",
                                         "found": reaction_existance, "elapsed_s": round(_time.time() - _t0, 3),
                                         "explored_reaction_size": len(self.explored_reaction)})
                # Trigger 3: Exists but invalid
                elif reaction_existance == True and reaction_valid == False:
                    _t0 = _time.time()
                    reaction_existance, reaction_key, new_reaction = self.blurry_search(
                        reaction, product[0], exploration_signal
                    )
                    _trigger_log.append({"step": i, "trigger": 3, "reason": "reaction_invalid",
                                         "found": reaction_existance, "elapsed_s": round(_time.time() - _t0, 3),
                                         "explored_reaction_size": len(self.explored_reaction)})

                if reaction_key == None:
                    new_reaction = reaction
                else:
                    # Re-verify only when Step 0 blurry search introduced a new
                    # reaction or when the initial reaction was never verified.
                    if step0_verified_reaction is None or new_reaction != step0_verified_reaction:
                        _t0 = _time.time()
                        (reaction_valid, updated_set_valid, corrected_updated_set,
                         generated_reactants, auto_fixed, corrections_map) = verify_reaction_step(
                            molecule_set, updated_molecule_set, new_reaction,
                            product, reactants, self.inventory, self.oracle
                        )
                        _reverify_s += _time.time() - _t0; _reverify_calls += 1
            else:
                # Steps 1+: blurry search must stay bound to the current step anchor.
                should_trigger_blurry = False
                blurry_reason = ""
                target_for_blurry = anchor_product

                if target_for_blurry is None:
                    if not product_inside and new_reaction is not None:
                        blurry_trigger_reason = "product not in molecule_set"
                        blurry_skipped_no_anchor = True
                        print(f"  ⏭️ Skipping blurry search for Step {i}: no valid product anchor")
                else:
                    # Trigger 1: Reaction not found
                    if reaction_key == None:
                        should_trigger_blurry = True
                        blurry_reason = "reaction not found"
                    # Trigger 2: Reaction already explored for the current anchor
                    elif new_reaction and (target_for_blurry, new_reaction) in self.explored_reaction:
                        should_trigger_blurry = True
                        blurry_reason = "reaction already explored"
                    # Trigger 3: Reaction exists but invalid on the current anchor
                    elif reaction_existance == True and reaction_valid == False:
                        should_trigger_blurry = True
                        blurry_reason = "reaction invalid"

                if should_trigger_blurry:
                    print(f"  🔍 Triggering blurry search for Step {i}: {blurry_reason}")
                    blurry_trigger_reason = blurry_reason

                    if i < len(route) - 1:
                        print(f"  ⚠️  WARNING: Blurry search on middle step may break downstream steps")
                        print(f"     Cascade breakage likely (downstream steps receive different molecule_set)")

                    already_explored_pair = (target_for_blurry, new_reaction if new_reaction else "")
                    allow_blurry_call = (
                        blurry_reason == "reaction already explored" or
                        already_explored_pair not in self.explored_reaction
                    )

                    if allow_blurry_call:
                        _t0 = _time.time()
                        blurry_existance, blurry_key, blurry_reaction = self.blurry_search(
                            new_reaction, target_for_blurry, exploration_signal=True
                        )
                        _trigger_num = {"reaction not found": 1, "reaction already explored": 2,
                                        "reaction invalid": 3, "product not in molecule_set": 4}.get(blurry_reason, 0)
                        _trigger_log.append({"step": i, "trigger": _trigger_num, "reason": blurry_reason,
                                             "found": blurry_existance, "elapsed_s": round(_time.time() - _t0, 3),
                                             "explored_reaction_size": len(self.explored_reaction)})

                        if blurry_existance and blurry_key:
                            print(f"  ✅ Blurry search found alternative reaction: {blurry_key}")
                            print(f"     Applied to molecule: {target_for_blurry[:50]}...")

                            self.explored_reaction.add((target_for_blurry, blurry_reaction))

                            # Update product to the molecule we searched on
                            product = [target_for_blurry]
                            current_step['Product'] = str(product)
                            anchor_product = target_for_blurry
                            anchor_source = f"blurry_{blurry_reason}"
                            product_inside = True

                            # Generate reactants from the blurry-searched reaction
                            try:
                                target_rd = rdchiralReactants(target_for_blurry)
                                reaction_outputs = run_retro(target_rd, blurry_reaction)
                                if len(reaction_outputs) > 1:
                                    reaction_outputs = self.rank_reactants(reaction_outputs)
                                if len(reaction_outputs) > 0:
                                    reactants = [reactant for reactant in reaction_outputs[0]]
                                    reactants = [_cached_sanitize(smi) for smi in reactants]
                                    current_step['Reactants'] = str(reactants)
                                    print(f"     Generated reactants: {reactants}")
                            except Exception as e:
                                print(f"  ⚠️  Error generating reactants: {e}")

                            new_reaction = blurry_reaction
                            reaction_existance = True
                            reaction_key = blurry_key
                            current_step['Reaction'] = str([new_reaction])

                            needs_reverification = True

                            # Blurry search metadata
                            blurry_search_fallback = True
                            blurry_search_at_step = i
                            blurry_search_molecule_used = target_for_blurry
                            cascade_breakage_risk = (i < len(route) - 1)

                            if i < len(route) - 1:
                                print(f"  ⚠️  Middle step using blurry-searched reaction")
                                print(f"     Downstream steps may fail product matching")
                            else:
                                print(f"  ℹ️  Final step using blurry-searched reaction")
                                print(f"     No downstream steps to break ✅")

                        else:
                            print(f"  ❌ Blurry search fallback failed - no alternative reaction found")
                    else:
                        print(f"  ⏭️ Skipping blurry search: (molecule, reaction) already explored")

            # --- (g) Re-verify if salvaging or blurry search changed the step ---
            if needs_reverification and reaction_existance == True:
                print(f"  🔄 Re-verifying reaction after salvaging/blurry search changes...")
                _t0 = _time.time()
                (reaction_valid, updated_set_valid, corrected_updated_set,
                 generated_reactants, auto_fixed, corrections_map) = verify_reaction_step(
                    molecule_set, updated_molecule_set, new_reaction,
                    product, reactants, self.inventory, self.oracle
                )
                _reverify_s += _time.time() - _t0; _reverify_calls += 1

            # --- (h) Acceptance gate ---
            if (len(invalid_molset_mol_id) == 0 and
                len(invalid_updated_mol_id) == 0 and
                reaction_valid and
                updated_set_valid and
                anchor_product is not None and
                starting_signal and
                (product_inside or match_type in ["canonical", "reaction_validated", "reaction_validated_best"])):
                step_validity = True

            # --- (i) Commit corrected bookkeeping for accepted steps ---
            if step_validity:
                current_step['Product'] = str([anchor_product])
                if corrected_updated_set is not None:
                    current_step['Updated molecule set'] = str(corrected_updated_set)
                if generated_reactants:
                    current_step['Reactants'] = str(generated_reactants)
                if auto_fixed and corrected_updated_set is not None:
                    print(f"✓ Step {i} accepted with auto-fixed molecule set")
                    print(f"  Generated reactants: {generated_reactants}")
                self.explored_reaction.add((anchor_product, new_reaction))

            # Persist reaction template into route
            current_step['Reaction'] = str([new_reaction])

            # --- (j) Update last_valid_molecule_set for next iteration ---
            if step_validity:
                corrected_set = corrected_updated_set
                if corrected_set is None:
                    try:
                        corrected_set = ast.literal_eval(current_step['Updated molecule set'])
                    except:
                        corrected_set = last_valid_molecule_set
                last_valid_molecule_set = corrected_set
            # Invalid step: keep last_valid unchanged (skip-and-continue)

            # --- Record per-step nocolon and blurry outcome ---
            _step_log.append({
                "s": i,
                "nc_s": round(_lookup_nocolon_s - _nc_s_before, 3),
                "nc_called": _lookup_nocolon_calls > _nc_calls_before,
                "found": bool(reaction_existance),
                "blurry": len(_trigger_log) > _tlog_len_before,
            })

            # --- Build step_info ---
            step_info = {
                "target_smi": target_smi,
                "starting_signal": starting_signal,
                "product_inside": product_inside,
                "molecule_set": molecule_set,
                "updated_molecule_set": updated_molecule_set,
                "corrected_updated_set": corrected_updated_set,
                "generated_reactants": generated_reactants,
                "auto_fixed": auto_fixed,
                "reaction": new_reaction,
                "reaction_key": reaction_key,
                "product": product,
                "reactants": reactants,
                "updated_set_signals": updated_set_signals,
                "invalid_updated_mol_id": invalid_updated_mol_id,
                "mol_set_signals": mol_set_signals,
                "invalid_molset_mol_id": invalid_molset_mol_id,
                "check_availability": check_availability,
                "unavailable_mol_id": unavailable_mol_id,
                "reaction_existance": reaction_existance,
                "reaction_valid": reaction_valid,
                "updated_set_valid": updated_set_valid,
                "anchor_product": anchor_product,
                "anchor_source": anchor_source,
                # Salvaging metadata
                "salvage_attempted": salvage_attempted,
                "salvage_succeeded": salvage_succeeded,
                "was_salvaged": salvage_succeeded,
                "salvage_similarity": float(match_type.split("_")[2]) if match_type and "similarity_matched" in match_type else None,
                "original_llm_product": original_llm_product if matched_mol and matched_mol != original_llm_product else None,
                "product_match_type": match_type,
                "product_matched_to": matched_mol,
                # Blurry search metadata
                "blurry_search_fallback": blurry_search_fallback,
                "blurry_search_at_step": blurry_search_at_step,
                "blurry_search_molecule_used": blurry_search_molecule_used,
                "cascade_breakage_risk": cascade_breakage_risk,
                "blurry_trigger_reason": blurry_trigger_reason,
                "blurry_skipped_no_anchor": blurry_skipped_no_anchor,
            }

            results.append((i, step_validity, step_info))

        # --- Per-trigger timing summary ---
        # Build compact summary and store on self for the caller to pick up
        _total_blurry_s = sum(e["elapsed_s"] for e in _trigger_log)
        _by_trigger = {}
        for e in _trigger_log:
            t = e["trigger"]
            _by_trigger.setdefault(t, {"calls": 0, "elapsed_s": 0.0})
            _by_trigger[t]["calls"] += 1
            _by_trigger[t]["elapsed_s"] = round(_by_trigger[t]["elapsed_s"] + e["elapsed_s"], 3)

        self._last_trigger_summary = {
            "blurry_calls": len(_trigger_log),
            "blurry_total_s": round(_total_blurry_s, 3),
            "blurry_explored_size": _trigger_log[-1]["explored_reaction_size"] if _trigger_log else len(self.explored_reaction),
            **{f"blurry_t{t}_{k}": v
               for t, stats in _by_trigger.items()
               for k, v in stats.items()},
            # Sub-operation timing breakdown
            "lookup_colon_s": round(_lookup_colon_s, 3),
            "lookup_colon_calls": _lookup_colon_calls,
            "lookup_nocolon_s": round(_lookup_nocolon_s, 3),
            "lookup_nocolon_calls": _lookup_nocolon_calls,
            "product_check_s": round(_product_check_s, 3),
            "product_check_calls": _product_check_calls,
            "verify_s": round(_verify_s, 3),
            "verify_calls": _verify_calls,
            "reverify_s": round(_reverify_s, 3),
            "reverify_calls": _reverify_calls,
            "salvage_s": round(_salvage_s, 3),
            "salvage_calls": _salvage_calls,
            "step_log": _step_log,
        }

        if _trigger_log:
            _summary_parts = [f"T{t}×{s['calls']}={s['elapsed_s']:.2f}s"
                               for t, s in sorted(_by_trigger.items())]
            print(f"[TRIGGER_LOG] blurry_calls={len(_trigger_log)}, "
                  f"total={_total_blurry_s:.2f}s, explored_size={self._last_trigger_summary['blurry_explored_size']}, "
                  f"breakdown={{{', '.join(_summary_parts)}}}")
        return results

    def rank_reactants(self, reactants_list):
        """Rank reactants based on the number of products generated"""
        non_empty_reactant_list = [item for item in reactants_list if item != []]
        scores = [self.oracle.reward(self.inventory, reactant, self.visited_molecules, self.dead_molecules) 
                 for reactant in non_empty_reactant_list]
        sorted_list = [x for _, x in sorted(zip(scores, non_empty_reactant_list), 
                                           key=lambda pair: pair[0], reverse=True)]
        return sorted_list

    def _get_llm_client(self, model):
        """Get or create a cached LLM client for this model.

        Reuses the same httpx.Client/OpenAI client across calls to avoid
        leaking connections. Creates a new client only on first call or
        when the model changes.
        """
        if not hasattr(self, '_llm_client_cache'):
            self._llm_client_cache = {}

        if model not in self._llm_client_cache:
            client, config = create_llm_client(model, self.args)
            self._llm_client_cache[model] = (client, config)

        return self._llm_client_cache[model]

    def query_LLM(self, question, model=None, temperature=0.7):
        """Query LLM for route generation

        Args:
            question: The prompt/question to send to the LLM
            model: OpenAI-compatible model name. If None, uses args.api_model.
            temperature: Sampling temperature (default: 0.7)

        Returns:
            tuple: (message_history, response_content)
        """
        # Use default model if not specified
        if model is None:
            model = getattr(self.args, "api_model", "llm-model")

        # Reuse cached client to avoid connection leaks
        client, config = self._get_llm_client(model)
        max_tokens = config.get("max_tokens", 4096)
        api_key_pool = get_api_key_pool(model, self.args)

        message = [{
            "role": "system",
            "content": "You are a retrosynthesis agent who can make multi-step retrosynthesis plans based on your molecule knowledge."
        }]

        message.append({"role": "user", "content": question})
        response_content = None

        attempt = 0
        rate_limit_count = 0
        query_start_time = time.time()

        while True:
            attempt += 1
            try:
                print(f"[LLM Query] Model: {model}, Attempt: {attempt}"
                      f"{f', rate_limit_retries: {rate_limit_count}' if rate_limit_count else ''}")
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )

                response_content = response.choices[0].message.content

                # Strip optional thinking tags if present.
                if response_content and '<think>' in response_content:
                    import re
                    response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()

                message.append({"role": "assistant", "content": response_content})
                elapsed = time.time() - query_start_time
                print(f"[LLM Query] Success - Response length: {len(response_content)} chars, "
                      f"total query time: {elapsed:.1f}s"
                      f"{f', after {rate_limit_count} rate limit retries' if rate_limit_count else ''}")
                break

            except Exception as e:
                error_name = type(e).__name__
                elapsed = time.time() - query_start_time
                print(f"[LLM Query] Error ({error_name}): {e} "
                      f"[attempt={attempt}, elapsed={elapsed:.0f}s]")

                is_rate_limit = ('RateLimitError' in error_name
                                 or 'rate' in str(e).lower()
                                 or 'Too Many Requests' in str(e))

                if is_rate_limit:
                    rate_limit_count += 1
                    # Randomly wait 1, 2, or 3 minutes
                    wait_time = random.choice([60, 120, 180])
                    print(f"[LLM Query] Rate limited (#{rate_limit_count}), "
                          f"waiting {wait_time}s before retry... "
                          f"[total elapsed: {elapsed:.0f}s]")
                    time.sleep(wait_time)

                    if len(api_key_pool) > 1:
                        next_key = random.choice(api_key_pool)
                        # Invalidate cached client and create fresh one with new key
                        self._llm_client_cache.pop(model, None)
                        client, config = self._get_llm_client(model)
                        client.api_key = next_key
                        print(f"[LLM Query] Rotated API key")
                else:
                    # Non-rate-limit error: give up after 5 attempts
                    if attempt >= 5:
                        print(f"[LLM Query] All retries exhausted after {attempt} attempts "
                              f"[total elapsed: {elapsed:.0f}s]")
                        response_content = None
                        break

        print("=>")
        return message, response_content

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)
        self.oracle.route_buffer = {}
        self.oracle.reaction_cache = dict()
        self.explored_reaction = set()
        self.visited_molecules = dict()
        self.dead_molecules = dict()

    def sort_buffer(self):
        self.oracle.sort_buffer()
    
    def log_intermediate(self, finish=False):
        self.oracle.log_intermediate(finish=finish)

    def save_result(self, suffix=None):
        print(f"Saving...")
        
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            suffix = suffix.replace("/", "")
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')
        self.sort_buffer()
        results_with_stats = {
            'oracle_calls': len(self.route_buffer),
            'routes': self.route_buffer
        }
        with open(output_file_path, 'w') as f:
            yaml.dump(self.route_buffer, f, sort_keys=False)

    @property
    def route_buffer(self):
        return self.oracle.route_buffer

    @property
    def finish(self):
        return self.oracle.finish
        
    def _optimize(self, target, route_list, all_fps, config):
        raise NotImplementedError
            
    def rewards(self, route_evaluation):
        return self.oracle(self.inventory, route_evaluation, self.visited_molecules, self.dead_molecules)
    
    def update_cache(self, mol_smiles, reaction):
        try:
            s = rdChemReactions.CreateDifferenceFingerprintForReaction(smiles_to_reaction(reaction))
            self.oracle.store_cache(mol_smiles, reaction)
        except:
            pass

    def optimize(self, target, route_list, all_fps, config, seed=0, project="test"):
        self.reset()
        clear_module_caches()
        self.seed = seed
        self.oracle.task_label = self.model_name + "_" + target + "_" + str(seed)
        
        try:
            with timeout(3600):  # 3600 seconds limit
                self._optimize(target, route_list, all_fps, config)
        except TimeoutError as e:
            print(f"Timeout for {target}: {e}")
            # Save existing results
            if len(self.oracle.route_buffer) > 0:
                print(f"Saved {len(self.oracle.route_buffer)} routes before timeout")
        
        if self.args.log_results:
            self.log_result()
        self.save_result(self.model_name + "_" + target + "_" + str(seed))


# Helper functions used in routes validation
# ============================================================================
# Module-level caches (cleared per search via clear_module_caches())
# ============================================================================
_canonical_cache = {}        # smiles -> canonical smiles (or None)
_morgan_fp_cache = {}        # smiles -> Morgan fingerprint (radius=2)


def clear_module_caches():
    """Clear search-local module caches.

    Retro template/result caches are intentionally kept process-local across
    target molecules; they depend only on product/template chemistry.
    """
    _canonical_cache.clear()
    _morgan_fp_cache.clear()


def _cached_sanitize(smi):
    """Cached version of sanitize_smiles."""
    if smi in _canonical_cache:
        return _canonical_cache[smi]
    result = sanitize_smiles(smi)
    _canonical_cache[smi] = result
    return result


def _cached_morgan_fp(smi):
    """Cached Morgan fingerprint (radius=2). Returns (mol, fp) or (None, None)."""
    if smi in _morgan_fp_cache:
        return _morgan_fp_cache[smi]
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprint(mol, 2)
        _morgan_fp_cache[smi] = (mol, fp)
        return (mol, fp)
    _morgan_fp_cache[smi] = (None, None)
    return (None, None)


def check_product_in_molecule_set(product_smiles, molecule_set, step_index=None,
                                  reaction_template=None, inventory=None, oracle=None):
    """
    Check if a product molecule exists in the molecule set.

    Why 3 layers: SMILES is not unique notation — the same molecule can be
    written multiple ways (e.g., "CCO" vs "OCC" vs "C(C)O" all mean ethanol).
    The LLM may use a different notation than what's in the molecule set.

    Layer 1: Exact string match — fast path, works when notation matches
    Layer 2: Canonical normalization — RDKit canonical form on both sides,
             catches notation differences
    Layer 3: Similarity + reaction validation — for >90% Tanimoto similarity,
             try running the reaction template via rdchiral. If it produces
             valid reactants, accept the match. Catches cases where the LLM
             wrote a slightly wrong SMILES for a very similar molecule.

    Args:
        product_smiles: Product SMILES string
        molecule_set: List of molecule SMILES
        step_index: Optional step index for logging
        reaction_template: Reaction template for Layer 3 validation
        inventory: Available molecules (for reaction validation)
        oracle: Oracle for ranking (if multiple valid candidates)

    Returns:
        (is_inside: bool, match_type: str, matched_molecule: str or None)
    """
    step_prefix = f"Step {step_index}: " if step_index is not None else ""

    # DEBUG: Show what we're checking
    if step_index is not None and step_index > 0:
        print(f"  🔍 DEBUG {step_prefix}Checking product_inside")
        print(f"     Product SMILES: '{product_smiles[:80]}...'")
        print(f"     Molecule set ({len(molecule_set)} molecules):")
        for i, mol in enumerate(molecule_set[:5]):  # Show first 5
            print(f"       [{i}] '{mol[:80]}...'")
        if len(molecule_set) > 5:
            print(f"       ... and {len(molecule_set) - 5} more")

    # Layer 1: Exact string match (fast path)
    if product_smiles in molecule_set:
        if step_index is not None and step_index > 0:
            print(f"  ✅ {step_prefix}Layer 1: Exact match found!")
        return (True, "exact", product_smiles)

    # Layer 2: Canonical SMILES match — O(1) lookup via pre-built dict
    product_canonical = _cached_sanitize(product_smiles)
    if product_canonical:
        # Build canonical->original mapping for the molecule set (cached per-smiles)
        canonical_to_original = {}
        for mol_smi in molecule_set:
            mol_canonical = _cached_sanitize(mol_smi)
            if mol_canonical:
                canonical_to_original[mol_canonical] = mol_smi
        if product_canonical in canonical_to_original:
            mol_smi = canonical_to_original[product_canonical]
            print(f"  ℹ️  {step_prefix}Product matched via canonical SMILES")
            print(f"     Product: '{product_smiles[:50]}...'")
            print(f"     Matched: '{mol_smi[:50]}...'")
            print(f"     (Same molecule, different SMILES notation)")
            return (True, "canonical", mol_smi)

    # Layer 3: Similarity-filtered reaction validation (hybrid approach)
    # Only attempt if we have reaction template (for validation)
    if reaction_template and len(molecule_set) <= 10:  # Don't try on huge sets
        try:
            from rdchiral.main import rdchiralReactants

            # Step 3a: Filter by similarity (>0.90 threshold)
            product_mol, product_fp = _cached_morgan_fp(product_smiles)
            if not product_mol:
                return (False, "none", None)

            # Find similar candidates
            candidates = []
            for mol_smi in molecule_set:
                mol, mol_fp = _cached_morgan_fp(mol_smi)
                if mol:
                    similarity = DataStructs.TanimotoSimilarity(product_fp, mol_fp)

                    if similarity > 0.90:  # High similarity threshold
                        candidates.append((mol_smi, similarity))

            # Sort by similarity (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)

            if not candidates:
                return (False, "none", None)

            print(f"  🔍 {step_prefix}Found {len(candidates)} similar candidates (>0.90)")
            print(f"     Original Product: '{product_smiles[:40]}...'")

            # Step 3b: Try reaction on candidates (in order of similarity)
            valid_matches = []
            for candidate_smi, sim in candidates:
                try:
                    # Attempt to run reaction backwards from candidate
                    target_rd = rdchiralReactants(candidate_smi)
                    reactants = run_retro(target_rd, reaction_template)

                    # Rank reactants if multiple outputs
                    if len(reactants) > 1 and inventory and oracle:
                        reactants = rank_reactants(reactants, inventory, oracle)

                    # Check if reaction produces valid reactants
                    if reactants and len(reactants) > 0:
                        reactants_flat = [r for r in reactants[0]]
                        reactants_clean = [_cached_sanitize(smi) for smi in reactants_flat]

                        if None not in reactants_clean and reactants_clean:
                            # Reaction works on this candidate!
                            valid_matches.append((candidate_smi, sim, reactants_clean))
                            print(f"     ✓ Candidate '{candidate_smi[:40]}...' (sim={sim:.3f})")
                            print(f"       → Reaction produces valid reactants: {reactants_clean}")

                except Exception as e:
                    # Reaction failed on this candidate
                    print(f"     ✗ Candidate '{candidate_smi[:40]}...' (sim={sim:.3f}): reaction failed")
                    continue

            # Return best match (highest similarity among valid ones)
            if len(valid_matches) == 1:
                matched_smi, sim, reactants = valid_matches[0]
                print(f"  ✅ {step_prefix}Reaction-validated match found!")
                print(f"     Original: '{product_smiles[:40]}...'")
                print(f"     Corrected: '{matched_smi[:40]}...' (similarity={sim:.3f})")
                print(f"     Reaction confirmed valid with reactants: {reactants}")
                return (True, "reaction_validated", matched_smi)

            elif len(valid_matches) > 1:
                # Multiple valid matches - pick most similar
                matched_smi, sim, reactants = valid_matches[0]  # Already sorted by similarity
                print(f"  ⚠️  {step_prefix}Multiple reaction-valid candidates")
                print(f"     Chose highest similarity: '{matched_smi[:40]}...' (sim={sim:.3f})")
                return (True, "reaction_validated_best", matched_smi)

            else:
                print(f"  ℹ️  {step_prefix}Similar candidates found but none chemically compatible")
                # Fall through to not found

        except Exception as e:
            print(f"  ⚠️  {step_prefix}Layer 3 validation failed: {e}")
            # Fall through to not found

    # Not found in any layer
    return (False, "none", None)


def find_best_product_match_aggressive(product_smiles, molecule_set, reaction_template,
                                       inventory, oracle, step_index=None,
                                       use_string_similarity=True, string_similarity_weight=0.3):
    """
    Find the best matching product in the molecule set when exact/canonical matching fails.

    When this triggers: After cascade, the molecule set may have changed
    (e.g., [A,B] -> [A,C]) so the LLM's declared product B no longer exists.
    But molecule C might be similar to B — the LLM was close but wrong.

    How it works:
      Score = 0.7 * Tanimoto(product, candidate) + 0.3 * string_similarity
      Rank all molecules in set by score.
      For each candidate (best first): run reaction template via rdchiral.
      First that produces valid reactants -> use this as the new product.

    Only triggered in Pass 2, Steps 1+, when the previous step was auto-fixed
    (because that's what caused the molecule set to change).

    Args:
        product_smiles: LLM's proposed product SMILES
        molecule_set: Available molecules after cascade update
        reaction_template: Reaction template to validate
        inventory: Available molecules for reactant ranking
        oracle: Oracle for ranking reactants
        step_index: Optional step index for logging
        use_string_similarity: Whether to include string similarity in ranking
        string_similarity_weight: Weight for string similarity (1 - tanimoto_weight)

    Returns:
        (matched_molecule: str or None, match_type: str, salvaged_reactants: list or None)
    """
    step_prefix = f"Step {step_index}: " if step_index is not None else ""

    try:
        from rdchiral.main import rdchiralReactants
        from difflib import SequenceMatcher

        # Step 1: Rank ALL molecules by hybrid similarity to LLM's proposed product
        product_mol, product_fp = _cached_morgan_fp(product_smiles)
        if not product_mol:
            return None, "none", None

        tanimoto_weight = 1.0 - string_similarity_weight

        candidates = []
        for mol_smi in molecule_set:
            mol, mol_fp = _cached_morgan_fp(mol_smi)
            if mol:
                # Tanimoto similarity (molecular fingerprint)
                tanimoto_sim = DataStructs.TanimotoSimilarity(product_fp, mol_fp)

                # String similarity (captures LLM's text-based reasoning)
                if use_string_similarity:
                    string_sim = SequenceMatcher(None, product_smiles, mol_smi).ratio()
                    hybrid_sim = tanimoto_weight * tanimoto_sim + string_similarity_weight * string_sim
                else:
                    string_sim = 0.0
                    hybrid_sim = tanimoto_sim

                candidates.append((mol_smi, hybrid_sim, tanimoto_sim, string_sim))

        # Sort by hybrid similarity (highest first) - try ALL candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"  🔄 {step_prefix}Aggressive salvaging: {len(candidates)} candidates")
        print(f"     LLM proposed: {product_smiles[:50]}...")
        if use_string_similarity:
            print(f"     Using hybrid similarity (Tanimoto={tanimoto_weight:.1f}, String={string_similarity_weight:.1f})")
        print(f"     Top 5 similar:")
        for i, (smi, hybrid_sim, tani_sim, str_sim) in enumerate(candidates[:5]):
            if use_string_similarity:
                print(f"       [{i+1}] {smi[:50]}... (hybrid={hybrid_sim:.3f}, T={tani_sim:.3f}, S={str_sim:.3f})")
            else:
                print(f"       [{i+1}] {smi[:50]}... (sim={hybrid_sim:.3f})")

        # Step 2: Try reaction on each candidate (in order of similarity)
        attempted_count = 0
        for candidate_smi, hybrid_sim, tani_sim, str_sim in candidates:
            attempted_count += 1
            try:
                target_rd = rdchiralReactants(candidate_smi)
                reactants = run_retro(target_rd, reaction_template)

                if reactants and len(reactants) > 0:
                    # Rank reactants if multiple outputs
                    if len(reactants) > 1 and inventory and oracle:
                        reactants = rank_reactants(reactants, inventory, oracle)

                    reactants_flat = [r for r in reactants[0]]
                    reactants_clean = [_cached_sanitize(smi) for smi in reactants_flat]

                    if None not in reactants_clean and reactants_clean:
                        # SUCCESS! Found valid chemistry
                        print(f"  ✅ {step_prefix}Found valid match via aggressive salvaging!")
                        print(f"     LLM proposed: {product_smiles[:40]}...")
                        if use_string_similarity:
                            print(f"     Using: {candidate_smi[:40]}... (hybrid={hybrid_sim:.3f}, T={tani_sim:.3f}, S={str_sim:.3f})")
                        else:
                            print(f"     Using: {candidate_smi[:40]}... (sim={hybrid_sim:.3f})")
                        print(f"     Reaction valid, reactants: {reactants_clean}")
                        print(f"     Attempts needed: {attempted_count}/{len(candidates)}")
                        return candidate_smi, f"similarity_matched_{hybrid_sim:.2f}", reactants_clean
            except Exception as e:
                continue

        # Step 3: All molecules failed - log for reaction search analysis
        print(f"  ❌ {step_prefix}Aggressive salvaging failed: no valid chemistry found")
        print(f"     Tried reaction on all {attempted_count} candidates, none worked")
        print(f"     LLM proposed molecule: {product_smiles[:60]}")
        print(f"     LLM proposed reaction: {reaction_template[:100] if len(reaction_template) < 100 else reaction_template[:100] + '...'}")
        print(f"     → POTENTIAL: Could try similar reactions here (future enhancement)")
        return None, "none", None

    except Exception as e:
        print(f"  ⚠️  {step_prefix}Aggressive salvaging error: {e}")
        return None, "none", None


def extract_molecules_from_output(output):
    """Extract molecules from LLM output"""
    try:
        parsed_output = ast.literal_eval(output)
        if isinstance(parsed_output, list):
            return parsed_output
        elif isinstance(parsed_output, str):
            return [parsed_output]
        else:
            return []
    except (ValueError, SyntaxError):
        return []
