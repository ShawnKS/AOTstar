from __future__ import annotations

import concurrent.futures
import heapq
import multiprocessing as mp
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem as _AllChem  # noqa: F401 - keep RDKit init parity
from rdkit.Chem import rdChemReactions
from rdchiral.initialization import rdchiralReactants
from syntheseus import Molecule

from aotcore.utils import run_retro, sanitize_smiles, smiles_to_reaction


DEFAULT_BLURRY_RERANK_TOPK = 20000
DEFAULT_BLURRY_RERANK_HEARTBEAT_SECONDS = 5 * 60

_PARALLEL_ORACLE_POOL = None
_PARALLEL_ORACLE_POOL_KEY = None
_PARALLEL_ORACLE_POOL_LOCK = threading.RLock()
_PARALLEL_ORACLE_WORKER = None


class _RouteBufferLen:
    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def parallel_oracle_enabled():
    return os.environ.get("AOT_PARALLEL_ORACLE_SCORING", "1") != "0"


def parallel_oracle_worker_count(default=4):
    requested = env_int("AOT_PARALLEL_ORACLE_WORKERS", default)
    cpu_count = os.cpu_count() or 1
    return max(0, min(requested, cpu_count))


def _parallel_oracle_worker_init(max_oracle_calls, freq_log, cache_size):
    global _PARALLEL_ORACLE_WORKER
    from aotcore.optimizer import Oracle

    previous_cache_size = os.environ.get("AOT_ORACLE_SCORE_CACHE_SIZE")
    os.environ["AOT_ORACLE_SCORE_CACHE_SIZE"] = str(cache_size)
    try:
        args = SimpleNamespace(
            max_oracle_calls=int(max_oracle_calls),
            freq_log=int(freq_log),
        )
        _PARALLEL_ORACLE_WORKER = Oracle(args=args, route_buffer=_RouteBufferLen(0))
    finally:
        if previous_cache_size is None:
            os.environ.pop("AOT_ORACLE_SCORE_CACHE_SIZE", None)
        else:
            os.environ["AOT_ORACLE_SCORE_CACHE_SIZE"] = previous_cache_size


def _parallel_oracle_score_one(task):
    smi, route_len = task
    if _PARALLEL_ORACLE_WORKER is None:
        raise RuntimeError("parallel oracle worker is not initialized")
    _PARALLEL_ORACLE_WORKER.route_buffer = _RouteBufferLen(route_len)
    try:
        score = float(_PARALLEL_ORACLE_WORKER.get_oracle_score(smi))
        static_scores = None
        cache = getattr(_PARALLEL_ORACLE_WORKER, "static_score_cache", None)
        if cache is not None and smi in cache:
            static_scores = cache[smi]
        return {
            "smi": smi,
            "ok": True,
            "score": score,
            "static_scores": static_scores,
            "error": None,
        }
    except Exception as exc:
        return {
            "smi": smi,
            "ok": False,
            "score": None,
            "static_scores": None,
            "error": repr(exc),
        }


def parallel_oracle_ping(_):
    return os.getpid()


def ensure_parallel_oracle_pool(max_oracle_calls, freq_log, cache_size, workers=None):
    global _PARALLEL_ORACLE_POOL, _PARALLEL_ORACLE_POOL_KEY
    if not parallel_oracle_enabled():
        return None
    if workers is None:
        workers = parallel_oracle_worker_count()
    workers = int(workers)
    if workers <= 1:
        return None
    key = (workers, int(max_oracle_calls), int(freq_log), int(cache_size))
    with _PARALLEL_ORACLE_POOL_LOCK:
        if _PARALLEL_ORACLE_POOL is not None and _PARALLEL_ORACLE_POOL_KEY == key:
            return _PARALLEL_ORACLE_POOL
        if _PARALLEL_ORACLE_POOL is not None:
            _PARALLEL_ORACLE_POOL.shutdown(wait=True, cancel_futures=False)
            _PARALLEL_ORACLE_POOL = None
            _PARALLEL_ORACLE_POOL_KEY = None
        try:
            ctx = mp.get_context("fork")
            _PARALLEL_ORACLE_POOL = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_parallel_oracle_worker_init,
                initargs=(max_oracle_calls, freq_log, cache_size),
            )
            _PARALLEL_ORACLE_POOL_KEY = key
            return _PARALLEL_ORACLE_POOL
        except Exception as exc:
            print(f"[PARALLEL_ORACLE] disabled after pool init error: {exc}", flush=True)
            _PARALLEL_ORACLE_POOL = None
            _PARALLEL_ORACLE_POOL_KEY = None
            return None


def shutdown_parallel_oracle_pool():
    global _PARALLEL_ORACLE_POOL, _PARALLEL_ORACLE_POOL_KEY
    with _PARALLEL_ORACLE_POOL_LOCK:
        if _PARALLEL_ORACLE_POOL is not None:
            _PARALLEL_ORACLE_POOL.shutdown(wait=True, cancel_futures=False)
            _PARALLEL_ORACLE_POOL = None
            _PARALLEL_ORACLE_POOL_KEY = None


def pattern_fps_from_templates(original_template_dict, template_keys, side_index):
    fps = []
    for key in template_keys:
        smarts = original_template_dict[key]
        try:
            side = smarts.split(">>", 1)[side_index]
            combined = None
            for part in side.split("."):
                mol = Chem.MolFromSmarts(part)
                if mol is None:
                    continue
                fp = Chem.PatternFingerprint(mol)
                combined = fp if combined is None else (combined | fp)
            fps.append(combined if combined is not None else DataStructs.ExplicitBitVect(2048))
        except Exception:
            fps.append(DataStructs.ExplicitBitVect(2048))
    return fps


def build_template_reactant_pattern_fps(opt):
    return pattern_fps_from_templates(
        {str(i): reaction for i, reaction in enumerate(opt.reaction_list)},
        [str(i) for i in range(len(opt.reaction_list))],
        1,
    )


def _is_purchasable(opt, smi):
    clean = sanitize_smiles(smi)
    if not clean:
        return False
    try:
        return bool(opt.inventory.is_purchasable(Molecule(clean)))
    except Exception:
        return False


def _purchase_status_for_score(opt, smi):
    try:
        return True, bool(opt.inventory.is_purchasable(Molecule(smi))), None
    except Exception as exc:
        return False, False, exc


def _score_reactants_via_reward(opt, clean, gamma):
    reward = opt.oracle.reward(
        opt.inventory,
        clean,
        opt.visited_molecules,
        opt.dead_molecules,
    )
    unsolved_count = sum(1 for smi in clean if not _is_purchasable(opt, smi))
    adjusted_value = float(reward) * (1.0 + unsolved_count * gamma)
    return {
        "reactants": clean,
        "reward": float(reward),
        "unsolved_count": unsolved_count,
        "adjusted_value": adjusted_value,
    }


def _score_reactants_fused(opt, clean, gamma):
    statuses = [_purchase_status_for_score(opt, smi) for smi in clean]
    unsolved_count = sum(1 for ok, is_purch, _ in statuses if (not ok) or (not is_purch))
    score_list = []
    for smi, (ok, is_purch, err) in zip(clean, statuses):
        if smi in opt.dead_molecules and opt.dead_molecules[smi] >= 1:
            print("dead molecules!")
            score_list.append(100)
            continue
        if not ok:
            print(f"Error: {err}")
            score_list.append(5)
            continue
        if is_purch:
            continue
        try:
            score = opt.oracle.get_oracle_score(smi)
            if smi in opt.visited_molecules:
                print(f"Visited times: {opt.visited_molecules[smi]}")
                if opt.visited_molecules[smi] > 15:
                    score = (opt.visited_molecules[smi] / 15) * score
                    print("Visited times adjust score")
            score_list.append(score)
        except Exception as exc:
            print(f"Error: {exc}")
            score_list.append(5)
    if score_list:
        reward = -((sum(score_list) / len(score_list)) + sum(score_list))
    else:
        reward = 0
    adjusted_value = float(reward) * (1.0 + unsolved_count * gamma)
    return {
        "reactants": clean,
        "reward": float(reward),
        "unsolved_count": unsolved_count,
        "adjusted_value": adjusted_value,
    }


def _score_reactants(opt, reactants, score_cache=None):
    clean = [sanitize_smiles(smi) for smi in reactants]
    clean = [smi for smi in clean if smi]
    if not clean:
        return None
    cache_key = tuple(clean)
    if score_cache is not None and cache_key in score_cache:
        return score_cache[cache_key]
    gamma = float(getattr(opt, "unsolved_penalty", 0.1))
    if os.environ.get("AOT_FUSED_SCORE_REACTANTS", "1") == "0":
        result = _score_reactants_via_reward(opt, clean, gamma)
    else:
        result = _score_reactants_fused(opt, clean, gamma)
    if score_cache is not None:
        score_cache[cache_key] = result
    return result


def _clean_reactants(reactants):
    clean = [sanitize_smiles(smi) for smi in reactants]
    return [smi for smi in clean if smi]


def _zero_score_from_statuses(clean, statuses, dead_molecules):
    if not clean:
        return False
    for smi, (ok, is_purch, _) in zip(clean, statuses):
        if smi in dead_molecules and dead_molecules[smi] >= 1:
            return False
        if (not ok) or (not is_purch):
            return False
    return True


def _merge_static_scores(oracle, rows):
    cache = getattr(oracle, "static_score_cache", None)
    cache_size = int(getattr(oracle, "static_score_cache_size", 0) or 0)
    if cache is None or cache_size <= 0:
        return
    for row in rows:
        static_scores = row.get("static_scores")
        if not static_scores:
            continue
        smi = row["smi"]
        new_sc, new_sa = static_scores
        old_sc, old_sa = cache.get(smi, (None, None))
        cache[smi] = (
            new_sc if new_sc is not None else old_sc,
            new_sa if new_sa is not None else old_sa,
        )
        cache.move_to_end(smi)
        while len(cache) > cache_size:
            cache.popitem(last=False)


def _parallel_oracle_scores(opt, smiles, route_len):
    if not smiles:
        return {}, set()
    cache_size = int(getattr(opt.oracle, "static_score_cache_size", 0) or 0)
    workers = parallel_oracle_worker_count()
    pool = ensure_parallel_oracle_pool(
        getattr(opt.oracle, "max_oracle_calls", 10000),
        getattr(opt.oracle, "freq_log", 1000000),
        cache_size,
        workers=workers,
    )
    if pool is None:
        scores = {}
        failures = set()
        for smi in smiles:
            try:
                scores[smi] = float(opt.oracle.get_oracle_score(smi))
            except Exception:
                failures.add(smi)
        return scores, failures

    chunksize = max(1, env_int("AOT_PARALLEL_ORACLE_CHUNKSIZE", 8))
    try:
        rows = list(pool.map(
            _parallel_oracle_score_one,
            [(smi, route_len) for smi in smiles],
            chunksize=chunksize,
        ))
    except Exception as exc:
        print(f"[PARALLEL_ORACLE] falling back to sequential after map error: {exc}", flush=True)
        scores = {}
        failures = set()
        for smi in smiles:
            try:
                scores[smi] = float(opt.oracle.get_oracle_score(smi))
            except Exception:
                failures.add(smi)
        return scores, failures

    _merge_static_scores(opt.oracle, rows)
    scores = {}
    failures = set()
    for row in rows:
        if row.get("ok"):
            scores[row["smi"]] = float(row["score"])
        else:
            failures.add(row["smi"])
    return scores, failures


def _oracle_scores_for_records(opt, records):
    seen = set()
    needed = []
    for record in records:
        clean = record["clean"]
        statuses = record["statuses"]
        for smi, (ok, is_purch, _) in zip(clean, statuses):
            if smi in opt.dead_molecules and opt.dead_molecules[smi] >= 1:
                continue
            if (not ok) or is_purch:
                continue
            if smi not in seen:
                seen.add(smi)
                needed.append(smi)

    route_len = len(getattr(opt.oracle, "route_buffer", {}))
    max_parallel_route_len = env_int("AOT_PARALLEL_ORACLE_MAX_ROUTE_LEN", 219)
    min_molecules = max(1, env_int("AOT_PARALLEL_ORACLE_MIN_MOLECULES", 24))
    sample_size = max(0, env_int("AOT_PARALLEL_ORACLE_SAMPLE_SIZE", 4))
    min_sample_ms = max(0.0, env_float("AOT_PARALLEL_ORACLE_MIN_SAMPLE_MS", 0.0))

    scores = {}
    failures = set()
    if not needed:
        return scores, failures, {
            "needed": 0,
            "parallel_used": False,
            "reason": "no_oracle_molecules",
        }

    sample = needed[: min(sample_size, len(needed))]
    sample_elapsed = 0.0
    if sample:
        t_sample = time.time()
        for smi in sample:
            try:
                scores[smi] = float(opt.oracle.get_oracle_score(smi))
            except Exception:
                failures.add(smi)
        sample_elapsed = time.time() - t_sample

    remaining = [smi for smi in needed if smi not in scores and smi not in failures]
    avg_sample_ms = (sample_elapsed / len(sample) * 1000.0) if sample else None
    should_parallel = (
        parallel_oracle_enabled()
        and route_len <= max_parallel_route_len
        and len(remaining) >= min_molecules
        and (
            avg_sample_ms is None
            or avg_sample_ms >= min_sample_ms
        )
    )
    if not should_parallel:
        reason = "below_threshold"
        if route_len > max_parallel_route_len:
            reason = "route_len"
        elif len(remaining) < min_molecules:
            reason = "few_molecules"
        elif avg_sample_ms is not None and avg_sample_ms < min_sample_ms:
            reason = "cheap_sample"
        for smi in remaining:
            try:
                scores[smi] = float(opt.oracle.get_oracle_score(smi))
            except Exception:
                failures.add(smi)
        return scores, failures, {
            "needed": len(needed),
            "sampled": len(sample),
            "sample_ms": avg_sample_ms,
            "remaining": len(remaining),
            "parallel_used": False,
            "reason": reason,
        }

    t_parallel = time.time()
    parallel_scores, parallel_failures = _parallel_oracle_scores(opt, remaining, route_len)
    parallel_s = time.time() - t_parallel
    scores.update(parallel_scores)
    for smi in parallel_failures:
        try:
            scores[smi] = float(opt.oracle.get_oracle_score(smi))
        except Exception:
            failures.add(smi)
    return scores, failures, {
        "needed": len(needed),
        "sampled": len(sample),
        "sample_ms": avg_sample_ms,
        "remaining": len(remaining),
        "parallel_used": True,
        "reason": "parallel",
        "parallel_s": parallel_s,
    }


def _score_from_statuses(opt, clean, statuses, oracle_scores, oracle_failures, gamma):
    score_list = []
    for smi, (ok, is_purch, err) in zip(clean, statuses):
        if smi in opt.dead_molecules and opt.dead_molecules[smi] >= 1:
            print("dead molecules!")
            score_list.append(100)
            continue
        if not ok:
            print(f"Error: {err}")
            score_list.append(5)
            continue
        if is_purch:
            continue
        if smi in oracle_failures:
            print(f"Error scoring oracle for {smi}")
            score_list.append(5)
            continue
        try:
            score = float(oracle_scores[smi])
            if smi in opt.visited_molecules:
                print(f"Visited times: {opt.visited_molecules[smi]}")
                if opt.visited_molecules[smi] > 15:
                    score = (opt.visited_molecules[smi] / 15) * score
                    print("Visited times adjust score")
            score_list.append(score)
        except Exception as exc:
            print(f"Error: {exc}")
            score_list.append(5)
    if score_list:
        reward = -((sum(score_list) / len(score_list)) + sum(score_list))
    else:
        reward = 0.0
    unsolved_count = sum(1 for ok, is_purch, _ in statuses if (not ok) or (not is_purch))
    adjusted_value = float(reward) * (1.0 + unsolved_count * gamma)
    return {
        "reactants": clean,
        "reward": float(reward),
        "unsolved_count": unsolved_count,
        "adjusted_value": adjusted_value,
    }


def _candidate_from_score(record, score_info):
    candidate = {
        "reaction_smarts": record["reaction_smarts"],
        "template_index": record["template_index"],
        "rank": record["rank"],
        "similarity": record["similarity"],
        **score_info,
    }
    if record["reactant_similarity"] is not None:
        candidate["reactant_similarity"] = record["reactant_similarity"]
    return candidate


def _candidate_better(candidate, best, uses_reactant_rerank, reactionfp_selection_mode):
    if best is None:
        return True
    if uses_reactant_rerank and reactionfp_selection_mode == "path_then_oracle":
        candidate_key = (
            candidate.get("reactant_similarity", 0.0),
            candidate["similarity"],
            candidate["adjusted_value"],
        )
        best_key = (
            best.get("reactant_similarity", 0.0),
            best["similarity"],
            best["adjusted_value"],
        )
        return candidate_key > best_key
    if uses_reactant_rerank and reactionfp_selection_mode in {"oracle", "oracle_first"}:
        candidate_key = (
            candidate["adjusted_value"],
            candidate.get("reactant_similarity", 0.0),
            candidate["similarity"],
        )
        best_key = (
            best["adjusted_value"],
            best.get("reactant_similarity", 0.0),
            best["similarity"],
        )
        return candidate_key > best_key
    return candidate["adjusted_value"] > best["adjusted_value"]


def _return_best_candidate(opt, best, topk, applicable, skipped_explored,
                           product_fp_skipped, early_stopped_remaining,
                           early_stop_reason):
    if best is None:
        return None
    key = opt.template_to_key.get(best["reaction_smarts"])
    reactant_sim_text = ""
    if "reactant_similarity" in best:
        reactant_sim_text = f" reactant_sim={best['reactant_similarity']:.4f}"
    print(
        "[BLURRY_RERANK] "
        f"topk={topk} applicable={applicable} skipped_explored={skipped_explored} "
        f"product_fp_skipped={product_fp_skipped} early_stop_remaining={early_stopped_remaining} "
        f"early_stop_reason={early_stop_reason or 'none'} "
        f"selected_rank={best['rank']} sim={best['similarity']:.4f} "
        f"reward={best['reward']:.4f} unsolved={best['unsolved_count']}{reactant_sim_text} "
        f"adjusted={best['adjusted_value']:.4f}"
    )
    return True, key, best["reaction_smarts"]


def _pattern_fp_from_smiles_parts(parts):
    combined = None
    for smi in parts:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = Chem.PatternFingerprint(mol)
        combined = fp if combined is None else (combined | fp)
    return combined


def _filter_by_product_pattern_fp(opt, product_smiles, ranked_templates):
    template_pattern_fps = getattr(opt, "template_pattern_fps", None)
    if not template_pattern_fps or not ranked_templates:
        return ranked_templates, 0
    try:
        prod_mol = Chem.MolFromSmiles(product_smiles)
        if prod_mol is None:
            return ranked_templates, 0
        prod_fp = Chem.PatternFingerprint(prod_mol)
        candidate_fps = [template_pattern_fps[template_index] for _, _, template_index, _ in ranked_templates]
        scores = DataStructs.BulkTverskySimilarity(prod_fp, candidate_fps, 0, 1)
    except Exception as exc:
        print(f"Error {exc} during product PatternFP hard-filter for {product_smiles[:50]}")
        return ranked_templates, 0
    filtered = [
        item for item, score in zip(ranked_templates, scores)
        if float(score) >= 1.0
    ]
    return filtered, len(ranked_templates) - len(filtered)


def _reactionfp_query(opt, reaction_smiles, product_smiles):
    sanitized_reaction = opt.sanitize_reaction(reaction_smiles)
    if ">>" not in sanitized_reaction:
        return sanitized_reaction, None
    _, rhs = sanitized_reaction.split(">>", 1)
    anchor_product = opt.sanitize_smiles(product_smiles)
    if anchor_product and rhs:
        sanitized_reaction = f"{anchor_product}>>{rhs}"
    rhs_parts = [part for part in rhs.split(".") if part]
    return sanitized_reaction, _pattern_fp_from_smiles_parts(rhs_parts)


def _rank_pool_by_reactant_similarity(llm_reactant_fp, template_reactant_fps, pool_indices, sims, topk):
    if not pool_indices:
        return []

    if os.environ.get("AOT_BULK_REACTANT_SIM", "1") != "0":
        pool_reactant_fps = [template_reactant_fps[i] for i in pool_indices]
        reactant_scores = DataStructs.BulkTverskySimilarity(
            llm_reactant_fp,
            pool_reactant_fps,
            0,
            1,
        )
        ordered_positions = sorted(
            range(len(pool_indices)),
            key=lambda pos: (reactant_scores[pos], sims[pool_indices[pos]]),
            reverse=True,
        )[: min(topk, len(pool_indices))]
        return [
            (pool_indices[pos], float(reactant_scores[pos]))
            for pos in ordered_positions
        ]

    reactant_sims = {
        i: float(DataStructs.TverskySimilarity(llm_reactant_fp, template_reactant_fps[i], 0, 1))
        for i in pool_indices
    }
    top_indices = sorted(
        pool_indices,
        key=lambda i: (reactant_sims[i], sims[i]),
        reverse=True,
    )[: min(topk, len(pool_indices))]
    return [(i, reactant_sims[i]) for i in top_indices]


def _rank_templates(opt, reaction_smiles, product_smiles, topk, reactionfp_mode):
    uses_reactant_rerank = False
    if opt.blurry_tversky and opt.template_pattern_fps is not None:
        prod_mol = Chem.MolFromSmiles(product_smiles)
        if prod_mol is None:
            raise ValueError(f"Invalid product SMILES: {product_smiles}")
        prod_fp = Chem.PatternFingerprint(prod_mol)
        sims = DataStructs.BulkTverskySimilarity(prod_fp, opt.template_pattern_fps, 0, 1)
        candidate_indices = [
            i for i, score in enumerate(sims)
            if float(score) >= 1.0
        ]
        top_indices = heapq.nlargest(
            min(topk, len(candidate_indices)),
            candidate_indices,
            key=lambda i: sims[i],
        )
        return [(opt.reaction_list[i], float(sims[i]), i, None) for i in top_indices], uses_reactant_rerank

    sanitized_reaction, llm_reactant_fp = _reactionfp_query(opt, reaction_smiles, product_smiles)
    if reactionfp_mode == "product_tversky_reactants":
        prod_mol = Chem.MolFromSmiles(product_smiles)
        if prod_mol is None:
            raise ValueError(f"Invalid product SMILES: {product_smiles}")
        if opt.template_pattern_fps is None:
            opt.template_pattern_fps = opt._build_template_pattern_fps()
        prod_fp = Chem.PatternFingerprint(prod_mol)
        sims = DataStructs.BulkTverskySimilarity(prod_fp, opt.template_pattern_fps, 0, 1)
    else:
        rxn_obj = smiles_to_reaction(sanitized_reaction)
        if rxn_obj is None:
            raise ValueError(f"Invalid reaction query: {sanitized_reaction}")
        fp_re = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_obj)
        if reactionfp_mode in {"tversky", "tversky_reactants"}:
            sims = DataStructs.BulkTverskySimilarity(fp_re, opt.all_reaction_fps, 0, 1)
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp_re, opt.all_reaction_fps)

    uses_reactant_rerank = reactionfp_mode in {
        "tversky_reactants",
        "product_tversky_reactants",
    }
    if uses_reactant_rerank and (
        llm_reactant_fp is not None or reactionfp_mode == "product_tversky_reactants"
    ):
        if reactionfp_mode == "product_tversky_reactants":
            pool_indices = [
                i for i, score in enumerate(sims)
                if float(score) >= 1.0
            ]
        else:
            candidate_pool = int(getattr(opt.args, "reactionfp_candidate_pool", 0) or 0)
            if candidate_pool <= 0:
                candidate_pool = max(topk, 2000)
            else:
                candidate_pool = max(topk, candidate_pool)
            pool_indices = heapq.nlargest(
                min(candidate_pool, len(sims)),
                range(len(sims)),
                key=lambda i: sims[i],
            )
        if llm_reactant_fp is not None:
            template_reactant_fps = getattr(opt, "template_reactant_pattern_fps", None)
            if template_reactant_fps is None:
                template_reactant_fps = build_template_reactant_pattern_fps(opt)
                opt.template_reactant_pattern_fps = template_reactant_fps
            ranked_with_reactant_scores = _rank_pool_by_reactant_similarity(
                llm_reactant_fp,
                template_reactant_fps,
                pool_indices,
                sims,
                topk,
            )
            return [
                (opt.reaction_list[i], float(sims[i]), i, reactant_sim)
                for i, reactant_sim in ranked_with_reactant_scores
            ], uses_reactant_rerank
        top_indices = heapq.nlargest(
            min(topk, len(pool_indices)),
            pool_indices,
            key=lambda i: sims[i],
        )
        return [
            (opt.reaction_list[i], float(sims[i]), i, None)
            for i in top_indices
        ], uses_reactant_rerank

    top_indices = heapq.nlargest(min(topk, len(sims)), range(len(sims)), key=lambda i: sims[i])
    return [(opt.reaction_list[i], float(sims[i]), i, None) for i in top_indices], uses_reactant_rerank


def blurry_search_oracle_rerank(opt, reaction_smiles, product_smiles, exploration_signal):
    topk = int(getattr(opt.args, "blurry_rerank_topk", DEFAULT_BLURRY_RERANK_TOPK))
    if topk <= 0:
        topk = DEFAULT_BLURRY_RERANK_TOPK
    heartbeat_interval = float(
        getattr(opt.args, "blurry_rerank_heartbeat_seconds", DEFAULT_BLURRY_RERANK_HEARTBEAT_SECONDS)
    )
    reactionfp_mode = str(getattr(opt.args, "reactionfp_rerank_mode", "tanimoto"))
    reactionfp_selection_mode = str(getattr(opt.args, "reactionfp_selection_mode", "oracle"))

    try:
        ranked_templates, uses_reactant_rerank = _rank_templates(
            opt, reaction_smiles, product_smiles, topk, reactionfp_mode
        )
    except Exception as exc:
        print(f"Error {exc} in blurry oracle-rerank for {product_smiles[:50]}!")
        return opt.rule_based_search(product_smiles, reaction_smiles)

    product_fp_skipped = 0
    if not opt.blurry_tversky and reactionfp_mode != "product_tversky_reactants":
        ranked_templates, product_fp_skipped = _filter_by_product_pattern_fp(
            opt,
            product_smiles,
            ranked_templates,
        )

    try:
        target_rd = rdchiralReactants(product_smiles)
    except Exception as exc:
        print(f"Error {exc} initializing rdchiralReactants for product {product_smiles}")
        return False, None, reaction_smiles

    best = None
    applicable = 0
    skipped_explored = 0
    early_stopped_remaining = 0
    early_stop_reason = ""
    score_info_cache = {}
    started_at = time.time()
    last_report = started_at
    batch_oracle_scoring = (
        parallel_oracle_enabled()
        and reactionfp_selection_mode in {"oracle", "oracle_first"}
    )
    pending_records = []

    for rank, (reaction_smarts, sim_score, template_index, reactant_sim_score) in enumerate(ranked_templates, start=1):
        stop_after_current = False
        now = time.time()
        if heartbeat_interval > 0 and now - last_report >= heartbeat_interval:
            best_rank = best["rank"] if best is not None else "none"
            print(
                "[BLURRY_RERANK_PROGRESS] "
                f"topk={topk} rank={rank}/{len(ranked_templates)} "
                f"applicable={applicable} skipped_explored={skipped_explored} "
                f"product_fp_skipped={product_fp_skipped} "
                f"best_rank={best_rank} elapsed={now - started_at:.0f}s",
                flush=True,
            )
            last_report = now

        if (
            best is not None
            and uses_reactant_rerank
            and reactionfp_selection_mode == "path_then_oracle"
            and reactant_sim_score is not None
        ):
            current_prefix = (float(reactant_sim_score), float(sim_score))
            best_prefix = (
                float(best.get("reactant_similarity", 0.0)),
                float(best["similarity"]),
            )
            if current_prefix < best_prefix:
                early_stopped_remaining = len(ranked_templates) - rank + 1
                early_stop_reason = "path_prefix"
                break

        if (product_smiles, reaction_smarts) in opt.explored_reaction:
            skipped_explored += 1
            continue
        try:
            reaction_outputs = run_retro(target_rd, reaction_smarts)
        except Exception as exc:
            print(f"Error {exc} testing reaction {reaction_smarts} on product {product_smiles}")
            continue
        if not reaction_outputs:
            continue

        applicable += 1
        for output in reaction_outputs:
            if batch_oracle_scoring:
                clean = _clean_reactants(output)
                if not clean:
                    continue
                statuses = [_purchase_status_for_score(opt, smi) for smi in clean]
                record = {
                    "reaction_smarts": reaction_smarts,
                    "template_index": template_index,
                    "rank": rank,
                    "similarity": sim_score,
                    "reactant_similarity": reactant_sim_score,
                    "clean": clean,
                    "statuses": statuses,
                }
                if _zero_score_from_statuses(clean, statuses, opt.dead_molecules):
                    score_info = {
                        "reactants": clean,
                        "reward": 0.0,
                        "unsolved_count": 0,
                        "adjusted_value": 0.0,
                    }
                    best = _candidate_from_score(record, score_info)
                    stop_after_current = True
                    early_stopped_remaining = len(ranked_templates) - rank
                    early_stop_reason = "oracle_upper_bound"
                    break
                pending_records.append(record)
                continue

            score_info = _score_reactants(opt, output, score_info_cache)
            if score_info is None:
                continue
            record = {
                "reaction_smarts": reaction_smarts,
                "template_index": template_index,
                "rank": rank,
                "similarity": sim_score,
                "reactant_similarity": reactant_sim_score,
            }
            candidate = _candidate_from_score(record, score_info)
            if _candidate_better(candidate, best, uses_reactant_rerank, reactionfp_selection_mode):
                best = candidate

            if (
                reactionfp_selection_mode in {"oracle", "oracle_first"}
                and best is not None
                and best["adjusted_value"] == 0.0
            ):
                stop_after_current = True
                early_stopped_remaining = len(ranked_templates) - rank
                early_stop_reason = "oracle_upper_bound"
                break

        if stop_after_current:
            break

    if batch_oracle_scoring and best is None and pending_records:
        gamma = float(getattr(opt, "unsolved_penalty", 0.1))
        t_score = time.time()
        oracle_scores, oracle_failures, oracle_stats = _oracle_scores_for_records(opt, pending_records)
        scoring_s = time.time() - t_score
        if oracle_stats.get("needed", 0) > 0:
            sample_ms = oracle_stats.get("sample_ms")
            sample_text = "none" if sample_ms is None else f"{sample_ms:.1f}"
            print(
                "[PARALLEL_ORACLE] "
                f"needed={oracle_stats.get('needed')} sampled={oracle_stats.get('sampled', 0)} "
                f"sample_ms={sample_text} remaining={oracle_stats.get('remaining', 0)} "
                f"used={oracle_stats.get('parallel_used')} reason={oracle_stats.get('reason')} "
                f"score_s={scoring_s:.2f}",
                flush=True,
            )
        score_info_cache = {}
        for record in pending_records:
            cache_key = tuple(record["clean"])
            if cache_key in score_info_cache:
                score_info = score_info_cache[cache_key]
            else:
                score_info = _score_from_statuses(
                    opt,
                    record["clean"],
                    record["statuses"],
                    oracle_scores,
                    oracle_failures,
                    gamma,
                )
                score_info_cache[cache_key] = score_info
            candidate = _candidate_from_score(record, score_info)
            if _candidate_better(candidate, best, uses_reactant_rerank, reactionfp_selection_mode):
                best = candidate
            if best is not None and best["adjusted_value"] == 0.0:
                early_stopped_remaining = len(ranked_templates) - record["rank"]
                early_stop_reason = "oracle_upper_bound"
                break

    result = _return_best_candidate(
        opt,
        best,
        topk,
        applicable,
        skipped_explored,
        product_fp_skipped,
        early_stopped_remaining,
        early_stop_reason,
    )
    if result is not None:
        return result

    return opt.rule_based_search(product_smiles, reaction_smiles)
