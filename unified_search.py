"""
Unified Search: hybrid multi-process + multi-thread parallelism.

N worker processes (each with own GIL for CPU parallelism) × M threads per
process (for LLM I/O concurrency), all consuming from a shared work queue
of (search parameters, molecule) pairs. Hard molecules only block 1 thread, not an
entire process slot.

Usage:
    python unified_search.py --dataset uspto_190 --log-dir runs/search
"""

import argparse
import gc
import os
import sys
import json
import time
import threading
import traceback
import multiprocessing as mp
import subprocess
import shutil
from datetime import datetime
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS = 2 * 60 * 60
HIGH_TOPK_NO_PROGRESS_TIMEOUT_SECONDS = 12 * 60 * 60
HEARTBEAT_MIN_INTERVAL_SECONDS = 5
DEFAULT_BLURRY_RERANK_TOPK = 20000
DEFAULT_BLURRY_RERANK_HEARTBEAT_SECONDS = 5 * 60
HIGH_TOPK_MIN_VALUE = 5000
DEFAULT_MAX_PROCESSES = 200
_PRELOADED_TRAINING_DATA = None
_PRELOADED_TEMPLATE_KEYS = None
_PRELOADED_TEMPLATE_PATTERN_FPS = None
_PRELOADED_TEMPLATE_REACTANT_PATTERN_FPS = None
_PRELOADED_INVENTORY = None


def apply_runtime_defaults():
    """Default runtime knobs for the validated high-process search path."""
    defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "AOT_FAST_TEMPLATE_LOOKUP": "1",
        "AOT_PRELOAD_TRAINING_DATA_BEFORE_FORK": "1",
        "AOT_PRELOAD_INVENTORY_BEFORE_FORK": "1",
        "AOT_SHARE_ORACLE_MODELS": "1",
        "AOT_SQLITE_INVENTORY_SKIP_COUNT": "1",
        "AOT_LAZY_STATIC_SCORES": "1",
        "AOT_FUSED_SCORE_REACTANTS": "1",
        "AOT_PARALLEL_ORACLE_SCORING": "1",
        "AOT_PARALLEL_ORACLE_WORKERS": "4",
        "AOT_PARALLEL_ORACLE_MIN_MOLECULES": "24",
        "AOT_PARALLEL_ORACLE_SAMPLE_SIZE": "4",
        "AOT_PARALLEL_ORACLE_MIN_SAMPLE_MS": "0",
        "AOT_PARALLEL_ORACLE_MAX_ROUTE_LEN": "219",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


class ProgressWriter:
    """Forward writes while sending cheap throttled heartbeats to the parent."""

    def __init__(self, wrapped, result_queue, process_id):
        self._wrapped = wrapped
        self._result_queue = result_queue
        self._process_id = process_id
        self._last_heartbeat = 0.0

    def write(self, data):
        result = self._wrapped.write(data)
        if data:
            now = time.time()
            if now - self._last_heartbeat >= HEARTBEAT_MIN_INTERVAL_SECONDS:
                self._last_heartbeat = now
                try:
                    self._result_queue.put(('heartbeat', self._process_id, now))
                except Exception:
                    pass
        return result

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def write_timeout_result(active_item, reason, timeout_seconds):
    """Write a conservative failed result so the round can finish."""
    result_file = active_item["result_file"]
    if os.path.exists(result_file):
        return False

    elapsed = max(0.0, time.time() - float(active_item.get("start_time", time.time())))
    round_max_iterations = int(active_item["round_max_iterations"])
    result = {
        "molecule_idx": int(active_item["original_idx"]),
        "original_molecule_idx": int(active_item["original_idx"]),
        "round_position": int(active_item["round_pos"]),
        "round_id": int(active_item["round_id"]),
        "round_max_iterations": round_max_iterations,
        "target_molecule": active_item["target"],
        "success": False,
        "partial_success": False,
        "search_time": elapsed,
        "iterations_completed": round_max_iterations,
        "total_and_nodes": 0,
        "solution": None,
        "timestamp": datetime.now().isoformat(),
        "process_id": active_item.get("process_id"),
        "thread_id": active_item.get("thread_id"),
        "error": reason,
        "timeout_seconds": timeout_seconds,
    }

    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    tmp_file = f"{result_file}.tmp.{os.getpid()}"
    with open(tmp_file, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_file, result_file)
    return True


# ── Helpers (used by both main and worker processes) ─────────────────────────

def config_to_dirname(cfg):
    """Convert search parameters to a stable output directory name."""
    name = f"g{cfg['gamma']}_c{cfg['c']}_r{cfg['reexp']}_i{cfg['init']}_d{cfg['depth']}_bp{cfg['bp']}"
    restarts = cfg.get('restarts', 1)
    temp = cfg.get('temp', 0.7)
    if restarts != 1:
        name += f"_k{restarts}"
    if temp != 0.7:
        name += f"_t{temp}"
    return name + "_1"


def config_key(cfg):
    """Hashable key for one search-parameter set."""
    return (cfg['gamma'], cfg['c'], cfg['reexp'], cfg['init'],
            cfg['depth'], cfg['bp'], cfg.get('restarts', 1), cfg.get('temp', 0.7))


def load_round_targets(targets_file, dataset):
    """Load the explicit target subset for one budget-pool round."""
    if not targets_file:
        sys.path.insert(0, BASE_DIR)
        from aotcore.data.loader import load_dataset_targets
        targets = load_dataset_targets(dataset)
        return [
            {"original_idx": idx, "smiles": target}
            for idx, target in enumerate(targets)
        ]

    with open(targets_file, 'r') as f:
        payload = json.load(f)

    records = payload.get("targets", payload) if isinstance(payload, dict) else payload
    round_targets = []
    for pos, item in enumerate(records):
        if isinstance(item, dict):
            if "smiles" not in item:
                raise ValueError(f"Target item {pos} missing smiles: {item}")
            original_idx = int(item.get("original_idx", item.get("idx", pos)))
            smiles = item["smiles"]
        else:
            original_idx = pos
            smiles = str(item)
        round_targets.append({"original_idx": original_idx, "smiles": smiles})

    return round_targets


def select_target_range(round_targets, start_idx=0, n_targets=None, end_idx=None):
    """Select a contiguous target range without requiring a generated JSON file."""
    start_idx = int(start_idx or 0)
    if start_idx < 0:
        raise ValueError(f"start_idx must be non-negative, got {start_idx}")
    if n_targets is not None and int(n_targets) < 0:
        raise ValueError(f"n_targets must be non-negative, got {n_targets}")
    if end_idx is not None and int(end_idx) < start_idx:
        raise ValueError(f"end_idx must be >= start_idx, got {end_idx} < {start_idx}")

    stop = int(end_idx) if end_idx is not None else len(round_targets)
    if n_targets is not None:
        stop = min(stop, start_idx + int(n_targets))
    return round_targets[start_idx:stop]


def resolve_processes_num(requested_processes_num, target_count):
    """Resolve worker process count after target selection."""
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("No targets selected")

    max_processes_num = min(target_count, DEFAULT_MAX_PROCESSES)
    if requested_processes_num is None:
        processes_num = max_processes_num
        return processes_num

    requested_processes_num = int(requested_processes_num)
    if requested_processes_num <= 0:
        raise ValueError(f"processes must be positive, got {requested_processes_num}")
    processes_num = min(requested_processes_num, max_processes_num)
    return processes_num


def default_run_name(args):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    timestamp = timestamp.replace(":", "").replace("+", "p")
    count = args.n_targets if args.n_targets is not None else "all"
    cap = getattr(args, "max_round_cap", args.max_iterations)
    return (
        f"{args.dataset}_top{args.blurry_rerank_topk}_n{count}_"
        f"i{args.max_iterations}_cap{cap}_{args.n_processes}x{args.n_threads}_{timestamp}"
    )


def build_run_spec(args):
    """Build one search run directly from command-line arguments."""
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = os.path.join(BASE_DIR, "runs", "search", args.run_name or default_run_name(args))
    elif args.run_name:
        log_dir = os.path.join(log_dir, args.run_name)

    timeout = (
        int(args.no_progress_timeout_seconds)
        if args.no_progress_timeout_seconds is not None
        else DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    )
    if int(args.blurry_rerank_topk) >= HIGH_TOPK_MIN_VALUE:
        timeout = max(timeout, HIGH_TOPK_NO_PROGRESS_TIMEOUT_SECONDS)

    return {
        'configs': [{
            'gamma': args.gamma,
            'c': args.c,
            'reexp': args.max_reexpansions,
            'init': args.num_initial_routes,
            'depth': args.max_depth,
            'bp': args.backprop_method,
            **({'restarts': args.max_restarts} if args.max_restarts != 1 else {}),
            **({'temp': args.api_temperature} if args.api_temperature != 0.7 else {}),
        }],
        'n_processes': args.n_processes,
        'n_threads': args.n_threads,
        'dataset': args.dataset,
        'log_dir': log_dir,
        'targets_file': args.targets_file,
        'target_start_idx': args.start_idx,
        'target_count': args.n_targets,
        'target_end_idx': args.end_idx,
        'round_id': args.round_id,
        'round_max_iterations': args.max_iterations,
        'no_progress_timeout_seconds': timeout,
        'blurry_rerank_topk': args.blurry_rerank_topk,
        'blurry_rerank_heartbeat_seconds': args.blurry_rerank_heartbeat_seconds,
        'no_blurry_tversky': not args.product_fp_rerank,
        'reactionfp_rerank_mode': args.reactionfp_rerank_mode,
        'reactionfp_candidate_pool': args.reactionfp_candidate_pool,
        'reactionfp_selection_mode': args.reactionfp_selection_mode,
        'use_local_cache': not args.disable_local_cache,
        'local_cache_path': args.local_cache_path,
    }


def count_local_cache_reactions(node):
    """Count solution-tree reactions restored from the local solved cache."""
    if isinstance(node, dict):
        count = 0
        reaction_id = node.get("reaction_id")
        if isinstance(reaction_id, str) and reaction_id.startswith("local_cache::"):
            count += 1
        for value in node.values():
            count += count_local_cache_reactions(value)
        return count
    if isinstance(node, list):
        return sum(count_local_cache_reactions(item) for item in node)
    return 0


def count_api_keys(*values):
    count = 0
    for value in values:
        if not value:
            continue
        count += sum(1 for key in str(value).split(",") if key.strip())
    return count


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_file = f"{path}.tmp.{os.getpid()}"
    with open(tmp_file, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_file, path)


def budget_state_path(root_log_dir):
    return os.path.join(root_log_dir, "budget_state.json")


def write_budget_targets(path, round_id, targets):
    write_json_atomic(path, {"round": round_id, "targets": targets})


def budget_result_dir(round_log_dir, cfg, dataset, model_name):
    return os.path.join(round_log_dir, config_to_dirname(cfg), dataset, model_name)


def local_cache_path_for_result_dir(result_dir):
    return os.path.join(result_dir, "local_solved_cache.db")


def write_budget_search_config(final_log_dir, cli_args, cfg, total_molecules, max_iterations, max_round_cap):
    os.makedirs(final_log_dir, exist_ok=True)
    max_restarts = cfg.get('restarts', 1)
    suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
    payload = {
        "dataset": cli_args.dataset,
        "total_molecules": total_molecules,
        "model_name": cli_args.api_model,
        "llm_base_url": cli_args.llm_base_url,
        "llm_timeout": cli_args.llm_timeout,
        "llm_max_tokens": cli_args.api_max_tokens,
        "llm_verify_ssl": cli_args.llm_verify_ssl,
        "llm_trust_env": cli_args.llm_trust_env,
        "llm_api_key_count": count_api_keys(cli_args.llm_api_key, cli_args.llm_api_keys),
        "api_temperature": cli_args.api_temperature,
        "max_oracle_calls": cli_args.max_oracle_calls,
        "num_initial_routes": cfg['init'],
        "max_iterations": max_iterations,
        "max_round_cap": max_round_cap,
        "max_depth": cfg['depth'],
        "backprop_method": cfg['bp'],
        "unsolved_penalty": cfg['gamma'],
        "c_param": cfg['c'],
        "max_reexpansions": cfg['reexp'],
        "max_restarts": max_restarts,
        "use_presort": not cli_args.no_presort,
        "use_tversky_cutoff": not cli_args.no_presort,
        "blurry_tversky": bool(cli_args.product_fp_rerank),
        "blurry_oracle_rerank": True,
        "blurry_rerank_topk": cli_args.blurry_rerank_topk,
        "blurry_rerank_heartbeat_seconds": cli_args.blurry_rerank_heartbeat_seconds,
        "reactionfp_rerank_mode": cli_args.reactionfp_rerank_mode,
        "reactionfp_candidate_pool": cli_args.reactionfp_candidate_pool,
        "reactionfp_selection_mode": cli_args.reactionfp_selection_mode,
        "use_local_cache": not cli_args.disable_local_cache,
        "local_cache_path": cli_args.local_cache_path,
        "runner": "unified_search",
    }
    write_json_atomic(os.path.join(final_log_dir, f"search_config{suffix}.json"), payload)


def merge_budget_round_outputs(round_log_dir, final_log_dir, cfg, dataset, model_name, targets):
    max_restarts = cfg.get('restarts', 1)
    suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
    source_dir = budget_result_dir(round_log_dir, cfg, dataset, model_name)
    os.makedirs(final_log_dir, exist_ok=True)
    merged = []
    for target in targets:
        original_idx = int(target["original_idx"])
        result_name = f"result_{original_idx:05d}{suffix}.json"
        source_result = os.path.join(source_dir, result_name)
        if os.path.exists(source_result):
            shutil.copy2(source_result, os.path.join(final_log_dir, result_name))
            merged.append(original_idx)
        log_name = f"search_log_{original_idx:05d}{suffix}.json"
        source_log = os.path.join(source_dir, log_name)
        if os.path.exists(source_log):
            shutil.copy2(source_log, os.path.join(final_log_dir, log_name))
    return merged


def collect_budget_round(round_log_dir, cfg, dataset, model_name, targets, round_cap):
    max_restarts = cfg.get('restarts', 1)
    suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
    log_dir = budget_result_dir(round_log_dir, cfg, dataset, model_name)
    results = []
    missing = []
    for target in targets:
        original_idx = int(target["original_idx"])
        result_file = os.path.join(log_dir, f"result_{original_idx:05d}{suffix}.json")
        if not os.path.exists(result_file):
            missing.append(original_idx)
            continue
        with open(result_file, "r") as f:
            results.append(json.load(f))

    succeeded = sorted(
        int(r.get("original_molecule_idx", r.get("molecule_idx")))
        for r in results
        if r.get("success") is True
    )
    failed = sorted(
        int(r.get("original_molecule_idx", r.get("molecule_idx")))
        for r in results
        if r.get("success") is not True
    )
    consumed = sum(int(r.get("iterations_completed", 0) or 0) for r in results)
    return {
        "target_count": len(targets),
        "completed": len(results),
        "missing": missing,
        "success": len(succeeded),
        "failed": len(failed),
        "succeeded_indices": succeeded,
        "failed_indices": failed,
        "consumed_iterations": consumed,
        "round_max_iterations": int(round_cap),
        "log_dir": log_dir,
    }


def write_final_budget_outputs(final_log_dir, cfg, dataset, model_name, targets, max_iterations, max_round_cap, rounds):
    max_restarts = cfg.get('restarts', 1)
    suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
    results = []
    missing = []
    for target in targets:
        original_idx = int(target["original_idx"])
        result_file = os.path.join(final_log_dir, f"result_{original_idx:05d}{suffix}.json")
        if not os.path.exists(result_file):
            missing.append(original_idx)
            continue
        with open(result_file, "r") as f:
            results.append(json.load(f))
    if missing:
        return {"missing": missing}
    successful = sum(1 for r in results if r.get("success") is True)
    total_search_time = sum(float(r.get("search_time", 0) or 0) for r in results)
    summary = {
        "dataset": f"{dataset}/{model_name}",
        "total_molecules": len(targets),
        "successful": successful,
        "failed": len(targets) - successful,
        "success_rate": successful / len(targets) * 100 if targets else 0.0,
        "total_time": total_search_time,
        "avg_time_per_molecule": total_search_time / len(targets) if targets else 0.0,
        "end_time": datetime.now().isoformat(),
        "config": cfg,
        "max_iterations": int(max_iterations),
        "max_round_cap": int(max_round_cap),
        "budget_round_count": len(rounds),
        "consumed_iterations": sum(int(r.get("consumed_iterations", 0) or 0) for r in rounds),
    }
    write_json_atomic(os.path.join(final_log_dir, f"summary{suffix}.json"), summary)
    write_json_atomic(os.path.join(final_log_dir, f"all_results{suffix}.json"), results)
    return summary


def single_round_command(cli_args, targets_file, round_log_dir, round_id, round_cap):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single-round",
        "--dataset", str(cli_args.dataset),
        "--targets-file", str(targets_file),
        "--log-dir", str(round_log_dir),
        "--round-id", str(round_id),
        "--max-iter", str(round_cap),
        "--processes", str(cli_args.n_processes),
        "--threads", str(cli_args.n_threads),
        "--topk", str(cli_args.blurry_rerank_topk),
        "--blurry-rerank-heartbeat-seconds", str(cli_args.blurry_rerank_heartbeat_seconds),
        "--gamma", str(cli_args.gamma),
        "--c", str(cli_args.c),
        "--max-reexpansions", str(cli_args.max_reexpansions),
        "--num-initial-routes", str(cli_args.num_initial_routes),
        "--max-depth", str(cli_args.max_depth),
        "--backprop-method", str(cli_args.backprop_method),
        "--max-restarts", str(cli_args.max_restarts),
        "--llm-model", str(cli_args.api_model),
        "--llm-temperature", str(cli_args.api_temperature),
        "--llm-timeout", str(cli_args.llm_timeout),
        "--llm-max-tokens", str(cli_args.api_max_tokens),
        "--reactionfp-rerank-mode", str(cli_args.reactionfp_rerank_mode),
        "--reactionfp-candidate-pool", str(cli_args.reactionfp_candidate_pool),
        "--reactionfp-selection-mode", str(cli_args.reactionfp_selection_mode),
        "--max-oracle-calls", str(cli_args.max_oracle_calls),
    ]
    if cli_args.llm_base_url:
        cmd.extend(["--llm-base-url", str(cli_args.llm_base_url)])
    if cli_args.llm_api_key:
        cmd.extend(["--llm-api-key", str(cli_args.llm_api_key)])
    if cli_args.llm_api_keys:
        cmd.extend(["--llm-api-keys", str(cli_args.llm_api_keys)])
    if not cli_args.llm_verify_ssl:
        cmd.append("--no-llm-verify-ssl")
    if cli_args.llm_trust_env:
        cmd.append("--llm-trust-env")
    if cli_args.disable_local_cache:
        cmd.append("--disable-local-cache")
    if cli_args.local_cache_path:
        cmd.extend(["--local-cache-path", str(cli_args.local_cache_path)])
    if cli_args.no_progress_timeout_seconds is not None:
        cmd.extend(["--no-progress-timeout-seconds", str(cli_args.no_progress_timeout_seconds)])
    if cli_args.no_presort:
        cmd.append("--no-presort")
    if cli_args.product_fp_rerank:
        cmd.append("--product-fp-rerank")
    if cli_args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_budgeted_search(cli_args):
    sys.path.insert(0, BASE_DIR)
    os.chdir(BASE_DIR)

    round_targets = load_round_targets(cli_args.targets_file, cli_args.dataset)
    round_targets = select_target_range(
        round_targets,
        cli_args.start_idx,
        cli_args.n_targets,
        cli_args.end_idx,
    )
    if not round_targets:
        raise ValueError("No targets selected")

    requested_processes_num = cli_args.n_processes
    cli_args.n_processes = resolve_processes_num(requested_processes_num, len(round_targets))
    run_spec = build_run_spec(cli_args)
    root_log_dir = run_spec["log_dir"]
    max_iterations = int(cli_args.max_iterations)
    max_round_cap = int(cli_args.max_round_cap)
    total_budget = len(round_targets) * max_iterations
    cfg = run_spec["configs"][0]
    final_log_dir = budget_result_dir(root_log_dir, cfg, cli_args.dataset, cli_args.api_model)
    if not cli_args.local_cache_path:
        cli_args.local_cache_path = local_cache_path_for_result_dir(final_log_dir)
    write_budget_search_config(
        final_log_dir,
        cli_args,
        cfg,
        len(round_targets),
        max_iterations,
        max_round_cap,
    )
    by_idx = {int(t["original_idx"]): t for t in round_targets}
    active_targets = list(round_targets)
    succeeded_indices = []
    consumed_iterations = 0
    rounds = []
    round_id = int(cli_args.round_id)
    state = {
        "dataset": cli_args.dataset,
        "target_count": len(round_targets),
        "max_iterations": max_iterations,
        "max_round_cap": max_round_cap,
        "total_budget": total_budget,
        "consumed_iterations": consumed_iterations,
        "remaining_budget": total_budget,
        "succeeded_indices": succeeded_indices,
        "failed_indices": [int(t["original_idx"]) for t in active_targets],
        "rounds": rounds,
        "status": "running",
        "updated_at": datetime.now().isoformat(),
    }
    write_json_atomic(budget_state_path(root_log_dir), state)

    while active_targets:
        remaining_budget = total_budget - consumed_iterations
        if remaining_budget < len(active_targets):
            state["status"] = "budget_exhausted"
            state["remaining_budget"] = remaining_budget
            state["updated_at"] = datetime.now().isoformat()
            write_json_atomic(budget_state_path(root_log_dir), state)
            break

        round_cap = min(max_round_cap, max(1, remaining_budget // len(active_targets)))
        round_dir = os.path.join(root_log_dir, ".budget_rounds", f"round_{round_id:02d}")
        round_log_dir = os.path.join(round_dir, "logs")
        targets_file = os.path.join(round_dir, "targets.json")
        write_budget_targets(targets_file, round_id, active_targets)

        cli_args.n_processes = resolve_processes_num(requested_processes_num, len(active_targets))
        cmd = single_round_command(cli_args, targets_file, round_log_dir, round_id, round_cap)
        print(f"\nBudget round {round_id}: targets={len(active_targets)} cap={round_cap}")
        try:
            subprocess.run(cmd, cwd=BASE_DIR, check=True)
        except subprocess.CalledProcessError as exc:
            state["status"] = "round_failed"
            state["failed_round"] = round_id
            state["returncode"] = exc.returncode
            state["remaining_budget"] = total_budget - consumed_iterations
            state["updated_at"] = datetime.now().isoformat()
            write_json_atomic(budget_state_path(root_log_dir), state)
            raise
        if cli_args.dry_run:
            state["status"] = "dry_run"
            state["updated_at"] = datetime.now().isoformat()
            write_json_atomic(budget_state_path(root_log_dir), state)
            return

        summary = collect_budget_round(
            round_log_dir,
            cfg,
            cli_args.dataset,
            cli_args.api_model,
            active_targets,
            round_cap,
        )
        summary["round"] = round_id
        rounds.append(summary)
        merge_budget_round_outputs(
            round_log_dir,
            final_log_dir,
            cfg,
            cli_args.dataset,
            cli_args.api_model,
            active_targets,
        )

        if summary["missing"]:
            state["status"] = "round_incomplete"
            state["remaining_budget"] = total_budget - consumed_iterations
            state["updated_at"] = datetime.now().isoformat()
            write_json_atomic(budget_state_path(root_log_dir), state)
            raise RuntimeError(f"round {round_id} missing results: {summary['missing'][:20]}")

        consumed_iterations += int(summary["consumed_iterations"])
        succeeded_indices = sorted(set(succeeded_indices) | set(summary["succeeded_indices"]))
        active_targets = [by_idx[int(idx)] for idx in summary["failed_indices"]]
        state.update({
            "consumed_iterations": consumed_iterations,
            "remaining_budget": total_budget - consumed_iterations,
            "succeeded_indices": succeeded_indices,
            "failed_indices": [int(t["original_idx"]) for t in active_targets],
            "rounds": rounds,
            "status": "running",
            "updated_at": datetime.now().isoformat(),
        })
        write_json_atomic(budget_state_path(root_log_dir), state)
        round_id += 1

    if not active_targets:
        state["status"] = "all_solved"
    elif state.get("status") == "running":
        state["status"] = "budget_exhausted"
    final_summary = write_final_budget_outputs(
        final_log_dir,
        cfg,
        cli_args.dataset,
        cli_args.api_model,
        round_targets,
        max_iterations,
        max_round_cap,
        rounds,
    )
    if final_summary.get("missing"):
        state["final_missing_indices"] = final_summary["missing"]
    state["updated_at"] = datetime.now().isoformat()
    write_json_atomic(budget_state_path(root_log_dir), state)


# ── Worker process ───────────────────────────────────────────────────────────

def worker_process(process_id, n_threads, work_queue, result_queue, base_args_dict):
    """Worker process: loads data, runs thread pool, pulls from shared queue."""
    sys.path.insert(0, BASE_DIR)
    os.chdir(BASE_DIR)
    sys.stdout = ProgressWriter(sys.stdout, result_queue, process_id)
    sys.stderr = ProgressWriter(sys.stderr, result_queue, process_id)

    from aotcore.data.loader import load_inventory, load_local_solved_cache, load_training_data
    from aotcore.llm_tree_optimizer import LLMGuidedTreeOptimizer
    from aotcore.oracle_rerank import (
        ensure_parallel_oracle_pool,
        env_int,
        parallel_oracle_enabled,
        parallel_oracle_ping,
        parallel_oracle_worker_count,
        shutdown_parallel_oracle_pool,
    )

    base_args = SimpleNamespace(**base_args_dict)

    print(f"[Process-{process_id}] Loading data...")
    t0 = time.time()

    global _PRELOADED_INVENTORY
    if _PRELOADED_INVENTORY is not None:
        inventory = _PRELOADED_INVENTORY
        print(f"[Process-{process_id}] Using fork-shared RAM inventory")
    else:
        inventory = load_inventory('./dataset/inventory.pkl')
    use_local_cache = bool(getattr(base_args, "use_local_cache", True))
    local_solved_cache = (
        load_local_solved_cache(base_args.local_cache_path)
        if use_local_cache
        else None
    )
    print(f"[Process-{process_id}] Local solved cache: {'enabled' if use_local_cache else 'disabled'}")
    if use_local_cache:
        print(f"[Process-{process_id}] Local solved cache path: {base_args.local_cache_path}")
    global _PRELOADED_TRAINING_DATA
    global _PRELOADED_TEMPLATE_KEYS
    global _PRELOADED_TEMPLATE_PATTERN_FPS
    global _PRELOADED_TEMPLATE_REACTANT_PATTERN_FPS
    if _PRELOADED_TRAINING_DATA is None:
        (route_list, all_fps, reaction_list, all_reaction_fps,
         datasub, template_dict) = load_training_data(use_ram_templates=True)
    else:
        print(f"[Process-{process_id}] Using fork-preloaded training data")
        (route_list, all_fps, reaction_list, all_reaction_fps,
         datasub, template_dict) = _PRELOADED_TRAINING_DATA

    print(f"[Process-{process_id}] Creating prototype optimizer...")
    prototype_args = base_args
    if _PRELOADED_TEMPLATE_PATTERN_FPS is not None:
        prototype_args = SimpleNamespace(**vars(base_args))
        prototype_args.use_presort = False
        prototype_args.no_presort = True
        prototype_args.blurry_tversky = False
        prototype_args.no_blurry_tversky = True
    prototype = LLMGuidedTreeOptimizer(
        prototype_args, inventory, template_dict,
        reaction_list, all_reaction_fps, datasub, local_solved_cache
    )
    shared_template_keys = _PRELOADED_TEMPLATE_KEYS or prototype.template_keys
    shared_template_pattern_fps = _PRELOADED_TEMPLATE_PATTERN_FPS or prototype.template_pattern_fps
    shared_template_reactant_pattern_fps = _PRELOADED_TEMPLATE_REACTANT_PATTERN_FPS
    reactionfp_rerank_mode = str(getattr(base_args, "reactionfp_rerank_mode", "tanimoto"))
    needs_product_side_pattern_fps = (
        not base_args.blurry_tversky
        and reactionfp_rerank_mode == "product_tversky_reactants"
    )
    if needs_product_side_pattern_fps and shared_template_pattern_fps is None:
        t_pat = time.time()
        shared_template_pattern_fps = prototype._build_template_pattern_fps()
        prototype.template_pattern_fps = shared_template_pattern_fps
        print(
            f"[Process-{process_id}] Built product-side PatternFPs for "
            f"reactionFP v3 in {time.time()-t_pat:.1f}s"
        )
    share_pattern_fps = (
        (base_args.use_presort or base_args.blurry_tversky or needs_product_side_pattern_fps)
        and shared_template_pattern_fps is not None
    )
    share_reactant_pattern_fps = (
        not base_args.blurry_tversky
        and reactionfp_rerank_mode in {"tversky_reactants", "product_tversky_reactants"}
    )
    if share_pattern_fps:
        print(
            f"[Process-{process_id}] Sharing "
            f"{len(shared_template_pattern_fps)} PatternFPs across threads"
        )
    if share_reactant_pattern_fps and shared_template_reactant_pattern_fps is None:
        t_react = time.time()
        shared_template_reactant_pattern_fps = prototype._build_template_reactant_pattern_fps()
        print(
            f"[Process-{process_id}] Sharing "
            f"{len(shared_template_reactant_pattern_fps)} reactant PatternFPs across threads "
            f"in {time.time()-t_react:.1f}s"
        )
    if parallel_oracle_enabled():
        parallel_workers = parallel_oracle_worker_count()
        if parallel_workers > 1:
            cache_size = env_int("AOT_ORACLE_SCORE_CACHE_SIZE", 50000)
            pool = ensure_parallel_oracle_pool(
                getattr(base_args, "max_oracle_calls", 10000),
                getattr(base_args, "freq_log", 1000000),
                cache_size,
                workers=parallel_workers,
            )
            if pool is not None:
                try:
                    pids = sorted(set(pool.map(parallel_oracle_ping, range(parallel_workers), chunksize=1)))
                    print(
                        f"[Process-{process_id}] Parallel oracle pool ready: "
                        f"workers={parallel_workers} pids={pids}"
                    )
                except Exception as exc:
                    print(f"[Process-{process_id}] Parallel oracle pool warmup failed: {exc}")
    print(f"[Process-{process_id}] Ready in {time.time()-t0:.0f}s")

    # Thread-local optimizer storage
    thread_local = threading.local()

    def get_thread_optimizer():
        """Create per-thread optimizer: own Oracle, shared heavy data."""
        if not hasattr(thread_local, 'optimizer'):
            init_args = base_args
            if share_pattern_fps:
                init_args = SimpleNamespace(**vars(base_args))
                init_args.use_presort = False
                init_args.no_presort = True
                init_args.blurry_tversky = False
                init_args.no_blurry_tversky = True
            opt = LLMGuidedTreeOptimizer(
                init_args, inventory, template_dict,
                reaction_list, all_reaction_fps, datasub, local_solved_cache
            )
            if share_pattern_fps:
                opt.args = base_args
                opt.use_presort = base_args.use_presort
                opt.blurry_tversky = base_args.blurry_tversky
                opt.template_keys = shared_template_keys
                opt.template_pattern_fps = shared_template_pattern_fps
            if share_reactant_pattern_fps:
                opt.args = base_args
                opt.template_reactant_pattern_fps = shared_template_reactant_pattern_fps
            thread_local.optimizer = opt
            print(f"[Process-{process_id}/Thread-{threading.current_thread().ident}] Optimizer initialized")
        return thread_local.optimizer

    def thread_worker():
        """Thread loop: pull work items from queue until poison pill."""
        while True:
            try:
                item = work_queue.get(timeout=5)
            except Exception:
                # Queue empty or timeout, check if we should stop
                if work_queue.empty():
                    break
                continue

            if item is None:
                # Poison pill
                break

            cfg = item['config']
            original_idx = item['original_idx']
            round_pos = item['round_pos']
            target = item['target']
            total_molecules = item['total_molecules']
            log_dir = item['log_dir']
            round_id = item['round_id']
            round_max_iterations = item['round_max_iterations']

            max_restarts = cfg.get('restarts', 1)
            suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
            result_file = os.path.join(log_dir, f"result_{original_idx:05d}{suffix}.json")

            # Skip if already done (race condition safety)
            if os.path.exists(result_file):
                result_queue.put(('skip', cfg, original_idx))
                continue

            optimizer = get_thread_optimizer()
            optimizer.clear_cache()

            # Set search params for this config
            optimizer.api_temperature = cfg.get('temp', 0.7)
            optimizer.max_depth = cfg['depth']
            optimizer.num_initial_routes = cfg['init']
            optimizer.backprop_method = cfg['bp']
            optimizer.unsolved_penalty = cfg['gamma']
            optimizer.c_param = cfg['c']
            optimizer.max_reexpansions = cfg['reexp']

            dirname = config_to_dirname(cfg)
            tid = threading.current_thread().ident
            start_time = time.time()
            active_item = {
                "process_id": process_id,
                "thread_id": tid,
                "original_idx": original_idx,
                "round_pos": round_pos,
                "round_id": round_id,
                "round_max_iterations": round_max_iterations,
                "target": target,
                "result_file": result_file,
                "start_time": start_time,
            }
            result_queue.put(('start', process_id, active_item))

            print(
                f"[P{process_id}/T{tid}] [{dirname}] "
                f"round {round_id} target {round_pos+1}/{total_molecules} "
                f"(orig {original_idx}): {target[:50]}..."
            )

            try:
                search_config = {
                    "max_iterations": round_max_iterations,
                    "max_depth": cfg['depth'],
                    "max_oracle_calls": base_args.max_oracle_calls,
                    "max_restarts": max_restarts,
                }

                solution = optimizer._optimize(target, route_list, all_fps, search_config)

                search_time = time.time() - start_time
                is_complete = solution is not None and solution.get("type") != "partial_solution"
                is_partial = solution is not None and solution.get("type") == "partial_solution"
                local_cache_hits = count_local_cache_reactions(solution)

                search_log = getattr(optimizer, 'search_log', None)

                result = {
                    "molecule_idx": original_idx,
                    "original_molecule_idx": original_idx,
                    "round_position": round_pos,
                    "round_id": round_id,
                    "round_max_iterations": round_max_iterations,
                    "target_molecule": target,
                    "success": is_complete,
                    "partial_success": is_partial,
                    "search_time": search_time,
                    "iterations_completed": getattr(optimizer, 'current_iteration', 0),
                    "total_and_nodes": optimizer.total_and_nodes,
                    "solution": solution,
                    "timestamp": datetime.now().isoformat(),
                    "process_id": process_id,
                    "thread_id": tid,
                    "local_cache_reaction_hits": local_cache_hits,
                    "used_local_cache": local_cache_hits > 0,
                }

                if search_log:
                    result["search_log_summary"] = {
                        "total_dropped_steps": sum(e.get("dropped_steps", 0) for e in search_log),
                        "total_valid_steps": sum(e.get("valid_steps", 0) for e in search_log),
                        "root_reexpansion_count": sum(1 for e in search_log if e.get("action") == "root_reexpansion"),
                    }

                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

                if search_log:
                    log_file = os.path.join(log_dir, f"search_log_{original_idx:05d}{suffix}.json")
                    with open(log_file, 'w') as f:
                        json.dump(search_log, f, indent=2)

                status = "SUCCESS" if is_complete else ("PARTIAL" if is_partial else "FAILED")
                print(f"[P{process_id}/T{tid}] [{dirname}] orig {original_idx} {status} "
                      f"({search_time:.0f}s, {optimizer.total_and_nodes} nodes)")

                result_queue.put(('done', cfg, original_idx, is_complete))
                result_queue.put(('finish', process_id, original_idx))

            except Exception as e:
                search_time = time.time() - start_time
                print(f"[P{process_id}/T{tid}] [{dirname}] orig {original_idx} ERROR: {e}")

                result = {
                    "molecule_idx": original_idx,
                    "original_molecule_idx": original_idx,
                    "round_position": round_pos,
                    "round_id": round_id,
                    "round_max_iterations": round_max_iterations,
                    "target_molecule": target,
                    "success": False,
                    "search_time": search_time,
                    "error": str(e),
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                    "process_id": process_id,
                    "thread_id": tid,
                }

                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

                result_queue.put(('error', cfg, original_idx))
                result_queue.put(('finish', process_id, original_idx))

    # Run thread pool
    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=thread_worker, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"[Process-{process_id}] All threads finished.")
    shutdown_parallel_oracle_pool()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    apply_runtime_defaults()

    parser = argparse.ArgumentParser(description='Unified hybrid pool search')
    parser.add_argument('--dataset', type=str, default='uspto_190')
    parser.add_argument('--start-idx', dest='start_idx', type=int, default=0)
    parser.add_argument('--n-targets', dest='n_targets', type=int, default=None)
    parser.add_argument('--end-idx', dest='end_idx', type=int, default=None)
    parser.add_argument('--targets-file', dest='targets_file', type=str, default=None)
    parser.add_argument('--log-dir', dest='log_dir', metavar='LOG_DIR', type=str, default=None)
    parser.add_argument('--run-name', dest='run_name', type=str, default=None)
    parser.add_argument('--processes', dest='n_processes', type=int, default=None)
    parser.add_argument('--threads', dest='n_threads', type=int, default=1)
    parser.add_argument('--max-iter', dest='max_iterations', type=int, default=100)
    parser.add_argument('--max-round-cap', dest='max_round_cap', type=int, default=50)
    parser.add_argument('--round-id', dest='round_id', type=int, default=1)
    parser.add_argument('--topk', '--blurry-rerank-topk',
                        dest='blurry_rerank_topk', type=int, default=DEFAULT_BLURRY_RERANK_TOPK)
    parser.add_argument('--blurry-rerank-heartbeat-seconds',
                        dest='blurry_rerank_heartbeat_seconds', type=float,
                        default=DEFAULT_BLURRY_RERANK_HEARTBEAT_SECONDS)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--c', type=float, default=0.5)
    parser.add_argument('--max-reexpansions', dest='max_reexpansions', type=int, default=5)
    parser.add_argument('--num-initial-routes', dest='num_initial_routes', type=int, default=3)
    parser.add_argument('--max-depth', dest='max_depth', type=int, default=20)
    parser.add_argument('--backprop-method',
                        dest='backprop_method', choices=['average', 'andor'], default='average')
    parser.add_argument('--max-restarts', dest='max_restarts', type=int, default=1)
    parser.add_argument('--llm-model', dest='api_model', metavar='MODEL', type=str,
                        default="llm-model")
    parser.add_argument('--llm-temperature', dest='api_temperature',
                        metavar='TEMPERATURE', type=float, default=0.7)
    parser.add_argument('--llm-base-url', dest='llm_base_url', type=str, default=None)
    parser.add_argument('--llm-api-key', dest='llm_api_key', type=str, default=None)
    parser.add_argument('--llm-api-keys', dest='llm_api_keys', type=str, default=None,
                        help='Comma-separated API keys for key rotation')
    parser.add_argument('--llm-timeout', dest='llm_timeout', type=int, default=1200)
    parser.add_argument('--llm-max-tokens', dest='api_max_tokens', type=int, default=4096)
    parser.add_argument('--llm-verify-ssl', dest='llm_verify_ssl', action='store_true', default=True)
    parser.add_argument('--no-llm-verify-ssl', dest='llm_verify_ssl', action='store_false')
    parser.add_argument('--llm-trust-env', dest='llm_trust_env', action='store_true', default=False)
    parser.add_argument('--no-llm-trust-env', dest='llm_trust_env', action='store_false')
    parser.add_argument('--reactionfp-rerank-mode',
                        dest='reactionfp_rerank_mode', default='product_tversky_reactants',
                        choices=['tanimoto', 'tversky', 'tversky_reactants', 'product_tversky_reactants'])
    parser.add_argument('--reactionfp-candidate-pool',
                        dest='reactionfp_candidate_pool', type=int, default=0)
    parser.add_argument('--reactionfp-selection-mode',
                        dest='reactionfp_selection_mode', default='oracle_first',
                        choices=['path_then_oracle', 'oracle', 'oracle_first'])
    parser.add_argument('--disable-local-cache', dest='disable_local_cache', action='store_true')
    parser.add_argument('--local-cache-path', dest='local_cache_path', type=str, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        help='Validate arguments, select targets, and write search_config.json without starting workers')
    parser.add_argument('--no-progress-timeout-seconds',
                        dest='no_progress_timeout_seconds', type=int, default=None)
    parser.add_argument('--max-oracle-calls', dest='max_oracle_calls', type=int, default=1000)
    parser.add_argument('--no-presort', dest='no_presort', action='store_true', default=False)
    parser.add_argument('--product-fp-rerank', dest='product_fp_rerank', action='store_true',
                        help='Use the older product-FP rerank path instead of reactionFP oracle-first')
    parser.add_argument('--single-round', dest='single_round', action='store_true', help=argparse.SUPPRESS)
    cli_args = parser.parse_args()

    if int(cli_args.max_iterations) <= 0:
        raise ValueError(f"max_iterations must be positive, got {cli_args.max_iterations}")
    if int(cli_args.max_round_cap) <= 0:
        raise ValueError(f"max_round_cap must be positive, got {cli_args.max_round_cap}")

    cli_args.use_presort = not cli_args.no_presort
    cli_args.blurry_tversky = bool(cli_args.product_fp_rerank)
    cli_args.no_blurry_tversky = not cli_args.blurry_tversky

    if not cli_args.single_round and int(cli_args.max_round_cap) < int(cli_args.max_iterations):
        run_budgeted_search(cli_args)
        return

    print("Loading search arguments from CLI")
    sys.path.insert(0, BASE_DIR)
    os.chdir(BASE_DIR)
    print(f"\nLoading dataset: {cli_args.dataset}")
    round_targets = load_round_targets(cli_args.targets_file, cli_args.dataset)
    round_targets = select_target_range(
        round_targets,
        cli_args.start_idx,
        cli_args.n_targets,
        cli_args.end_idx,
    )
    total_molecules = len(round_targets)
    print(f"  {total_molecules} round targets")
    cli_args.n_processes = resolve_processes_num(cli_args.n_processes, total_molecules)

    run_spec = build_run_spec(cli_args)
    configs = run_spec['configs']
    n_processes = run_spec['n_processes']
    n_threads = run_spec['n_threads']
    dataset = run_spec['dataset']
    root_log_dir = run_spec['log_dir']
    targets_file = run_spec['targets_file']
    round_id = int(run_spec['round_id'])
    round_max_iterations = int(run_spec['round_max_iterations'])
    primary_result_dir = budget_result_dir(root_log_dir, configs[0], dataset, cli_args.api_model)
    local_cache_path = run_spec.get('local_cache_path') or local_cache_path_for_result_dir(primary_result_dir)
    if round_max_iterations <= 0:
        raise ValueError(f"ROUND_MAX_ITERATIONS must be positive, got {round_max_iterations}")
    no_progress_timeout = int(run_spec['no_progress_timeout_seconds'])
    if no_progress_timeout <= 0:
        raise ValueError(f"NO_PROGRESS_TIMEOUT_SECONDS must be positive, got {no_progress_timeout}")

    print(f"  {len(configs)} search parameter set(s)")
    print(f"  {n_processes} processes × {n_threads} threads = {n_processes * n_threads} total concurrency")
    print(f"  Dataset: {dataset}, Log dir: {root_log_dir}")
    print(f"  Round: {round_id}, max iterations: {round_max_iterations}")
    print(f"  No-progress timeout: {no_progress_timeout}s")
    blurry_rerank_topk = int(run_spec['blurry_rerank_topk'])
    blurry_rerank_heartbeat_seconds = float(run_spec['blurry_rerank_heartbeat_seconds'])
    reactionfp_rerank_mode = str(run_spec.get('reactionfp_rerank_mode', 'product_tversky_reactants'))
    reactionfp_candidate_pool = int(run_spec.get('reactionfp_candidate_pool', 0) or 0)
    reactionfp_selection_mode = str(run_spec.get('reactionfp_selection_mode', 'oracle_first'))
    if bool(run_spec.get('no_blurry_tversky', False)):
        cli_args.no_blurry_tversky = True
        cli_args.blurry_tversky = False
    blurry_rerank_mode = "product_fp_tversky" if cli_args.blurry_tversky else f"reaction_fp_{reactionfp_rerank_mode}"
    print(f"  Blurry oracle rerank topK: {blurry_rerank_topk}")
    print(f"  Blurry oracle rerank mode: {blurry_rerank_mode}")
    if not cli_args.blurry_tversky and reactionfp_rerank_mode == "product_tversky_reactants":
        print("  ReactionFP product-side filter: Tversky score >= 1.0")
    elif not cli_args.blurry_tversky and reactionfp_candidate_pool > 0:
        print(f"  ReactionFP candidate pool: {reactionfp_candidate_pool}")
    if not cli_args.blurry_tversky:
        print(f"  ReactionFP selection mode: {reactionfp_selection_mode}")
    print(f"  Blurry oracle rerank heartbeat: {blurry_rerank_heartbeat_seconds:.0f}s")
    print(f"  Local solved cache: {'enabled' if bool(run_spec.get('use_local_cache', True)) else 'disabled'}")
    if bool(run_spec.get('use_local_cache', True)):
        print(f"  Local solved cache path: {local_cache_path}")
    print(f"  LLM model: {cli_args.api_model}")
    if cli_args.llm_base_url:
        print(f"  LLM base URL: {cli_args.llm_base_url}")
    if targets_file:
        print(f"  Targets file: {targets_file}")

    # base_args as dict (serializable for multiprocessing)
    base_args_dict = dict(
        dataset=dataset,
        api_model=cli_args.api_model,
        api_temperature=0.7,
        api_max_tokens=cli_args.api_max_tokens,
        llm_base_url=cli_args.llm_base_url,
        llm_api_key=cli_args.llm_api_key,
        llm_api_keys=cli_args.llm_api_keys,
        llm_timeout=cli_args.llm_timeout,
        llm_verify_ssl=cli_args.llm_verify_ssl,
        llm_trust_env=cli_args.llm_trust_env,
        threads=n_threads,
        start_idx=0,
        end_idx=None,
        single_test=None,
        template_path='./dataset/idx2template_retro.json',
        inventory_path='./dataset/inventory.pkl',
        rule_based_set_path=os.environ.get(
            "AOT_RULE_BASED_SET_PATH",
            os.path.join(BASE_DIR, "scscore", "data", "data_processed.csv"),
        ),
        max_oracle_calls=cli_args.max_oracle_calls,
        freq_log=100,
        output_dir=root_log_dir,
        log_results=True,
        expansion=1,
        num_initial_routes=3,
        depth_decay=1.0,
        backprop_method='average',
        unsolved_penalty=0.5,
        c_param=0.5,
        max_reexpansions=3,
        max_depth=12,
        max_restarts=1,
        round_id=round_id,
        round_max_iterations=round_max_iterations,
        test_mode=False,
        use_ram_templates=True,
        use_presort=cli_args.use_presort,
        no_presort=cli_args.no_presort,
        blurry_tversky=cli_args.blurry_tversky,
        no_blurry_tversky=cli_args.no_blurry_tversky,
        blurry_oracle_rerank=True,
        blurry_rerank_topk=blurry_rerank_topk,
        blurry_rerank_heartbeat_seconds=blurry_rerank_heartbeat_seconds,
        reactionfp_rerank_mode=reactionfp_rerank_mode,
        reactionfp_candidate_pool=reactionfp_candidate_pool,
        reactionfp_selection_mode=reactionfp_selection_mode,
        use_local_cache=bool(run_spec.get('use_local_cache', True)),
        local_cache_path=local_cache_path,
    )

    if not total_molecules:
        raise ValueError("No targets selected")

    # Build work queue and write search_configs (main process only)
    print(f"\nBuilding work queue...")
    work_queue = None if cli_args.dry_run else mp.Queue()
    work_count = 0
    skipped = 0

    for cfg in configs:
        dirname = config_to_dirname(cfg)
        log_dir = os.path.join(root_log_dir, dirname, dataset, cli_args.api_model)
        os.makedirs(log_dir, exist_ok=True)

        # Write search_config.json (main process, no race condition)
        max_restarts = cfg.get('restarts', 1)
        suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
        config_file = os.path.join(log_dir, f"search_config{suffix}.json")
        if not os.path.exists(config_file):
            config_info = {
                "dataset": dataset,
                "total_molecules": total_molecules,
                "model_name": cli_args.api_model,
                "llm_base_url": cli_args.llm_base_url,
                "llm_timeout": cli_args.llm_timeout,
                "llm_max_tokens": cli_args.api_max_tokens,
                "llm_verify_ssl": cli_args.llm_verify_ssl,
                "llm_trust_env": cli_args.llm_trust_env,
                "llm_api_key_count": count_api_keys(cli_args.llm_api_key, cli_args.llm_api_keys),
                "api_temperature": cfg.get('temp', 0.7),
                "max_oracle_calls": cli_args.max_oracle_calls,
                "num_initial_routes": cfg['init'],
                "max_iterations": round_max_iterations,
                "max_depth": cfg['depth'],
                "backprop_method": cfg['bp'],
                "unsolved_penalty": cfg['gamma'],
                "c_param": cfg['c'],
                "max_reexpansions": cfg['reexp'],
                "max_restarts": max_restarts,
                "use_presort": cli_args.use_presort,
                "use_tversky_cutoff": cli_args.use_presort,
                "blurry_tversky": cli_args.blurry_tversky,
                "blurry_oracle_rerank": True,
                "blurry_rerank_topk": blurry_rerank_topk,
                "blurry_rerank_heartbeat_seconds": blurry_rerank_heartbeat_seconds,
                "reactionfp_rerank_mode": reactionfp_rerank_mode,
                "reactionfp_candidate_pool": reactionfp_candidate_pool,
                "reactionfp_selection_mode": reactionfp_selection_mode,
                "use_local_cache": bool(run_spec.get('use_local_cache', True)),
                "local_cache_path": local_cache_path,
                "runner": "unified_search",
                "round_id": round_id,
                "targets_file": targets_file,
                "no_progress_timeout_seconds": no_progress_timeout,
            }
            with open(config_file, 'w') as f:
                json.dump(config_info, f, indent=2)

        for round_pos, target_record in enumerate(round_targets):
            original_idx = int(target_record["original_idx"])
            target = target_record["smiles"]
            result_file = os.path.join(log_dir, f"result_{original_idx:05d}{suffix}.json")
            if os.path.exists(result_file):
                skipped += 1
                continue

            if work_queue is not None:
                work_queue.put({
                    'config': cfg,
                    'original_idx': original_idx,
                    'round_pos': round_pos,
                    'target': target,
                    'total_molecules': total_molecules,
                    'log_dir': log_dir,
                    'round_id': round_id,
                    'round_max_iterations': round_max_iterations,
                })
            work_count += 1

    print(f"  {work_count} work items ({skipped} skipped)")

    if cli_args.dry_run:
        print("\nDry run complete. No workers started.")
        return

    if work_count == 0:
        print("\nNothing to do. All complete.")
        return
    if not cli_args.llm_base_url:
        raise ValueError("Real LLM search requires --llm-base-url")
    if count_api_keys(cli_args.llm_api_key, cli_args.llm_api_keys) == 0:
        raise ValueError("Real LLM search requires --llm-api-key or --llm-api-keys")

    if os.environ.get("AOT_PRELOAD_TRAINING_DATA_BEFORE_FORK", "0") == "1":
        global _PRELOADED_TRAINING_DATA
        global _PRELOADED_TEMPLATE_KEYS
        global _PRELOADED_TEMPLATE_PATTERN_FPS
        global _PRELOADED_TEMPLATE_REACTANT_PATTERN_FPS
        sys.path.insert(0, BASE_DIR)
        os.chdir(BASE_DIR)
        from aotcore.data.loader import load_training_data
        from aotcore.oracle_rerank import pattern_fps_from_templates

        print("\nPreloading training data before fork for copy-on-write sharing...")
        t_preload = time.time()
        _PRELOADED_TRAINING_DATA = load_training_data(use_ram_templates=True)
        _, _, reaction_list_pre, _, _, template_dict_pre = _PRELOADED_TRAINING_DATA
        with open('./dataset/idx2template_retro.json', 'r') as f:
            original_template_dict_pre = json.load(f)
        _PRELOADED_TEMPLATE_KEYS = list(template_dict_pre.keys())
        needs_product_side_pattern_fps = (
            bool(base_args_dict["use_presort"])
            or bool(base_args_dict["blurry_tversky"])
            or str(base_args_dict["reactionfp_rerank_mode"]) == "product_tversky_reactants"
        )
        if needs_product_side_pattern_fps:
            t_pat = time.time()
            _PRELOADED_TEMPLATE_PATTERN_FPS = pattern_fps_from_templates(
                original_template_dict_pre, _PRELOADED_TEMPLATE_KEYS, 0
            )
            print(
                f"  Preloaded {len(_PRELOADED_TEMPLATE_PATTERN_FPS)} product PatternFPs "
                f"in {time.time()-t_pat:.0f}s"
            )
        if (
            not bool(base_args_dict["blurry_tversky"])
            and str(base_args_dict["reactionfp_rerank_mode"]) in {
                "tversky_reactants",
                "product_tversky_reactants",
            }
        ):
            t_react = time.time()
            _PRELOADED_TEMPLATE_REACTANT_PATTERN_FPS = pattern_fps_from_templates(
                {str(i): reaction for i, reaction in enumerate(reaction_list_pre)},
                [str(i) for i in range(len(reaction_list_pre))],
                1,
            )
            print(
                f"  Preloaded {len(_PRELOADED_TEMPLATE_REACTANT_PATTERN_FPS)} reactant "
                f"PatternFPs in {time.time()-t_react:.0f}s"
            )
        if os.environ.get("AOT_SHARE_ORACLE_MODELS", "0") == "1":
            t_oracle = time.time()
            from aotcore.optimizer import preload_oracle_models
            preload_oracle_models()
            print(f"  Preloaded oracle scorer models in {time.time()-t_oracle:.0f}s")
        if os.environ.get("AOT_PRELOAD_INVENTORY_BEFORE_FORK", "0") == "1":
            global _PRELOADED_INVENTORY
            t_inventory = time.time()
            from aotcore.data.loader import load_inventory
            _PRELOADED_INVENTORY = load_inventory('./dataset/inventory.pkl')
            print(
                f"  Preloaded RAM inventory in {time.time()-t_inventory:.0f}s "
                f"({len(_PRELOADED_INVENTORY):,} entries)"
            )
        gc.collect()
        if hasattr(gc, "freeze"):
            gc.freeze()
            print("  Froze parent GC objects before fork")
        print(f"  Preloaded training data in {time.time()-t_preload:.0f}s")

    # Add poison pills (one per thread across all processes)
    total_threads = n_processes * n_threads
    for _ in range(total_threads):
        work_queue.put(None)

    # Result queue for tracking completion
    result_queue = mp.Queue()

    # Start worker processes
    print(f"\nStarting {n_processes} worker processes × {n_threads} threads...")
    print(f"{'='*60}")
    start_time = time.time()

    processes = {}
    active_by_process = {}
    last_progress = {}
    next_process_id = 0

    def start_worker(process_id):
        p = mp.Process(
            target=worker_process,
            args=(process_id, n_threads, work_queue, result_queue, base_args_dict),
        )
        p.start()
        processes[process_id] = p
        active_by_process.setdefault(process_id, {})
        last_progress[process_id] = time.time()

    for _ in range(n_processes):
        start_worker(next_process_id)
        next_process_id += 1

    # Monitor progress (main process)
    completed = 0
    errors = 0
    skips = 0

    def handle_message(msg):
        nonlocal completed, errors, skips
        kind = msg[0]
        if kind == 'done':
            completed += 1
        elif kind == 'error':
            errors += 1
            completed += 1
        elif kind == 'skip':
            skips += 1
        elif kind == 'start':
            _, proc_id, active_item = msg
            active_by_process.setdefault(proc_id, {})[int(active_item["original_idx"])] = active_item
            last_progress[proc_id] = time.time()
        elif kind == 'finish':
            _, proc_id, original_idx = msg
            active_by_process.setdefault(proc_id, {}).pop(int(original_idx), None)
            last_progress[proc_id] = time.time()
        elif kind == 'heartbeat':
            _, proc_id, ts = msg
            last_progress[proc_id] = float(ts)

        if kind in {'done', 'error', 'skip'} and (completed + skips) % 50 == 0 and completed > 0:
            elapsed = time.time() - start_time
            rate = completed / elapsed * 3600
            remaining = work_count - completed
            eta = remaining / (completed / elapsed) if completed > 0 else 0
            print(f"\n  Progress: {completed}/{work_count} completed "
                  f"({completed/work_count*100:.1f}%) "
                  f"Rate: {rate:.0f}/h ETA: {eta/3600:.1f}h "
                  f"Errors: {errors}\n")

    def drain_result_queue():
        while not result_queue.empty():
            try:
                handle_message(result_queue.get_nowait())
            except Exception:
                break

    def mark_active_transient(proc_id, reason):
        nonlocal errors
        active_items = list(active_by_process.get(proc_id, {}).values())
        for active_item in active_items:
            errors += 1
            print(
                f"  TRANSIENT {reason}: process {proc_id}, "
                f"orig {active_item['original_idx']} -> no result written; driver will rerun"
            )
        active_by_process[proc_id] = {}

    # Wait for all workers, collecting results
    alive = True
    while alive:
        # Check result queue
        drain_result_queue()

        now = time.time()
        for proc_id, p in list(processes.items()):
            active_items = active_by_process.get(proc_id, {})
            if p.is_alive() and active_items and now - last_progress.get(proc_id, now) > no_progress_timeout:
                print(
                    f"  WATCHDOG: process {proc_id} produced no output for "
                    f"{int(now - last_progress.get(proc_id, now))}s; terminating"
                )
                p.terminate()
                p.join(timeout=30)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=10)
                mark_active_transient(proc_id, "no_progress_timeout")
                del processes[proc_id]
                if completed + skips < work_count:
                    start_worker(next_process_id)
                    next_process_id += 1
                continue

            if not p.is_alive():
                p.join(timeout=0)
                drain_result_queue()
                had_active = bool(active_by_process.get(proc_id))
                if had_active:
                    mark_active_transient(proc_id, "worker_exited_with_active_targets")
                del processes[proc_id]
                if had_active and completed + skips < work_count:
                    start_worker(next_process_id)
                    next_process_id += 1

        # Check if all processes are still alive
        alive = any(p.is_alive() for p in processes.values())
        if alive:
            time.sleep(5)

    # Drain remaining results
    drain_result_queue()

    # Wait for all processes to finish
    for p in processes.values():
        p.join(timeout=10)

    total_time = time.time() - start_time

    # Generate summaries for completed configs
    print(f"\n{'='*60}")
    print("Generating summaries...")
    print(f"{'='*60}")

    for cfg in configs:
        dirname = config_to_dirname(cfg)
        log_dir = os.path.join(root_log_dir, dirname, dataset, cli_args.api_model)
        max_restarts = cfg.get('restarts', 1)
        suffix = f"_k{max_restarts}" if max_restarts != 1 else ""

        summary_file = os.path.join(log_dir, f"summary{suffix}.json")
        if os.path.exists(summary_file):
            continue

        # Check if all round targets are done
        results = []
        for target_record in round_targets:
            original_idx = int(target_record["original_idx"])
            rf = os.path.join(log_dir, f"result_{original_idx:05d}{suffix}.json")
            if os.path.exists(rf):
                try:
                    results.append(json.load(open(rf)))
                except:
                    pass

        if len(results) < total_molecules:
            print(f"  {dirname}: {len(results)}/{total_molecules} (incomplete)")
            continue

        successful = sum(1 for r in results if r.get('success'))
        total_search_time = sum(r.get('search_time', 0) for r in results)

        summary = {
            "dataset": f"{dataset}/{cli_args.api_model}",
            "total_molecules": total_molecules,
            "successful": successful,
            "failed": total_molecules - successful,
            "success_rate": successful / total_molecules * 100,
            "total_time": total_search_time,
            "avg_time_per_molecule": total_search_time / total_molecules,
            "end_time": datetime.now().isoformat(),
            "config": cfg,
            "round_id": round_id,
            "round_max_iterations": round_max_iterations,
            "targets_file": targets_file,
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(log_dir, f"all_results{suffix}.json"), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  {dirname}: {successful}/{total_molecules} ({summary['success_rate']:.1f}%)")

    # Final report
    print(f"\n{'='*60}")
    print(f"Unified search completed!")
    print(f"  {completed} molecules searched in {total_time/3600:.1f}h")
    print(f"  Rate: {completed/max(total_time,1)*3600:.0f} molecules/h")
    print(f"  Errors: {errors}")
    print(f"{'='*60}")

    # Check for missing results
    missing = []
    for cfg in configs:
        dirname = config_to_dirname(cfg)
        log_dir = os.path.join(root_log_dir, dirname, dataset, cli_args.api_model)
        max_restarts = cfg.get('restarts', 1)
        suffix = f"_k{max_restarts}" if max_restarts != 1 else ""
        for target_record in round_targets:
            original_idx = int(target_record["original_idx"])
            rf = os.path.join(log_dir, f"result_{original_idx:05d}{suffix}.json")
            if not os.path.exists(rf):
                missing.append(f"{dirname}/mol_{original_idx:05d}")

    if missing:
        print(f"\n  WARNING: {len(missing)} missing results (worker crash?)")
        for m in missing[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")


if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
