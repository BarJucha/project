"""Evaluation metrics for decentralized graph routing experiments.

Implements core metrics from Pisacane et al. (LoG 2024) and loop/cycle diagnostics:
    - Truncation Rate (R_trunc %)
    - Mean Oracle Ratio (R_bar_oracle) with 95% Confidence Intervals
    - Win Rate (R_win %) with random tie-breaking
    - Revisit Rate and 2-step / multi-step cycle counts
"""
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.stats as sp_stats


def analyze_trajectory(trajectory: List[int]) -> Dict[str, Any]:
    """Analyzes a single trajectory for node revisits, ping-pong oscillations, and loops.
    
    Args:
        trajectory: Chronological list of visited node IDs [u_0, u_1, ..., u_T].
        
    Returns:
        dict containing revisit counts, 2-hop oscillations, and multi-hop cycles.
    """
    total_steps = len(trajectory) - 1
    total_nodes = len(trajectory)
    unique_nodes = len(set(trajectory))
    revisit_count = total_nodes - unique_nodes
    revisit_rate = revisit_count / total_nodes if total_nodes > 0 else 0.0
    
    # 2-step oscillations (ping-pong: u -> v -> u)
    ping_pong_count = 0
    for t in range(len(trajectory) - 2):
        if trajectory[t] == trajectory[t + 2]:
            ping_pong_count += 1
            
    # General cycles (revisiting any previous node after >= 2 steps)
    cycles_detected = 0
    seen = {}
    for idx, node in enumerate(trajectory):
        if node in seen:
            prev_idx = seen[node]
            if (idx - prev_idx) >= 2:
                cycles_detected += 1
        seen[node] = idx
        
    return {
        "steps": total_steps,
        "unique_nodes": unique_nodes,
        "revisit_count": revisit_count,
        "revisit_rate": revisit_rate,
        "ping_pong_count": ping_pong_count,
        "cycle_count": cycles_detected,
    }


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> float:
    """Computes the margin of error for a given confidence interval using Student's t."""
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n <= 1:
        return 0.0
    sem = sp_stats.sem(arr)
    h = sem * sp_stats.t.ppf((1.0 + confidence) / 2.0, n - 1)
    return float(h)


def summarize_evaluation_results(
    episode_records: List[Dict[str, Any]],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Aggregates episode records into summary benchmark metrics.
    
    Args:
        episode_records: List of per-episode dictionaries with keys:
            'steps', 'shortest_path_len', 'truncated', 'terminated', etc.
        confidence: Confidence level for CI calculation (default 0.95).
        
    Returns:
        Dictionary of aggregated metrics matching Table 1 format.
    """
    n_episodes = len(episode_records)
    if n_episodes == 0:
        return {}
        
    truncated_flags = [rec["truncated"] for rec in episode_records]
    r_trunc = (sum(truncated_flags) / n_episodes) * 100.0
    
    # Oracle ratios
    oracle_ratios = []
    path_lengths = []
    revisit_rates = []
    ping_pongs = []
    
    for rec in episode_records:
        steps = rec["steps"]
        sp_len = rec["shortest_path_len"]
        ratio = steps / sp_len if sp_len > 0 else 1.0
        oracle_ratios.append(ratio)
        path_lengths.append(steps)
        if "revisit_rate" in rec:
            revisit_rates.append(rec["revisit_rate"])
        if "ping_pong_count" in rec:
            ping_pongs.append(rec["ping_pong_count"])
            
    mean_ratio = float(np.mean(oracle_ratios))
    ci_ratio = compute_confidence_interval(oracle_ratios, confidence=confidence)
    sem_ratio = float(sp_stats.sem(oracle_ratios)) if n_episodes > 1 else 0.0
    
    summary = {
        "num_episodes": n_episodes,
        "r_trunc": round(r_trunc, 2),
        "mean_oracle_ratio": round(mean_ratio, 2),
        "oracle_ratio_ci": round(ci_ratio, 2),
        "oracle_ratio_sem": round(sem_ratio, 2),
        "mean_path_length": round(float(np.mean(path_lengths)), 2),
        "mean_revisit_rate": round(float(np.mean(revisit_rates)), 4) if revisit_rates else 0.0,
        "mean_ping_pong_count": round(float(np.mean(ping_pongs)), 2) if ping_pongs else 0.0,
    }
    return summary


def compute_win_rates(
    model_path_lengths: Dict[str, List[int]],
    seed: int = 42,
) -> Dict[str, float]:
    """Computes Win Rate (R_win %) among competing methods with random tie-breaking.
    
    Args:
        model_path_lengths: Mapping of model_name -> list of path lengths across N episodes.
        seed: Random seed for tie-breaking.
        
    Returns:
        Mapping of model_name -> win_rate_percentage.
    """
    rng = random.Random(seed)
    model_names = list(model_path_lengths.keys())
    if not model_names:
        return {}
        
    n_episodes = len(model_path_lengths[model_names[0]])
    wins = {name: 0 for name in model_names}
    
    for ep_idx in range(n_episodes):
        lengths = {name: model_path_lengths[name][ep_idx] for name in model_names}
        min_len = min(lengths.values())
        tied_models = [name for name, l in lengths.items() if l == min_len]
        
        # Break tie uniformly at random
        winner = rng.choice(tied_models)
        wins[winner] += 1
        
    win_rates = {name: round((wins[name] / n_episodes) * 100.0, 2) for name in model_names}
    return win_rates
