"""Temperature tuning for stochastic heuristic baselines on validation sets."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.env.graph_env import GraphNavEnv
from src.baselines.heuristics import DistanceWalker, ConnectionWalker
from src.utils.evaluator import evaluate_agent
from src.data.splits import load_splits_and_pairs

TUNING_TEMPERATURES = [
    1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 10.0, 100.0
]


def tune_stochastic_walker(
    walker_cls,
    env: GraphNavEnv,
    val_pairs: List[Tuple[int, int]],
    temperatures: List[float] = TUNING_TEMPERATURES,
    seed: int = 42,
) -> Tuple[float, Dict[float, float]]:
    """Grid searches for the optimal temperature tau minimizing Mean Oracle Ratio R_oracle."""
    best_temp = temperatures[0]
    best_ratio = float("inf")
    results = {}
    
    for tau in temperatures:
        walker = walker_cls(temperature=tau, seed=seed)
        summary, _ = evaluate_agent(walker, env, val_pairs, show_progress=False)
        ratio = summary["mean_oracle_ratio"]
        results[tau] = ratio
        if ratio < best_ratio:
            best_ratio = ratio
            best_temp = tau
            
    return best_temp, results


def tune_all_baselines_for_graph(
    graph_dir: str,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Tunes DistanceWalker and ConnectionWalker on a processed graph."""
    import pickle
    
    g_path = Path(graph_dir)
    with open(g_path / "graph.pkl", "rb") as f:
        g = pickle.load(f)
    features = np.load(g_path / "features.npy")
    splits, val_pairs, _ = load_splits_and_pairs(graph_dir)
    
    env = GraphNavEnv(g, features, max_steps=100)
    
    print(f"[Tune] Tuning DistanceWalker on {g_path.name} ({len(val_pairs)} val pairs)...")
    best_dist_tau, dist_curve = tune_stochastic_walker(DistanceWalker, env, val_pairs, seed=seed)
    
    print(f"[Tune] Tuning ConnectionWalker on {g_path.name}...")
    best_conn_tau, conn_curve = tune_stochastic_walker(ConnectionWalker, env, val_pairs, seed=seed)
    
    best_params = {
        "graph_name": g_path.name,
        "DistanceWalker": {"best_tau": best_dist_tau, "curve": {str(k): v for k, v in dist_curve.items()}},
        "ConnectionWalker": {"best_tau": best_conn_tau, "curve": {str(k): v for k, v in conn_curve.items()}},
    }
    
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=4)
            
    return best_params
