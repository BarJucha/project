"""Benchmarking script for classical heuristic and Tabu walkers."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np

from src.env.graph_env import GraphNavEnv
from src.data.splits import load_splits_and_pairs
from src.baselines import (
    ShortestPathOracle,
    GreedyWalker,
    DistanceWalker,
    ConnectionWalker,
    RandomWalker,
    TabuWalker,
    tune_stochastic_walker,
)
from src.utils.evaluator import evaluate_agent
from src.utils.metrics import compute_win_rates


def benchmark_heuristics(
    graph_dir: str = "data/processed/facebook_1",
    num_val_pairs: int = 50,
    num_test_pairs: int = 100,
    seed: int = 42,
    output_path: str = "results/heuristic_baselines.json",
):
    g_path = Path(graph_dir)
    with open(g_path / "graph.pkl", "rb") as f:
        g = pickle.load(f)
    features = np.load(g_path / "features.npy")
    splits, val_pairs, test_pairs = load_splits_and_pairs(graph_dir)
    
    val_subset = val_pairs[:num_val_pairs]
    test_subset = test_pairs[:num_test_pairs]
    
    env = GraphNavEnv(g, features, max_steps=100)
    
    print(f"\n[Benchmark] Tuning stochastic walkers on {g_path.name} ({len(val_subset)} val pairs)...")
    best_dist_tau, _ = tune_stochastic_walker(DistanceWalker, env, val_subset, seed=seed)
    best_conn_tau, _ = tune_stochastic_walker(ConnectionWalker, env, val_subset, seed=seed)
    print(f"  Best DistanceWalker tau: {best_dist_tau}")
    print(f"  Best ConnectionWalker tau: {best_conn_tau}")
    
    models = {
        "DijkstraOracle": ShortestPathOracle(g),
        "GreedyWalker": GreedyWalker(),
        "DistanceWalker": DistanceWalker(temperature=best_dist_tau, seed=seed),
        "ConnectionWalker": ConnectionWalker(temperature=best_conn_tau, seed=seed),
        "RandomWalker": RandomWalker(seed=seed),
        "TabuWalker(k=2)": TabuWalker(buffer_size=2, decision_mode="greedy", seed=seed),
        "TabuWalker(k=5)": TabuWalker(buffer_size=5, decision_mode="greedy", seed=seed),
    }
    
    summaries = {}
    path_lengths = {}
    
    print(f"[Benchmark] Evaluating {len(models)} models on {len(test_subset)} test pairs...")
    for name, agent in models.items():
        summary, recs = evaluate_agent(agent, env, test_subset, show_progress=False)
        summaries[name] = summary
        path_lengths[name] = [r["steps"] for r in recs]
        print(f"  Finished {name}: R_trunc={summary['r_trunc']}%, R_bar_oracle={summary['mean_oracle_ratio']}")
        
    # Calculate Win Rates (excluding global Oracle)
    decentralized_lengths = {k: v for k, v in path_lengths.items() if k != "DijkstraOracle"}
    win_rates = compute_win_rates(decentralized_lengths, seed=seed)
    for name in win_rates:
        summaries[name]["win_rate"] = win_rates[name]
    summaries["DijkstraOracle"]["win_rate"] = 0.0
    
    print("\n" + "=" * 105)
    print(f"Benchmark Results on {g_path.name} ({len(test_subset)} test episodes):")
    print(f"{'Model':<18} | {'R_trunc (%)':<11} | {'R_bar_oracle':<17} | {'Revisit Rate':<12} | {'Ping-Pong':<9} | {'Win Rate (%)':<10}")
    print("-" * 105)
    for name, s in summaries.items():
        wr_str = f"{s.get('win_rate', 0.0):.1f}%" if name != "DijkstraOracle" else "N/A"
        r_str = f"{s['mean_oracle_ratio']:.2f} +/- {s['oracle_ratio_ci']:.2f}"
        print(f"{name:<18} | {s['r_trunc']:>9.1f}% | {r_str:>17} | {s['mean_revisit_rate']:>12.2f} | {s['mean_ping_pong_count']:>9.2f} | {wr_str:>10}")
    print("=" * 105 + "\n")
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4)
        
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="data/processed/facebook_1")
    parser.add_argument("--num_val_pairs", type=int, default=50)
    parser.add_argument("--num_test_pairs", type=int, default=100)
    args = parser.parse_args()
    benchmark_heuristics(
        graph_dir=args.graph,
        num_val_pairs=args.num_val_pairs,
        num_test_pairs=args.num_test_pairs,
    )
