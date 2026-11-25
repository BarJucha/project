"""Auditing utilities to verify graph statistics, connectivity, and leak-free splits."""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
import networkx as nx
import numpy as np


# Ground-truth statistics from Table 4 (Pisacane et al., LoG 2024)
TABLE_4_GROUND_TRUTH = {
    "facebook_1": {"ego_id": 414, "n": 148, "avg_sp": 2.69, "density": 0.16, "d": 105},
    "facebook_2": {"ego_id": 686, "n": 168, "avg_sp": 2.43, "density": 0.12, "d": 63},
    "facebook_3": {"ego_id": 348, "n": 224, "avg_sp": 2.52, "density": 0.13, "d": 161},
    "facebook_4": {"ego_id": 0, "n": 324, "avg_sp": 3.75, "density": 0.05, "d": 224},
    "facebook_5": {"ego_id": 3437, "n": 532, "avg_sp": 3.45, "density": 0.03, "d": 262},
}


def audit_processed_graph(graph_dir: str) -> Dict[str, Any]:
    """Audits a processed graph directory for connectivity, features, and split purity."""
    p = Path(graph_dir)
    if not (p / "graph.pkl").exists():
        raise FileNotFoundError(f"Missing graph.pkl in {graph_dir}")
        
    with open(p / "graph.pkl", "rb") as f:
        g: nx.Graph = pickle.load(f)
        
    features = np.load(p / "features.npy")
    with open(p / "splits.pkl", "rb") as f:
        splits = pickle.load(f)
    with open(p / "val_pairs.pkl", "rb") as f:
        val_pairs = pickle.load(f)
    with open(p / "test_pairs.pkl", "rb") as f:
        test_pairs = pickle.load(f)
        
    n = g.number_of_nodes()
    m = g.number_of_edges()
    is_connected = nx.is_connected(g)
    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0
    avg_sp = nx.average_shortest_path_length(g) if is_connected else float("inf")
    d = features.shape[1]
    
    # Check splits
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    leakage = False
    if not train_set.isdisjoint(val_set) or not train_set.isdisjoint(test_set) or not val_set.isdisjoint(test_set):
        leakage = True
        
    # Check evaluation pairs
    val_targets_in_val = all(tgt in val_set for _, tgt in val_pairs)
    test_targets_in_test = all(tgt in test_set for _, tgt in test_pairs)
    no_self_loops_val = all(src != tgt for src, tgt in val_pairs)
    no_self_loops_test = all(src != tgt for src, tgt in test_pairs)
    
    # Feature completeness
    all_nodes_have_features = all("pos" in g.nodes[u] for u in g.nodes())
    features_finite = bool(np.all(np.isfinite(features)))
    
    return {
        "name": p.name,
        "nodes": n,
        "edges": m,
        "is_connected": is_connected,
        "density": round(density, 4),
        "avg_shortest_path": round(avg_sp, 4),
        "attribute_dim": d,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "leak_free": not leakage,
        "val_target_purity": val_targets_in_val and no_self_loops_val,
        "test_target_purity": test_targets_in_test and no_self_loops_test,
        "feature_check_passed": all_nodes_have_features and features_finite,
    }


def print_audit_report(processed_dir: str = "data/processed") -> None:
    """Audits all processed graphs and prints a summary report."""
    base_path = Path(processed_dir)
    graph_dirs = [d for d in base_path.iterdir() if d.is_dir() and (d / "graph.pkl").exists()]
    
    print("\n" + "=" * 95)
    print(f"{'Graph':<15} | {'Nodes':<6} | {'Edges':<6} | {'Avg SP':<8} | {'Density':<8} | {'Dim':<5} | {'Connected':<9} | {'Leak-Free':<9}")
    print("-" * 95)
    
    for gdir in sorted(graph_dirs):
        stats = audit_processed_graph(str(gdir))
        print(f"{stats['name']:<15} | {stats['nodes']:<6} | {stats['edges']:<6} | {stats['avg_shortest_path']:<8.2f} | {stats['density']:<8.4f} | {stats['attribute_dim']:<5} | {str(stats['is_connected']):<9} | {str(stats['leak_free']):<9}")
        
    print("=" * 95 + "\n")


if __name__ == "__main__":
    print_audit_report()
