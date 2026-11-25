"""End-to-end data preparation pipeline.

Downloads SNAP Facebook ego-networks, processes graphs, extracts features,
generates synthetic homophilic graphs, and prepares leak-free 80/10/10 splits.
"""
import json
import pickle
from pathlib import Path
import numpy as np

from src.data.download_snap import download_and_extract_snap
from src.data.facebook_loader import FACEBOOK_GRAPH_MAP, load_facebook_ego_graph
from src.data.kaiser_hilgetag import generate_kaiser_hilgetag_graph
from src.data.splits import create_target_splits, generate_evaluation_pairs, save_splits_and_pairs
from src.data.audit import print_audit_report


def process_and_save_graph(
    graph,
    features,
    save_dir: Path,
    metadata: dict,
    num_val_pairs: int = 1000,
    num_test_pairs: int = 1000,
    seed: int = 42,
) -> None:
    """Saves graph, features, leak-free splits, and metadata to save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save NetworkX graph
    with open(save_dir / "graph.pkl", "wb") as f:
        pickle.dump(graph, f)
        
    # 2. Save feature matrix
    np.save(save_dir / "features.npy", features)
    
    # 3. Create and save splits
    all_nodes = list(graph.nodes())
    splits = create_target_splits(all_nodes, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed)
    val_pairs = generate_evaluation_pairs(all_nodes, splits["val"], num_pairs=num_val_pairs, seed=seed + 1)
    test_pairs = generate_evaluation_pairs(all_nodes, splits["test"], num_pairs=num_test_pairs, seed=seed + 2)
    save_splits_and_pairs(str(save_dir), splits, val_pairs, test_pairs)
    
    # 4. Save metadata / info.json
    metadata.update({
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "feature_dim": features.shape[1],
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
    })
    with open(save_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def run_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    force_download: bool = False,
) -> None:
    """Executes the full data preparation pipeline."""
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Download SNAP Facebook dataset
    download_and_extract_snap(raw_dir=raw_dir, force=force_download)
    
    # 2. Process Facebook Graphs 1 to 5
    print("\n[Pipeline] Processing 5 SNAP Facebook ego-networks...")
    for graph_idx in range(1, 6):
        ego_id = FACEBOOK_GRAPH_MAP[graph_idx]
        graph_name = f"facebook_{graph_idx}"
        save_dir = proc_path / graph_name
        
        print(f"  -> Processing {graph_name} (SNAP Ego {ego_id})...")
        g, feats, id_map = load_facebook_ego_graph(ego_id, raw_dir=f"{raw_dir}/facebook")
        
        metadata = {
            "dataset_type": "facebook_ego",
            "graph_index": graph_idx,
            "ego_id": ego_id,
        }
        process_and_save_graph(g, feats, save_dir, metadata, seed=42)
        
    # 3. Generate benchmark synthetic graphs
    print("\n[Pipeline] Generating synthetic Kaiser-Hilgetag benchmark graphs...")
    synthetic_betas = [0.1, 0.4, 1.0, 5.0]
    for beta in synthetic_betas:
        graph_name = f"synthetic_beta_{str(beta).replace('.', 'p')}"
        save_dir = proc_path / graph_name
        print(f"  -> Generating {graph_name} (n=200, alpha=30, beta={beta})...")
        g, feats = generate_kaiser_hilgetag_graph(n=200, alpha=30.0, beta=beta, seed=42)
        
        metadata = {
            "dataset_type": "synthetic_kaiser_hilgetag",
            "n": 200,
            "alpha": 30.0,
            "beta": beta,
        }
        process_and_save_graph(g, feats, save_dir, metadata, seed=42)
        
    print("\n[Pipeline] Auditing all processed graphs:")
    print_audit_report(processed_dir=processed_dir)
    print("[Pipeline] Data preparation complete!\n")


if __name__ == "__main__":
    run_pipeline()
