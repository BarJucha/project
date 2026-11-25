"""Leak-free dataset partitioning and evaluation pair generator."""
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np


def create_target_splits(
    nodes: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Partitions the node set into disjoint train, val, and test target sets.
    
    Args:
        nodes: List of all node indices in the graph.
        train_ratio: Fraction for training targets (default: 0.8).
        val_ratio: Fraction for validation targets (default: 0.1).
        test_ratio: Fraction for test targets (default: 0.1).
        seed: Random seed for deterministic splitting.
        
    Returns:
        dict with keys 'train', 'val', 'test', containing disjoint node lists.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    rng = random.Random(seed)
    shuffled = sorted(list(nodes))
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    
    train_nodes = shuffled[:n_train]
    val_nodes = shuffled[n_train:n_train + n_val]
    test_nodes = shuffled[n_train + n_val:]
    
    # Guarantee at least 1 node in val and test if graph is small
    if not val_nodes and len(train_nodes) > 2:
        val_nodes.append(train_nodes.pop())
    if not test_nodes and len(train_nodes) > 2:
        test_nodes.append(train_nodes.pop())
        
    # Verify zero leakage
    s_train, s_val, s_test = set(train_nodes), set(val_nodes), set(test_nodes)
    assert s_train.isdisjoint(s_val), "Train and Val target sets overlap!"
    assert s_train.isdisjoint(s_test), "Train and Test target sets overlap!"
    assert s_val.isdisjoint(s_test), "Val and Test target sets overlap!"
    assert len(s_train | s_val | s_test) == n, "Total split nodes do not match graph nodes!"
    
    return {
        "train": sorted(train_nodes),
        "val": sorted(val_nodes),
        "test": sorted(test_nodes),
    }


def generate_evaluation_pairs(
    all_nodes: List[int],
    target_nodes: List[int],
    num_pairs: int = 1000,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Samples reproducible (source, target) routing pairs.
    
    The source node is chosen uniformly from all graph nodes.
    The target node is chosen uniformly from target_nodes (e.g. V_val or V_test).
    Self-routing pairs (source == target) are strictly excluded.
    
    Args:
        all_nodes: List of all accessible graph nodes.
        target_nodes: Subset of nodes designated as valid targets (V_val or V_test).
        num_pairs: Number of pairs to sample.
        seed: Random seed.
        
    Returns:
        List of (source, target) integer tuples.
    """
    rng = random.Random(seed)
    pairs = []
    attempts = 0
    max_attempts = num_pairs * 100
    
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        src = rng.choice(all_nodes)
        tgt = rng.choice(target_nodes)
        if src != tgt:
            pairs.append((src, tgt))
            
    if len(pairs) < num_pairs:
        raise RuntimeError(f"Could not sample {num_pairs} distinct source-target pairs.")
        
    return pairs


def save_splits_and_pairs(
    save_dir: str,
    splits: Dict[str, List[int]],
    val_pairs: List[Tuple[int, int]],
    test_pairs: List[Tuple[int, int]],
) -> None:
    """Serializes splits and evaluation pairs to disk."""
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    with open(out_path / "splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    with open(out_path / "val_pairs.pkl", "wb") as f:
        pickle.dump(val_pairs, f)
    with open(out_path / "test_pairs.pkl", "wb") as f:
        pickle.dump(test_pairs, f)


def load_splits_and_pairs(
    load_dir: str,
) -> Tuple[Dict[str, List[int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Loads pre-saved splits and evaluation pairs from disk."""
    in_path = Path(load_dir)
    with open(in_path / "splits.pkl", "rb") as f:
        splits = pickle.load(f)
    with open(in_path / "val_pairs.pkl", "rb") as f:
        val_pairs = pickle.load(f)
    with open(in_path / "test_pairs.pkl", "rb") as f:
        test_pairs = pickle.load(f)
        
    return splits, val_pairs, test_pairs
