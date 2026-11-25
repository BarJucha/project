"""Unit tests for data loaders, synthetic generator, splitting, and auditing."""
import pytest
import networkx as nx
import numpy as np

from src.data.download_snap import download_and_extract_snap
from src.data.facebook_loader import FACEBOOK_GRAPH_MAP, load_facebook_ego_graph
from src.data.kaiser_hilgetag import generate_kaiser_hilgetag_graph
from src.data.splits import (
    create_target_splits,
    generate_evaluation_pairs,
    save_splits_and_pairs,
    load_splits_and_pairs,
)
from src.data.audit import audit_processed_graph, TABLE_4_GROUND_TRUTH


def test_download_snap_exists():
    """Verify SNAP dataset was downloaded and extracted."""
    extract_path = download_and_extract_snap(force=False)
    assert extract_path.exists()
    assert (extract_path / "0.edges").exists()
    assert (extract_path / "0.feat").exists()


@pytest.mark.parametrize("graph_idx,ego_id", list(FACEBOOK_GRAPH_MAP.items()))
def test_load_facebook_ego_graph(graph_idx, ego_id):
    """Verify that each Facebook ego network matches Table 4 specifications."""
    expected = TABLE_4_GROUND_TRUTH[f"facebook_{graph_idx}"]
    g, feats, id_map = load_facebook_ego_graph(ego_id)
    
    # Check node count
    assert g.number_of_nodes() == expected["n"]
    assert feats.shape[0] == expected["n"]
    assert feats.shape[1] == expected["d"]
    assert len(id_map) == expected["n"]
    
    # Check connectivity
    assert nx.is_connected(g)
    
    # Check average shortest path length
    avg_sp = nx.average_shortest_path_length(g)
    assert pytest.approx(avg_sp, abs=0.01) == expected["avg_sp"]
    
    # Check density
    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = 2.0 * m / (n * (n - 1))
    assert pytest.approx(density, abs=0.01) == expected["density"]
    
    # Check node attributes
    for u in g.nodes():
        assert "pos" in g.nodes[u]
        assert "degree" in g.nodes[u]
        assert "degree_scaled" in g.nodes[u]
        assert 0.0 <= g.nodes[u]["degree_scaled"] <= 1.0


def test_kaiser_hilgetag_generator():
    """Verify synthetic graph generator produces valid, connected spatial graphs."""
    g, feats = generate_kaiser_hilgetag_graph(n=50, alpha=30.0, beta=0.4, seed=123)
    assert g.number_of_nodes() == 50
    assert feats.shape == (50, 2)
    assert nx.is_connected(g)
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_leak_free_splits():
    """Verify 80/10/10 target splits have zero overlap and exact partition."""
    nodes = list(range(100))
    splits = create_target_splits(nodes, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    
    s_train = set(splits["train"])
    s_val = set(splits["val"])
    s_test = set(splits["test"])
    
    assert len(s_train) == 80
    assert len(s_val) == 10
    assert len(s_test) == 10
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert (s_train | s_val | s_test) == set(nodes)


def test_evaluation_pairs_no_self_loops():
    """Verify evaluation pairs do not have self-loops and target is in target_nodes."""
    all_nodes = list(range(100))
    target_nodes = list(range(90, 100))
    pairs = generate_evaluation_pairs(all_nodes, target_nodes, num_pairs=200, seed=42)
    
    assert len(pairs) == 200
    for src, tgt in pairs:
        assert src != tgt
        assert src in all_nodes
        assert tgt in target_nodes


def test_audit_processed_facebook_graphs():
    """Verify processed graphs pass all audit criteria."""
    for graph_idx in range(1, 6):
        report = audit_processed_graph(f"data/processed/facebook_{graph_idx}")
        assert report["is_connected"] is True
        assert report["leak_free"] is True
        assert report["val_target_purity"] is True
        assert report["test_target_purity"] is True
        assert report["feature_check_passed"] is True
