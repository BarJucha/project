"""Unit tests for Dijkstra Oracle, metrics calculation, and evaluation loop."""
import networkx as nx
import numpy as np
import pytest

from src.baselines.oracle import ShortestPathOracle
from src.env.graph_env import GraphNavEnv
from src.utils.metrics import (
    analyze_trajectory,
    compute_confidence_interval,
    summarize_evaluation_results,
    compute_win_rates,
)
from src.utils.evaluator import evaluate_agent


def test_shortest_path_oracle():
    """Verify DijkstraOracle finds true shortest paths and lengths."""
    g = nx.cycle_graph(10)
    oracle = ShortestPathOracle(g)
    
    path = oracle.get_shortest_path(0, 4)
    assert path == [0, 1, 2, 3, 4]
    assert oracle.get_shortest_path_length(0, 4) == 4
    
    # 0 to 6 in C_10 is shorter via 9, 8, 7, 6 (length 4)
    assert oracle.get_shortest_path_length(0, 6) == 4


def test_trajectory_diagnostics():
    """Verify detection of revisit rates, ping-pongs, and cycles."""
    # Linear path with no revisits: 0 -> 1 -> 2 -> 3
    diag1 = analyze_trajectory([0, 1, 2, 3])
    assert diag1["steps"] == 3
    assert diag1["revisit_count"] == 0
    assert diag1["ping_pong_count"] == 0
    assert diag1["cycle_count"] == 0
    
    # Ping-pong oscillation: 0 -> 1 -> 0 -> 1 -> 0
    diag2 = analyze_trajectory([0, 1, 0, 1, 0])
    assert diag2["steps"] == 4
    assert diag2["revisit_count"] == 3
    assert diag2["ping_pong_count"] == 3  # (0,1,0), (1,0,1), (0,1,0)
    assert diag2["cycle_count"] == 3
    
    # Triangular cycle: 0 -> 1 -> 2 -> 0
    diag3 = analyze_trajectory([0, 1, 2, 0])
    assert diag3["revisit_count"] == 1
    assert diag3["ping_pong_count"] == 0
    assert diag3["cycle_count"] == 1


def test_confidence_interval():
    """Verify confidence interval calculation."""
    data = [10.0, 10.0, 10.0, 10.0]
    ci = compute_confidence_interval(data)
    assert ci == 0.0
    
    data_var = [8.0, 10.0, 12.0, 9.0, 11.0]
    ci_var = compute_confidence_interval(data_var)
    assert ci_var > 0.0


def test_win_rates_tie_breaking():
    """Verify win rate computation and fair tie breaking."""
    competing = {
        "model_A": [2, 5, 10],
        "model_B": [3, 2, 10],
    }
    win_rates = compute_win_rates(competing, seed=42)
    assert "model_A" in win_rates
    assert "model_B" in win_rates
    assert pytest.approx(win_rates["model_A"] + win_rates["model_B"]) == 100.0


def test_evaluate_agent_oracle_on_env():
    """Verify evaluating ShortestPathOracle gives R_oracle = 1.0 and R_trunc = 0%."""
    g = nx.cycle_graph(6)
    features = np.zeros((6, 4), dtype=np.float32)
    env = GraphNavEnv(g, features, max_steps=10)
    oracle = ShortestPathOracle(g)
    
    pairs = [(0, 3), (1, 4), (2, 5)]
    summary, records = evaluate_agent(oracle, env, pairs)
    
    assert summary["r_trunc"] == 0.0
    assert summary["mean_oracle_ratio"] == 1.0
    assert len(records) == 3
    for rec in records:
        assert rec["is_success"] is True
        assert rec["steps"] == rec["shortest_path_len"]
