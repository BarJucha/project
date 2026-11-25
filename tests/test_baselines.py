"""Unit tests for heuristic baselines, Tabu walker, and tuning utilities."""
import networkx as nx
import numpy as np
import pytest

from src.env.graph_env import GraphNavEnv
from src.baselines.heuristics import (
    RandomWalker,
    GreedyWalker,
    DistanceWalker,
    ConnectionWalker,
)
from src.baselines.tabu import TabuWalker
from src.baselines.tune import tune_stochastic_walker
from src.utils.evaluator import evaluate_agent


@pytest.fixture
def test_setup():
    """Builds a test graph with 2D attributes where Euclidean distance points toward target."""
    # Line graph: 0 - 1 - 2 - 3 - 4
    g = nx.path_graph(5)
    features = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ], dtype=np.float32)
    env = GraphNavEnv(g, features, max_steps=20)
    return env


def test_random_walker_valid_steps(test_setup):
    """Verify RandomWalker always chooses valid neighbors."""
    env = test_setup
    walker = RandomWalker(seed=42)
    obs, _ = env.reset(source=1, target=4)
    action = walker.select_action(obs)
    assert action in obs.neighbors


def test_greedy_walker_direct_progress(test_setup):
    """Verify GreedyWalker selects closest neighbor in feature space."""
    env = test_setup
    walker = GreedyWalker()
    obs, _ = env.reset(source=1, target=4)
    # Node 1 neighbors: [0, 2]. Node 2 is closer to target 4 ([2,0] vs [4,0]).
    action = walker.select_action(obs)
    assert action == 2


def test_distance_walker_distribution(test_setup):
    """Verify DistanceWalker selects valid actions probabilistically."""
    env = test_setup
    walker = DistanceWalker(temperature=0.1, seed=42)
    obs, _ = env.reset(source=1, target=4)
    action = walker.select_action(obs)
    assert action in obs.neighbors


def test_connection_walker_hub_bias():
    """Verify ConnectionWalker is biased toward high-degree hub nodes."""
    # Star graph: center 0 has degree 4, leaves 1,2,3,4 have degree 1
    g = nx.star_graph(4)
    features = np.zeros((5, 2), dtype=np.float32)
    env = GraphNavEnv(g, features, max_steps=10)
    
    walker = ConnectionWalker(temperature=0.01, seed=42)
    obs, _ = env.reset(source=1, target=4)
    # Neighbor of leaf 1 is only hub 0
    action = walker.select_action(obs)
    assert action == 0


def test_tabu_walker_prevents_immediate_backtrack():
    """Verify TabuWalker avoids immediately backtracking to the previous node."""
    # Triangle graph: 0 - 1, 1 - 2, 2 - 0
    g = nx.cycle_graph(3)
    features = np.zeros((3, 2), dtype=np.float32)
    env = GraphNavEnv(g, features, max_steps=10)
    
    walker = TabuWalker(buffer_size=2, decision_mode="random", seed=42)
    walker.reset(source=0, target=2)
    
    # At node 0: neighbors are 1 and 2. Target 2 is immediate neighbor -> jumps to 2!
    obs, _ = env.reset(source=0, target=2)
    action = walker.select_action(obs)
    assert action == 2
    
    # Path without immediate target:
    walker.reset(source=0, target=99)  # fictitious target
    obs, _, _, _, _ = env.step(1)      # now at node 1, came from 0
    # Tabu buffer contains: [0, 1]
    # Neighbors of 1 are [0, 2]. 0 is tabu! Action must be 2.
    action = walker.select_action(obs)
    assert action == 2


def test_tune_walker(test_setup):
    """Verify temperature tuner returns a best temperature and dictionary curve."""
    env = test_setup
    val_pairs = [(0, 4), (1, 4), (0, 3)]
    best_tau, curve = tune_stochastic_walker(
        DistanceWalker,
        env,
        val_pairs,
        temperatures=[0.01, 0.1, 1.0],
        seed=42,
    )
    assert best_tau in [0.01, 0.1, 1.0]
    assert len(curve) == 3
