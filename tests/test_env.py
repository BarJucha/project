"""Unit tests for GraphNavEnv and EgoGraphData precomputations."""
import pytest
import networkx as nx
import numpy as np
import torch as T

from src.env.graph_env import GraphNavEnv
from src.env.ego_graphs import precompute_ego_graphs


@pytest.fixture
def sample_graph_and_features():
    """Creates a simple connected test graph with known features."""
    g = nx.cycle_graph(6)  # 0-1-2-3-4-5-0
    features = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
    ], dtype=np.float32)
    return g, features


def test_ego_graph_precomputations(sample_graph_and_features):
    """Verify 1-hop ego graphs are correctly constructed."""
    g, features = sample_graph_and_features
    features_t = T.tensor(features, dtype=T.float32)
    ego_data_list = precompute_ego_graphs(g, features_t)
    
    assert len(ego_data_list) == 6
    ego_0 = ego_data_list[0]
    
    # In cycle graph, node 0 has neighbors [1, 5]
    assert ego_0.center_node == 0
    assert ego_0.node_list == [0, 1, 5]
    assert ego_0.action_nodes == [1, 5]
    assert ego_0.action_indices == [1, 2]
    
    # Check PyG Data
    pyg_data = ego_0.pyg_data
    assert pyg_data.num_nodes == 3
    # Features have dim 2 + 1 (indicator) = 3
    assert pyg_data.x.shape == (3, 3)
    # Center node (index 0) has indicator 1.0, others 0.0
    assert pyg_data.x[0, 2] == 1.0
    assert pyg_data.x[1, 2] == 0.0
    assert pyg_data.x[2, 2] == 0.0


def test_env_reset_and_observation(sample_graph_and_features):
    """Verify reset initializes the environment and returns valid observations."""
    g, features = sample_graph_and_features
    env = GraphNavEnv(g, features, max_steps=10)
    
    obs, info = env.reset(source=0, target=3)
    assert obs.current_node == 0
    assert obs.target_node == 3
    assert sorted(obs.neighbors) == [1, 5]
    assert obs.neighbor_features.shape == (2, 2)
    assert obs.neighbor_degrees.shape == (2,)
    assert obs.neighbor_degrees_scaled.shape == (2,)
    assert info["steps"] == 0
    assert info["shortest_path_len"] == 3


def test_env_valid_and_invalid_steps(sample_graph_and_features):
    """Verify valid transitions work and invalid hops raise ValueError."""
    g, features = sample_graph_and_features
    env = GraphNavEnv(g, features, max_steps=10)
    env.reset(source=0, target=3)
    
    # Invalid action (not a neighbor of 0)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(action=3)
        
    # Valid action (hop to 1)
    obs, reward, terminated, truncated, info = env.step(action=1)
    assert obs.current_node == 1
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert info["steps"] == 1
    assert info["trajectory"] == [0, 1]


def test_env_success_termination(sample_graph_and_features):
    """Verify reaching the destination awards +1 reward and sets terminated=True."""
    g, features = sample_graph_and_features
    env = GraphNavEnv(g, features, max_steps=10)
    env.reset(source=0, target=2)
    
    # Path: 0 -> 1 -> 2
    obs, reward, term, trunc, _ = env.step(1)
    assert not term
    assert reward == 0.0
    
    obs, reward, term, trunc, info = env.step(2)
    assert term
    assert not trunc
    assert reward == 1.0
    assert info["is_success"]
    assert info["trajectory"] == [0, 1, 2]


def test_env_truncation_limit(sample_graph_and_features):
    """Verify episode truncates at max_steps with reward 0."""
    g, features = sample_graph_and_features
    env = GraphNavEnv(g, features, max_steps=4)
    env.reset(source=0, target=3)
    
    # Ping-pong between 0 and 1: 0 -> 1 -> 0 -> 1 -> 0
    env.step(1)
    env.step(0)
    env.step(1)
    obs, reward, term, trunc, info = env.step(0)
    
    assert not term
    assert trunc
    assert reward == 0.0
    assert info["steps"] == 4
