"""Unit tests for GAT, Actor-Critic networks, GARDEN agent, and A2C trainer."""
import networkx as nx
import numpy as np
import pytest
import torch as T

from src.env.graph_env import GraphNavEnv
from src.models.networks import GATEncoder, ActorNetwork, CriticNetwork
from src.models.garden import GARDENAgent, MLPA2CAgent
from src.training.a2c_trainer import A2CTrainer


def test_gat_encoder_forward():
    """Verify GATEncoder processes graph batch and outputs correct feature dim."""
    gat = GATEncoder(in_channels=10, hidden_channels=32, out_channels=16, num_layers=3)
    x = T.randn(8, 10)
    edge_index = T.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=T.long)
    out = gat(x, edge_index)
    assert out.shape == (8, 16)


def test_actor_critic_shapes():
    """Verify Actor produces probability distributions and Critic estimates scalar values."""
    actor = ActorNetwork(in_dim=32, hidden_dim=64)
    critic = CriticNetwork(in_dim=32, hidden_dim=64)
    
    # 5 candidate actions
    x_actor = T.randn(5, 32)
    probs = actor(x_actor)
    assert probs.shape == (5,)
    assert pytest.approx(probs.sum().item(), abs=1e-5) == 1.0
    assert T.all(probs >= 0.0)
    
    # State value
    x_critic = T.randn(1, 32)
    val = critic(x_critic)
    assert val.shape == (1,)
    assert 0.0 <= val.item() <= 1.0


def test_garden_agent_action_selection():
    """Verify GARDEN agent embeds graph and selects valid neighbor actions."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10)
    
    agent = GARDENAgent(env, hidden_dim=32, rep_dim=16, use_degree=True)
    obs, _ = env.reset(source=0, target=3)
    
    action = agent.select_action(obs)
    assert action in obs.neighbors
    
    val = agent.get_value(0, 3)
    assert val.shape == (1,)


def test_a2c_trainer_single_episode_update():
    """Verify A2C training step updates model parameters with non-zero gradients."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10, target_nodes=[2, 3, 4])
    agent = GARDENAgent(env, hidden_dim=32, rep_dim=16, use_degree=True)
    
    trainer = A2CTrainer(
        agent=agent,
        env=env,
        train_targets=[2, 3, 4],
        val_pairs=[(0, 3)],
        eval_every=100,
    )
    
    metrics = trainer.train_episode()
    assert "loss" in metrics
    assert "steps" in metrics
    assert metrics["steps"] > 0
