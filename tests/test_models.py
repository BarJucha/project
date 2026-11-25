"""Unit tests for GAT, Actor-Critic, GARDEN, GRU, Deep Sets agents, and trainers."""
import networkx as nx
import numpy as np
import pytest
import torch as T

from src.env.graph_env import GraphNavEnv
from src.models.networks import GATEncoder, ActorNetwork, CriticNetwork
from src.models.garden import GARDENAgent, MLPA2CAgent
from src.models.memory import GRUMemoryAgent, DeepSetsMemoryAgent
from src.training.a2c_trainer import A2CTrainer
from src.training.recurrent_trainer import RecurrentA2CTrainer
from src.utils.evaluator import evaluate_agent


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
    """Verify GARDEN agent embeds graph, switches train/eval modes, and selects valid actions."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10)

    agent = GARDENAgent(env, hidden_dim=32, rep_dim=16, use_degree=True)
    obs, _ = env.reset(source=0, target=3)

    action = agent.select_action(obs)
    assert action in obs.neighbors

    val = agent.get_value(0, 3)
    assert val.shape == (1,)

    agent.eval()
    assert not agent.train_mode
    assert agent.greedy_mode


def test_a2c_trainer_multi_episode_update(tmp_path):
    """Verify A2CTrainer runs multiple episodes and validation checkpointing cleanly."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10, target_nodes=[2, 3, 4], seed=42)
    agent = GARDENAgent(env, hidden_dim=32, rep_dim=16, use_degree=True)

    trainer = A2CTrainer(
        agent=agent,
        env=env,
        train_targets=[2, 3, 4],
        val_pairs=[(0, 3), (1, 4)],
        checkpoint_dir=str(tmp_path),
        eval_every=2,
    )

    history = trainer.train(num_episodes=3, val_subset_size=2, verbose=False)
    assert len(history["loss"]) == 3
    assert len(history["val_oracle_ratio"]) == 2
    assert (tmp_path / f"{agent.name}_best.pt").exists()


def test_gru_memory_agent_state_and_checkpoint(tmp_path):
    """Verify GRUMemoryAgent updates hidden state sequentially and saves/loads checkpoints."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10)

    agent = GRUMemoryAgent(env, hidden_dim=32, rep_dim=16, memory_dim=12)
    agent.eval()
    agent.reset(source=0, target=3)
    z0 = agent.get_memory_state().clone()
    assert z0.shape == (12,)

    agent.update_memory(1)
    z1 = agent.get_memory_state().clone()
    assert not T.allclose(z0, z1)

    # Reset must restore initial memory for the same source
    agent.reset(source=0, target=3)
    assert T.allclose(z0, agent.get_memory_state(), atol=1e-6)

    ckpt_path = tmp_path / "gru_test.pt"
    agent.save_checkpoint(str(ckpt_path))
    agent.load_checkpoint(str(ckpt_path))

    obs, _ = env.reset(source=0, target=3)
    action = agent.select_action(obs)
    assert action in obs.neighbors


def test_deepsets_memory_permutation_invariance_and_window():
    """Verify DeepSetsMemoryAgent is permutation-invariant over visited nodes and supports windowing."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10)

    agent = DeepSetsMemoryAgent(env, hidden_dim=32, rep_dim=16, memory_dim=12, deduplicate=True)
    agent.eval()

    # Visit {0, 1, 2}
    agent.reset(source=0, target=4)
    agent.update_memory(1)
    agent.update_memory(2)
    z_order_1 = agent.get_memory_state().clone()

    # Visit {0, 2, 1} + duplicate 1
    agent.reset(source=0, target=4)
    agent.update_memory(2)
    agent.update_memory(1)
    agent.update_memory(1)
    z_order_2 = agent.get_memory_state().clone()
    assert T.allclose(z_order_1, z_order_2, atol=1e-6)

    # Window size caps stored unique nodes
    windowed = DeepSetsMemoryAgent(env, hidden_dim=32, rep_dim=16, memory_dim=12, window_size=2)
    windowed.eval()
    windowed.reset(source=0, target=4)
    windowed.update_memory(1)
    windowed.update_memory(2)
    assert windowed._visited == [1, 2]


@pytest.mark.parametrize("agent_cls", [GRUMemoryAgent, DeepSetsMemoryAgent])
def test_recurrent_a2c_trainer_bptt_and_eval(agent_cls, tmp_path):
    """Verify RecurrentA2CTrainer updates memory & policy weights via BPTT and works with evaluate_agent."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10, target_nodes=[2, 3, 4], seed=7)

    agent = agent_cls(env, hidden_dim=32, rep_dim=16, memory_dim=12, lr=1e-3, name=agent_cls.__name__)
    initial_actor_weight = agent.actor.fc1.weight.clone().detach()
    initial_mem_params = [p.clone().detach() for mod in agent._memory_modules().values() for p in mod.parameters()]

    trainer = RecurrentA2CTrainer(
        agent=agent,
        env=env,
        train_targets=[2, 3, 4],
        val_pairs=[(0, 3), (1, 4)],
        checkpoint_dir=str(tmp_path),
        eval_every=2,
        step_penalty=-0.01,
    )

    history = trainer.train(num_episodes=3, val_subset_size=2, verbose=False)
    assert len(history["loss"]) == 3
    assert len(history["val_oracle_ratio"]) == 2
    assert (tmp_path / f"{agent.name}_best.pt").exists()

    # Parameters should have updated after training episodes
    assert not T.allclose(initial_actor_weight, agent.actor.fc1.weight)
    updated_mem_params = [p for mod in agent._memory_modules().values() for p in mod.parameters()]
    assert any(not T.allclose(p0, p1) for p0, p1 in zip(initial_mem_params, updated_mem_params))

    # End-to-end evaluator integration check
    agent.eval()
    summary, records = evaluate_agent(agent, env, [(0, 3), (1, 4)])
    assert summary["num_episodes"] == 2
    assert len(records) == 2


def test_freeze_gat_keeps_encoder_weights_constant(tmp_path):
    """Verify freeze_gat=True prevents GAT parameter updates during BPTT training."""
    g = nx.cycle_graph(6)
    features = np.random.randn(6, 8).astype(np.float32)
    env = GraphNavEnv(g, features, max_steps=10, target_nodes=[2, 3, 4], seed=11)

    agent = GRUMemoryAgent(env, hidden_dim=32, rep_dim=16, memory_dim=12, freeze_gat=True)
    initial_gnn = [p.clone().detach() for p in agent.gnn.parameters()]

    trainer = RecurrentA2CTrainer(
        agent=agent,
        env=env,
        train_targets=[2, 3, 4],
        val_pairs=[(0, 3)],
        checkpoint_dir=str(tmp_path),
        eval_every=5,
    )
    trainer.train_episode()

    for p0, p1 in zip(initial_gnn, agent.gnn.parameters()):
        assert T.allclose(p0, p1)
