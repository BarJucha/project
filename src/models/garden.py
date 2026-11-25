"""GARDEN and MLP Advantage Actor-Critic (A2C) agents.

Reference:
    Alexei Pisacane, Victor-Alexandru Darvariu, Mirco Musolesi.
    Reinforcement Learning Discovers Efficient Decentralized Graph Path Search Strategies.
    Learning on Graphs Conference (LoG 2024).
"""
import os
from typing import Optional, List, Dict, Any, Tuple
import torch as T
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric import utils as tg_utils

from src.env.graph_env import GraphNavEnv, Observation
from src.baselines.heuristics import BaseWalker
from src.models.networks import GATEncoder, ActorNetwork, CriticNetwork


class GARDENAgent(BaseWalker):
    """Graph Attention-guided Routing for DEcentralised Networks (GARDEN).
    
    Memory-less graph navigation baseline using 3-layer GATv2 to embed 1-hop
    ego-neighborhoods, trained under CTDE with Advantage Actor-Critic.
    """
    def __init__(
        self,
        env: GraphNavEnv,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 1e-3,
        use_degree: bool = True,
        device: str = "cpu",
        name: str = "GARDEN",
    ):
        super().__init__(name=name)
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_degree = use_degree
        self.rep_dim = rep_dim
        self.device = T.device(device)
        self.train_mode = True
        self.greedy_mode = False

        # Build PyG Ego-Graph Batch for fast parallel GAT forward pass
        ego_pyg_list = [ego.pyg_data for ego in env.ego_graphs]
        self.batch_data = Batch.from_data_list(ego_pyg_list).to(self.device)
        
        # Ego feature dim: raw attribute dim + 1 (center indicator)
        in_channels = ego_pyg_list[0].x.shape[1]
        
        # Networks
        self.gnn = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=rep_dim,
            num_layers=3,
        ).to(self.device)
        
        node_emb_dim = rep_dim + (1 if use_degree else 0)
        in_actor_critic = 2 * node_emb_dim
        
        self.actor = ActorNetwork(in_dim=in_actor_critic, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(in_dim=in_actor_critic, hidden_dim=hidden_dim).to(self.device)
        
        # Degrees tensor on device
        self.degrees_scaled = env.degrees_scaled_tensor.unsqueeze(-1).to(self.device)
        
        params = list(self.gnn.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = T.optim.Adam(params, lr=lr)
        
        # Cached continuous node embeddings
        self.cached_embeddings: Optional[T.Tensor] = None

    def embed_graph(self) -> T.Tensor:
        """Computes continuous latent representations for all nodes in the graph."""
        node_out = self.gnn(self.batch_data.x, self.batch_data.edge_index)
        # Mean pool over ego-graph nodes
        graph_embs = tg_utils.scatter(
            node_out, self.batch_data.batch, dim=0, reduce="mean"
        )
        if self.use_degree:
            graph_embs = T.cat([graph_embs, self.degrees_scaled], dim=-1)
            
        self.cached_embeddings = graph_embs
        return graph_embs

    def reset(self, source: int, target: int) -> None:
        """Called at start of each episode."""
        if self.train_mode or self.cached_embeddings is None:
            self.embed_graph()

    def train(self, mode: bool = True) -> "GARDENAgent":
        """Switches between training (stochastic) and evaluation (greedy) modes."""
        self.train_mode = mode
        self.greedy_mode = not mode
        self.gnn.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.cached_embeddings = None
        return self

    def eval(self) -> "GARDENAgent":
        """Sets the agent to greedy evaluation mode."""
        return self.train(False)

    def get_action_distribution(self, obs: Observation) -> Tuple[T.Tensor, T.Tensor]:
        """Calculates policy probabilities and log-probabilities for candidate actions."""
        if self.cached_embeddings is None:
            self.embed_graph()
            
        u = obs.current_node
        tgt = obs.target_node
        neighbors = obs.neighbors
        
        nbr_embs = self.cached_embeddings[neighbors]  # Shape (|N(u)|, node_emb_dim)
        tgt_emb = self.cached_embeddings[tgt].unsqueeze(0).expand(len(neighbors), -1)
        
        pairs = T.cat([nbr_embs, tgt_emb], dim=-1)
        probs = self.actor(pairs)
        log_probs = T.log(probs + 1e-10)
        return probs, log_probs

    def get_value(self, state_node: int, target_node: int) -> T.Tensor:
        """Estimates state value v(u, u_tgt)."""
        if self.cached_embeddings is None:
            self.embed_graph()
            
        u_emb = self.cached_embeddings[state_node]
        tgt_emb = self.cached_embeddings[target_node]
        pair = T.cat([u_emb, tgt_emb], dim=-1).unsqueeze(0)
        return self.critic(pair)

    def select_action(self, obs: Observation) -> int:
        """Selects next hop neighbor during evaluation or rollout."""
        with T.no_grad():
            probs, _ = self.get_action_distribution(obs)
            if self.greedy_mode:
                action_idx = int(T.argmax(probs).item())
            else:
                action_idx = int(T.multinomial(probs, 1).item())
        return obs.neighbors[action_idx]

    def save_checkpoint(self, filepath: str) -> None:
        """Saves weights and optimizer state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        T.save({
            "gnn": self.gnn.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads weights and optimizer state."""
        ckpt = T.load(filepath, map_location=self.device)
        self.gnn.load_state_dict(ckpt["gnn"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.cached_embeddings = None


class MLPA2CAgent(BaseWalker):
    """Multi-Layer Perceptron A2C baseline using raw attributes (without GAT message-passing)."""
    def __init__(
        self,
        env: GraphNavEnv,
        hidden_dim: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 1e-3,
        use_degree: bool = False,
        device: str = "cpu",
        name: str = "MLPA2C",
    ):
        super().__init__(name=name)
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_degree = use_degree
        self.device = T.device(device)
        self.train_mode = True
        self.greedy_mode = False

        # Raw features
        feats = env.features_tensor.to(self.device)
        if use_degree:
            degrees = env.degrees_scaled_tensor.unsqueeze(-1).to(self.device)
            self.node_features = T.cat([feats, degrees], dim=-1)
        else:
            self.node_features = feats
            
        node_dim = self.node_features.shape[1]
        in_actor_critic = 2 * node_dim
        
        self.actor = ActorNetwork(in_dim=in_actor_critic, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(in_dim=in_actor_critic, hidden_dim=hidden_dim).to(self.device)
        
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = T.optim.Adam(params, lr=lr)

    def reset(self, source: int, target: int) -> None:
        pass

    def train(self, mode: bool = True) -> "MLPA2CAgent":
        """Switches between training (stochastic) and evaluation (greedy) modes."""
        self.train_mode = mode
        self.greedy_mode = not mode
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def eval(self) -> "MLPA2CAgent":
        """Sets the agent to greedy evaluation mode."""
        return self.train(False)

    def get_action_distribution(self, obs: Observation) -> Tuple[T.Tensor, T.Tensor]:
        nbr_feats = self.node_features[obs.neighbors]
        tgt_feat = self.node_features[obs.target_node].unsqueeze(0).expand(len(obs.neighbors), -1)
        pairs = T.cat([nbr_feats, tgt_feat], dim=-1)
        probs = self.actor(pairs)
        log_probs = T.log(probs + 1e-10)
        return probs, log_probs

    def get_value(self, state_node: int, target_node: int) -> T.Tensor:
        u_feat = self.node_features[state_node]
        tgt_feat = self.node_features[target_node]
        pair = T.cat([u_feat, tgt_feat], dim=-1).unsqueeze(0)
        return self.critic(pair)

    def select_action(self, obs: Observation) -> int:
        with T.no_grad():
            probs, _ = self.get_action_distribution(obs)
            if self.greedy_mode:
                action_idx = int(T.argmax(probs).item())
            else:
                action_idx = int(T.multinomial(probs, 1).item())
        return obs.neighbors[action_idx]

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        T.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        ckpt = T.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
