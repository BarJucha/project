"""Memory-augmented decentralized graph navigation agents (Phase 2).

Implements two history-representation paradigms on top of the GARDEN GAT encoder:
    - GRUMemoryAgent: Sequential / order-based memory (inspired by Shen et al., NeurIPS 2018).
    - DeepSetsMemoryAgent: Permutation-invariant set-based memory (inspired by Pándy et al., LoG 2022).
"""
import os
from typing import Dict, List, Optional, Tuple
import torch as T
import torch.nn as nn
from torch_geometric import utils as tg_utils
from torch_geometric.data import Batch

from src.baselines.heuristics import BaseWalker
from src.env.graph_env import GraphNavEnv, Observation
from src.models.networks import ActorNetwork, CriticNetwork, GATEncoder


class BaseMemoryAgent(BaseWalker):
    """Shared scaffolding for memory-augmented GAT Actor-Critic routing agents."""

    def __init__(
        self,
        env: GraphNavEnv,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        memory_dim: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 1e-3,
        use_degree: bool = True,
        freeze_gat: bool = False,
        device: str = "cpu",
        name: str = "MemoryAgent",
    ):
        super().__init__(name=name)
        self.env = env
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim
        self.memory_dim = memory_dim
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.use_degree = use_degree
        self.freeze_gat = freeze_gat
        self.device = T.device(device)
        self.train_mode = True
        self.greedy_mode = False

        ego_pyg_list = [ego.pyg_data for ego in env.ego_graphs]
        self.batch_data = Batch.from_data_list(ego_pyg_list).to(self.device)
        in_channels = ego_pyg_list[0].x.shape[1]

        self.gnn = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=rep_dim,
            num_layers=3,
        ).to(self.device)

        if self.freeze_gat:
            for param in self.gnn.parameters():
                param.requires_grad_(False)
            self.gnn.eval()

        self.node_emb_dim = rep_dim + (1 if use_degree else 0)
        in_actor_critic = 2 * self.node_emb_dim + memory_dim

        self.actor = ActorNetwork(in_dim=in_actor_critic, hidden_dim=hidden_dim).to(self.device)
        self.critic = CriticNetwork(in_dim=in_actor_critic, hidden_dim=hidden_dim).to(self.device)

        self.degrees_scaled = env.degrees_scaled_tensor.unsqueeze(-1).to(self.device)
        self.cached_embeddings: Optional[T.Tensor] = None
        self.optimizer: Optional[T.optim.Optimizer] = None

    def _memory_modules(self) -> Dict[str, nn.Module]:
        """Returns memory submodules to include in optimization and checkpoints."""
        raise NotImplementedError

    def _init_optimizer(self) -> None:
        params: List[nn.Parameter] = []
        if not self.freeze_gat:
            params.extend(self.gnn.parameters())
        for mod in self._memory_modules().values():
            params.extend(mod.parameters())
        params.extend(self.actor.parameters())
        params.extend(self.critic.parameters())
        self.optimizer = T.optim.Adam(params, lr=self.lr)

    def parameters_list(self) -> List[nn.Parameter]:
        """Returns all trainable parameters managed by the optimizer."""
        params: List[nn.Parameter] = []
        if not self.freeze_gat:
            params.extend(p for p in self.gnn.parameters() if p.requires_grad)
        for mod in self._memory_modules().values():
            params.extend(p for p in mod.parameters() if p.requires_grad)
        params.extend(p for p in self.actor.parameters() if p.requires_grad)
        params.extend(p for p in self.critic.parameters() if p.requires_grad)
        return params

    def load_gat_weights(self, filepath: str, freeze: Optional[bool] = None) -> None:
        """Loads pretrained GAT encoder weights from a GARDEN or memory checkpoint."""
        ckpt = T.load(filepath, map_location=self.device)
        if "gnn" not in ckpt:
            raise KeyError(f"Checkpoint at {filepath} does not contain 'gnn' state_dict.")
        self.gnn.load_state_dict(ckpt["gnn"])
        if freeze is not None:
            self.freeze_gat = freeze
            for param in self.gnn.parameters():
                param.requires_grad_(not freeze)
            if freeze:
                self.gnn.eval()
            self._init_optimizer()
        self.cached_embeddings = None

    def embed_graph(self) -> T.Tensor:
        """Computes continuous latent representations for all nodes in the graph."""
        if self.freeze_gat or not self.train_mode:
            with T.no_grad():
                node_out = self.gnn(self.batch_data.x, self.batch_data.edge_index)
        else:
            node_out = self.gnn(self.batch_data.x, self.batch_data.edge_index)

        graph_embs = tg_utils.scatter(node_out, self.batch_data.batch, dim=0, reduce="mean")
        if self.use_degree:
            graph_embs = T.cat([graph_embs, self.degrees_scaled], dim=-1)

        self.cached_embeddings = graph_embs
        return graph_embs

    def train(self, mode: bool = True) -> "BaseMemoryAgent":
        """Switches between training (stochastic) and evaluation (greedy) modes."""
        self.train_mode = mode
        self.greedy_mode = not mode
        self.gnn.train(mode and not self.freeze_gat)
        for mod in self._memory_modules().values():
            mod.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.cached_embeddings = None
        return self

    def eval(self) -> "BaseMemoryAgent":
        """Sets the agent to greedy evaluation mode."""
        return self.train(False)

    def _reset_memory(self, source: int) -> None:
        raise NotImplementedError

    def _update_memory(self, node_id: int) -> None:
        raise NotImplementedError

    def get_memory_state(self) -> T.Tensor:
        """Returns the current fixed-size latent memory vector z_t of shape (memory_dim,)."""
        raise NotImplementedError

    def reset(self, source: int, target: int) -> None:
        """Resets episode graph cache and initializes memory with the starting node."""
        if (self.train_mode and not self.freeze_gat) or self.cached_embeddings is None:
            self.embed_graph()
        self._reset_memory(int(source))

    def update_memory(self, node_id: int) -> None:
        """Advances internal memory after transitioning to node_id."""
        self._update_memory(int(node_id))

    def get_action_distribution(self, obs: Observation) -> Tuple[T.Tensor, T.Tensor]:
        """Calculates policy probabilities and log-probabilities conditioned on memory z_t."""
        if self.cached_embeddings is None:
            self.embed_graph()

        nbr_embs = self.cached_embeddings[obs.neighbors]
        num_nbrs = len(obs.neighbors)
        tgt_emb = self.cached_embeddings[obs.target_node].unsqueeze(0).expand(num_nbrs, -1)
        mem_emb = self.get_memory_state().unsqueeze(0).expand(num_nbrs, -1)

        inputs = T.cat([nbr_embs, tgt_emb, mem_emb], dim=-1)
        probs = self.actor(inputs)
        log_probs = T.log(probs + 1e-10)
        return probs, log_probs

    def get_value(
        self,
        state_node: int,
        target_node: int,
        memory_state: Optional[T.Tensor] = None,
    ) -> T.Tensor:
        """Estimates state value v(x_u, x_tgt, z_t)."""
        if self.cached_embeddings is None:
            self.embed_graph()

        u_emb = self.cached_embeddings[state_node]
        tgt_emb = self.cached_embeddings[target_node]
        z_t = memory_state if memory_state is not None else self.get_memory_state()
        pair = T.cat([u_emb, tgt_emb, z_t], dim=-1).unsqueeze(0)
        return self.critic(pair)

    def select_action(self, obs: Observation) -> int:
        """Selects next neighbor and updates memory state for online rollouts/evaluation."""
        with T.no_grad():
            probs, _ = self.get_action_distribution(obs)
            if self.greedy_mode:
                action_idx = int(T.argmax(probs).item())
            else:
                action_idx = int(T.multinomial(probs, 1).item())
            chosen_node = obs.neighbors[action_idx]
            self.update_memory(chosen_node)
        return chosen_node

    def save_checkpoint(self, filepath: str) -> None:
        """Saves GNN, memory, Actor-Critic, and optimizer state."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = {
            "gnn": self.gnn.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "memory": {k: m.state_dict() for k, m in self._memory_modules().items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
        }
        T.save(payload, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Loads weights and optimizer state."""
        ckpt = T.load(filepath, map_location=self.device)
        self.gnn.load_state_dict(ckpt["gnn"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "memory" in ckpt:
            for k, mod in self._memory_modules().items():
                if k in ckpt["memory"]:
                    mod.load_state_dict(ckpt["memory"][k])
        if ckpt.get("optimizer") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.cached_embeddings = None


class GRUMemoryAgent(BaseMemoryAgent):
    """Sequential / order-based memory agent using a GRU over visited node embeddings."""

    def __init__(
        self,
        env: GraphNavEnv,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        memory_dim: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 1e-3,
        use_degree: bool = True,
        freeze_gat: bool = False,
        window_size: Optional[int] = None,
        device: str = "cpu",
        name: str = "GRU_GARDEN",
    ):
        super().__init__(
            env=env,
            hidden_dim=hidden_dim,
            rep_dim=rep_dim,
            memory_dim=memory_dim,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
            use_degree=use_degree,
            freeze_gat=freeze_gat,
            device=device,
            name=name,
        )
        self.window_size = window_size
        self.gru = nn.GRUCell(input_size=self.node_emb_dim, hidden_size=self.memory_dim).to(self.device)
        self._hidden: Optional[T.Tensor] = None
        self._visited_window: List[int] = []
        self._init_optimizer()

    def _memory_modules(self) -> Dict[str, nn.Module]:
        return {"gru": self.gru}

    def _reset_memory(self, source: int) -> None:
        self._visited_window = [source]
        if self.window_size is None:
            h0 = T.zeros(self.memory_dim, device=self.device)
            self._hidden = self.gru(self.cached_embeddings[source], h0)
        else:
            self._hidden = None

    def _update_memory(self, node_id: int) -> None:
        if self.window_size is None:
            if self._hidden is None:
                h0 = T.zeros(self.memory_dim, device=self.device)
                self._hidden = self.gru(self.cached_embeddings[node_id], h0)
            else:
                self._hidden = self.gru(self.cached_embeddings[node_id], self._hidden)
        else:
            self._visited_window.append(node_id)
            if len(self._visited_window) > self.window_size:
                self._visited_window = self._visited_window[-self.window_size:]

    def get_memory_state(self) -> T.Tensor:
        if self.window_size is None:
            if self._hidden is None:
                return T.zeros(self.memory_dim, device=self.device)
            return self._hidden

        if not self._visited_window:
            return T.zeros(self.memory_dim, device=self.device)

        h = T.zeros(self.memory_dim, device=self.device)
        for nid in self._visited_window:
            h = self.gru(self.cached_embeddings[nid], h)
        return h


class DeepSetsMemoryAgent(BaseMemoryAgent):
    """Set-based / permutation-invariant memory agent using Deep Sets mean pooling."""

    def __init__(
        self,
        env: GraphNavEnv,
        hidden_dim: int = 64,
        rep_dim: int = 32,
        memory_dim: int = 32,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 1e-3,
        use_degree: bool = True,
        freeze_gat: bool = False,
        deduplicate: bool = True,
        window_size: Optional[int] = None,
        device: str = "cpu",
        name: str = "DeepSets_GARDEN",
    ):
        super().__init__(
            env=env,
            hidden_dim=hidden_dim,
            rep_dim=rep_dim,
            memory_dim=memory_dim,
            lr=lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
            use_degree=use_degree,
            freeze_gat=freeze_gat,
            device=device,
            name=name,
        )
        self.deduplicate = deduplicate
        self.window_size = window_size

        self.phi = nn.Sequential(
            nn.Linear(self.node_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
            nn.ReLU(),
        ).to(self.device)

        self.rho = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        ).to(self.device)

        self._visited: List[int] = []
        self._init_optimizer()

    def _memory_modules(self) -> Dict[str, nn.Module]:
        return {"phi": self.phi, "rho": self.rho}

    def _reset_memory(self, source: int) -> None:
        self._visited = [source]

    def _update_memory(self, node_id: int) -> None:
        if self.deduplicate:
            if node_id in self._visited:
                if self.window_size is not None:
                    self._visited.remove(node_id)
                    self._visited.append(node_id)
            else:
                self._visited.append(node_id)
        else:
            self._visited.append(node_id)

        if self.window_size is not None and len(self._visited) > self.window_size:
            self._visited = self._visited[-self.window_size:]

    def get_memory_state(self) -> T.Tensor:
        if not self._visited:
            return T.zeros(self.memory_dim, device=self.device)
        visited_embs = self.cached_embeddings[self._visited]
        pooled = self.phi(visited_embs).mean(dim=0)
        return self.rho(pooled)
