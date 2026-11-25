"""Action-Masked Tabu Walker: Symbolic memory baseline.

Maintains a FIFO buffer of the last k visited node IDs in the message header.
Immediate backward hops and recently visited nodes are masked out (logits = -inf),
directly preventing short action cycles and ping-pong oscillations.
"""
from collections import deque
from typing import Optional, List
import numpy as np
import torch as T
import torch.nn.functional as F

from src.env.graph_env import Observation
from src.baselines.heuristics import BaseWalker


class TabuWalker(BaseWalker):
    """Symbolic memory baseline with action-masked k-step tabu list."""
    def __init__(
        self,
        buffer_size: int = 5,
        decision_mode: str = "greedy",  # 'greedy', 'softmax', or 'random'
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(name=f"TabuWalker(k={buffer_size},{decision_mode})")
        self.buffer_size = buffer_size
        self.decision_mode = decision_mode
        self.temperature = max(temperature, 1e-6)
        self.rng = np.random.RandomState(seed)
        self.tabu_buffer: deque = deque(maxlen=buffer_size)

    def reset(self, source: int, target: int) -> None:
        """Clears the tabu buffer and initializes with source node."""
        self.tabu_buffer.clear()
        self.tabu_buffer.append(source)

    def select_action(self, obs: Observation) -> int:
        u = obs.current_node
        target = obs.target_node
        neighbors = obs.neighbors
        
        # 1. If target is in immediate neighborhood, jump directly to target
        if target in neighbors:
            self.tabu_buffer.append(u)
            return target
            
        # 2. Filter out neighbors currently in tabu buffer
        tabu_set = set(self.tabu_buffer)
        valid_neighbors = [v for v in neighbors if v not in tabu_set]
        
        # If all neighbors are in tabu buffer, fall back to full neighborhood
        candidates = valid_neighbors if valid_neighbors else neighbors
        
        # 3. Decision rule over available candidates
        if self.decision_mode == "random":
            chosen = self.rng.choice(candidates)
        elif self.decision_mode in ("greedy", "softmax"):
            candidate_indices = [neighbors.index(c) for c in candidates]
            nbr_feats = obs.neighbor_features[candidate_indices]
            target_feat = obs.target_features.unsqueeze(0)
            dists = T.norm(nbr_feats - target_feat, p=2, dim=1)
            
            if self.decision_mode == "greedy":
                best_sub_idx = int(T.argmin(dists).item())
                chosen = candidates[best_sub_idx]
            else:  # softmax
                logits = -dists / self.temperature
                probs = F.softmax(logits, dim=0).cpu().numpy()
                if np.isnan(probs).any() or np.sum(probs) == 0:
                    chosen = self.rng.choice(candidates)
                else:
                    probs = probs / np.sum(probs)
                    chosen_idx = self.rng.choice(len(candidates), p=probs)
                    chosen = candidates[chosen_idx]
        else:
            raise ValueError(f"Unknown decision_mode: {self.decision_mode}")
            
        self.tabu_buffer.append(u)
        return chosen
