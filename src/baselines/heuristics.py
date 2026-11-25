"""Decentralized heuristic graph walkers.

Implements benchmark walkers from Pisacane et al. (LoG 2024):
    - RandomWalker: Uniform random neighbor selection.
    - GreedyWalker: Deterministic Euclidean distance minimization to target attributes.
    - DistanceWalker: SoftMax distribution over attribute distances with temperature tau.
    - ConnectionWalker: SoftMax distribution over neighbor node degrees with temperature tau.
"""
import random
from typing import Optional, List
import numpy as np
import torch as T
import torch.nn.functional as F

from src.env.graph_env import Observation


class BaseWalker:
    """Base class for decentralized routing agents."""
    def __init__(self, name: str = "BaseWalker"):
        self.name = name

    def reset(self, source: int, target: int) -> None:
        """Called at the beginning of each episode."""
        pass

    def select_action(self, obs: Observation) -> int:
        raise NotImplementedError


class RandomWalker(BaseWalker):
    """Selects an immediate neighbor uniformly at random: pi(u'|u) = 1 / deg(u)."""
    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="RandomWalker")
        self.rng = random.Random(seed)

    def select_action(self, obs: Observation) -> int:
        return self.rng.choice(obs.neighbors)


class GreedyWalker(BaseWalker):
    """Deterministic walker picking neighbor with smallest Euclidean distance to target."""
    def __init__(self):
        super().__init__(name="GreedyWalker")

    def select_action(self, obs: Observation) -> int:
        # Distance between each neighbor feature and target feature
        target_feat = obs.target_features.unsqueeze(0)  # Shape (1, d)
        nbr_feats = obs.neighbor_features             # Shape (|N(u)|, d)
        dists = T.norm(nbr_feats - target_feat, p=2, dim=1)  # Shape (|N(u)|,)
        best_idx = int(T.argmin(dists).item())
        return obs.neighbors[best_idx]


class DistanceWalker(BaseWalker):
    """Stochastic walker with SoftMax distribution over negative attribute distances.
    
    pi(u'|u) ~ exp(-||x_u' - x_tgt||_2 / tau)
    """
    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None):
        super().__init__(name=f"DistanceWalker(tau={temperature})")
        self.temperature = max(temperature, 1e-6)
        self.rng = np.random.RandomState(seed)

    def select_action(self, obs: Observation) -> int:
        target_feat = obs.target_features.unsqueeze(0)
        nbr_feats = obs.neighbor_features
        dists = T.norm(nbr_feats - target_feat, p=2, dim=1)
        
        logits = -dists / self.temperature
        probs = F.softmax(logits, dim=0).cpu().numpy()
        
        # Guard against NaNs
        if np.isnan(probs).any() or np.sum(probs) == 0:
            return random.choice(obs.neighbors)
            
        probs = probs / np.sum(probs)
        chosen_idx = self.rng.choice(len(obs.neighbors), p=probs)
        return obs.neighbors[chosen_idx]


class ConnectionWalker(BaseWalker):
    """Stochastic walker with SoftMax distribution over neighbor node degrees (hubs).
    
    pi(u'|u) ~ exp(deg(u') / tau)
    """
    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None):
        super().__init__(name=f"ConnectionWalker(tau={temperature})")
        self.temperature = max(temperature, 1e-6)
        self.rng = np.random.RandomState(seed)

    def select_action(self, obs: Observation) -> int:
        degrees = obs.neighbor_degrees
        logits = degrees / self.temperature
        probs = F.softmax(logits, dim=0).cpu().numpy()
        
        if np.isnan(probs).any() or np.sum(probs) == 0:
            return random.choice(obs.neighbors)
            
        probs = probs / np.sum(probs)
        chosen_idx = self.rng.choice(len(obs.neighbors), p=probs)
        return obs.neighbors[chosen_idx]
