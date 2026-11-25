"""Kaiser-Hilgetag spatial growth model for synthetic homophilic graphs.

Reference:
    Marcus Kaiser and Claus C. Hilgetag. Spatial growth of real-world networks.
    Physical Review E, 69(3):036103, 2004.
"""
import math
import random
from typing import Tuple, Optional
import networkx as nx
import numpy as np


def generate_kaiser_hilgetag_graph(
    n: int = 200,
    alpha: float = 30.0,
    beta: float = 0.4,
    domain: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, np.ndarray]:
    """Generates a synthetic spatial homophilic graph according to Kaiser & Hilgetag (2004).
    
    Nodes are sequentially placed in the spatial domain [xmin, xmax] x [ymin, ymax].
    Probability of connection between candidate node i and existing node j:
        p(i, j) = beta * exp(-alpha * dist(i, j))
    A node is added only when it connects to at least one existing node,
    ensuring graph connectivity to the root node 0.
    
    Args:
        n: Number of nodes to generate.
        alpha: Distance decay coefficient controlling homophily sharpness.
        beta: Base edge probability coefficient controlling density.
        domain: Spatial domain bounds (xmin, ymin, xmax, ymax).
        seed: Random seed for reproducibility.
        
    Returns:
        G: Connected NetworkX graph with node attributes 'pos', 'degree', 'degree_scaled'.
        features: numpy array of shape (n, 2) containing 2D spatial positions.
    """
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = domain
    
    G = nx.Graph()
    xcenter, ycenter = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    G.add_node(0, pos=np.array([xcenter, ycenter], dtype=np.float32))
    
    positions = [np.array([xcenter, ycenter], dtype=np.float32)]
    
    i = 1
    max_attempts_per_node = 10000
    
    while i < n:
        attempts = 0
        connected = False
        while not connected and attempts < max_attempts_per_node:
            attempts += 1
            pos_i = np.array([
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax)
            ], dtype=np.float32)
            
            edges_to_add = []
            for j in range(i):
                pos_j = positions[j]
                dist = float(np.linalg.norm(pos_i - pos_j))
                prob = beta * math.exp(-alpha * dist)
                if rng.random() < prob:
                    edges_to_add.append((i, j))
                    
            if edges_to_add:
                G.add_node(i, pos=pos_i)
                G.add_edges_from(edges_to_add)
                positions.append(pos_i)
                connected = True
                i += 1
                
        if attempts >= max_attempts_per_node:
            # Fallback if density beta is extremely low: connect to nearest neighbor
            pos_i = np.array([
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax)
            ], dtype=np.float32)
            dists = [np.linalg.norm(pos_i - p) for p in positions]
            nearest_j = int(np.argmin(dists))
            G.add_node(i, pos=pos_i)
            G.add_edge(i, nearest_j)
            positions.append(pos_i)
            i += 1

    # Extract LCC just in case
    lcc_nodes = max(nx.connected_components(G), key=len)
    if len(lcc_nodes) < G.number_of_nodes():
        G = G.subgraph(lcc_nodes).copy()
        G = nx.convert_node_labels_to_integers(G)
        
    n_final = G.number_of_nodes()
    features = np.zeros((n_final, 2), dtype=np.float32)
    degrees = dict(G.degree())
    min_deg = min(degrees.values())
    max_deg = max(degrees.values())
    deg_range = (max_deg - min_deg) if max_deg > min_deg else 1.0
    
    for u in G.nodes():
        pos_u = G.nodes[u]["pos"]
        features[u] = pos_u
        deg = degrees[u]
        G.nodes[u]["degree"] = deg
        G.nodes[u]["degree_scaled"] = float((deg - min_deg) / deg_range)
        
    return G, features
