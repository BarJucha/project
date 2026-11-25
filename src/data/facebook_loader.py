"""Loader and preprocessor for SNAP Facebook ego-networks."""
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import networkx as nx
import numpy as np


# Mapping from Table 4 graph index (1..5) to SNAP ego network ID
FACEBOOK_GRAPH_MAP = {
    1: 414,   # n=148, d=105
    2: 686,   # n=168, d=63
    3: 348,   # n=224, d=161
    4: 0,     # n=324, d=224
    5: 3437,  # n=532, d=262
}


def load_facebook_ego_graph(
    ego_id: int,
    raw_dir: str = "data/raw/facebook",
) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
    """Loads a single SNAP Facebook ego network.
    
    Extracts the Largest Connected Component (LCC), maps node IDs to
    continuous integers 0..n-1, attaches node attributes as 'pos' and 'degree',
    and computes min-max normalized degrees.
    
    Args:
        ego_id: SNAP ego network ID (e.g. 0, 348, 414, 686, 3437).
        raw_dir: Directory containing extracted SNAP facebook files.
        
    Returns:
        g: NetworkX graph (LCC) with node labels 0..n-1 and node attributes.
        features: numpy array of shape (n, d) of binary attribute vectors.
        node_id_map: Mapping from original SNAP node IDs to new continuous indices.
    """
    path = Path(raw_dir) / str(ego_id)
    edges_file = f"{path}.edges"
    feat_file = f"{path}.feat"
    
    if not Path(edges_file).exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")
    if not Path(feat_file).exists():
        raise FileNotFoundError(f"Feature file not found: {feat_file}")
        
    # Read edges
    edges = []
    with open(edges_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
                
    g_raw = nx.Graph(edges)
    
    # Read attributes from .feat (each row: node_id f_0 f_1 ... f_{d-1})
    feat_dict: Dict[int, List[float]] = {}
    with open(feat_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                node_id = int(parts[0])
                feat_dict[node_id] = [float(x) for x in parts[1:]]
                
    # Extract Largest Connected Component (LCC)
    lcc_nodes = max(nx.connected_components(g_raw), key=len)
    g_lcc = g_raw.subgraph(lcc_nodes).copy()
    
    # Relabel nodes to contiguous integers 0..n-1
    # Sorted by original ID for deterministic mapping
    sorted_nodes = sorted(list(g_lcc.nodes()))
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_nodes)}
    g = nx.relabel_nodes(g_lcc, node_id_map)
    
    n = g.number_of_nodes()
    # Build aligned feature matrix
    d = len(next(iter(feat_dict.values())))
    features = np.zeros((n, d), dtype=np.float32)
    for new_id, old_id in enumerate(sorted_nodes):
        if old_id not in feat_dict:
            raise ValueError(f"Missing attributes for node {old_id} in ego {ego_id}")
        features[new_id] = feat_dict[old_id]
        
    # Calculate degree and min-max scaled degree
    degrees = dict(g.degree())
    min_deg = min(degrees.values())
    max_deg = max(degrees.values())
    deg_range = (max_deg - min_deg) if max_deg > min_deg else 1.0
    
    for u in g.nodes():
        deg = degrees[u]
        scaled_deg = float((deg - min_deg) / deg_range)
        g.nodes[u]["pos"] = features[u]
        g.nodes[u]["degree"] = deg
        g.nodes[u]["degree_scaled"] = scaled_deg
        
    return g, features, node_id_map
