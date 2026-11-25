"""Centralized Shortest-Path Oracle (Dijkstra / BFS)."""
from typing import List, Optional
import networkx as nx
from src.env.graph_env import Observation


class ShortestPathOracle:
    """Centralized shortest path oracle with full global visibility.
    
    Serves as the theoretical upper bound for graph search efficiency
    (R_oracle = 1.0, R_trunc = 0.0%).
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.name = "DijkstraOracle"

    def get_shortest_path(self, source: int, target: int) -> List[int]:
        """Returns the exact shortest path between source and target."""
        return nx.shortest_path(self.graph, source=source, target=target)

    def get_shortest_path_length(self, source: int, target: int) -> int:
        """Returns the length (number of edges) of the shortest path."""
        return nx.shortest_path_length(self.graph, source=source, target=target)

    def select_action(self, obs: Observation) -> int:
        """Picks the next optimal node along the shortest path to target."""
        path = nx.shortest_path(self.graph, source=obs.current_node, target=obs.target_node)
        if len(path) > 1:
            return path[1]
        return obs.current_node
