import networkx as nx
import numpy as np
from typing import Optional


class NeighborhoodRiskEstimator:
    """
    Estimates fraud risk for a node based on its local
    neighborhood in the directed graph.

    Captures ring/mule patterns that individual node
    features miss — e.g. a clean node surrounded by
    fraud nodes is itself high risk.
    """

    def __init__(self, hops: int = 2, decay: float = 0.5):
        """
        hops  : how many hops out to look (1 = immediate neighbors,
                2 = neighbors of neighbors)
        decay : weight discount per hop (0.5 means 2-hop neighbors
                count half as much as 1-hop neighbors)
        """
        self.hops = hops
        self.decay = decay
        self._fraud_nodes: set = set()

    def fit(self, fraud_nodes: set) -> "NeighborhoodRiskEstimator":
        """Register the set of known fraud node IDs."""
        self._fraud_nodes = set(fraud_nodes)
        return self

    def score(self, G: nx.DiGraph, node) -> float:
        """
        Returns a risk score in [0, 1] based on weighted
        proximity to known fraud nodes.

        Score = sum over hops h of:
            decay^h * (fraud neighbors at hop h / total neighbors at hop h)
        Normalised so max possible = 1.0
        """
        if node not in G or not self._fraud_nodes:
            return 0.0

        total_score = 0.0
        max_score = 0.0
        visited = {node}

        frontier = {node}
        for h in range(1, self.hops + 1):
            next_frontier = set()
            for n in frontier:
                next_frontier |= set(G.successors(n)) | set(G.predecessors(n))
            next_frontier -= visited

            if not next_frontier:
                break

            hop_weight = self.decay ** h
            fraud_count = len(next_frontier & self._fraud_nodes)
            total_score += hop_weight * (fraud_count / len(next_frontier))
            max_score += hop_weight

            visited |= next_frontier
            frontier = next_frontier

        return total_score / max_score if max_score > 0 else 0.0

    def score_nodes(self, G: nx.DiGraph, nodes: list) -> np.ndarray:
        """Score a list of nodes, returns a numpy array."""
        return np.array([self.score(G, n) for n in nodes])