import pandas as pd
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


VALID_EDGE_TYPES = {
    "account_merchant",
    "device_account",
    "email_account",
    "ip_account",
    "account_beneficiary",
    "card_merchant",
}


@dataclass
class TransactionGraphBuilder:
    """
    Builds a directed transaction graph from a DataFrame.
    Enforces causal integrity: when computing features for a
    transaction at time t, only edges with timestamp < t are used.
    """

    source_col: str
    target_col: str
    timestamp_col: str
    edge_type_col: Optional[str] = None
    amount_col: Optional[str] = None

    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, init=False, repr=False)

    def fit(self, df: pd.DataFrame) -> "TransactionGraphBuilder":
        """Build the full graph from the transaction DataFrame."""
        self._graph = nx.DiGraph()
        for _, row in df.iterrows():
            src = row[self.source_col]
            tgt = row[self.target_col]
            edge_attrs = {
                "timestamp": row[self.timestamp_col],
                "edge_type": row[self.edge_type_col] if self.edge_type_col else "unknown",
                "amount": row[self.amount_col] if self.amount_col else 1.0,
            }
            if self._graph.has_edge(src, tgt):
                self._graph[src][tgt]["weight"] += edge_attrs["amount"]
                self._graph[src][tgt]["count"] += 1
            else:
                self._graph.add_edge(src, tgt, weight=edge_attrs["amount"],
                                     count=1, **edge_attrs)
        return self

    def snapshot(self, before_timestamp) -> nx.DiGraph:
        """
        Return a subgraph containing only edges with
        timestamp strictly before the given value.
        Ensures no future leakage when computing features.
        """
        edges = [
            (u, v) for u, v, d in self._graph.edges(data=True)
            if d["timestamp"] < before_timestamp
        ]
        return self._graph.edge_subgraph(edges).copy()

    def full_graph(self) -> nx.DiGraph:
        """Return the complete graph (use only for training, never for inference)."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()