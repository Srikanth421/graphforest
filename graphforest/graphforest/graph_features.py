import networkx as nx
import numpy as np
import pandas as pd
from typing import Optional


def in_degree(G: nx.DiGraph, node) -> int:
    """Number of incoming edges to this node."""
    return G.in_degree(node) if node in G else 0


def out_degree(G: nx.DiGraph, node) -> int:
    """Number of outgoing edges from this node."""
    return G.out_degree(node) if node in G else 0


def weighted_in_degree(G: nx.DiGraph, node) -> float:
    """Sum of incoming edge weights (e.g. total received amount)."""
    if node not in G:
        return 0.0
    return sum(d.get("weight", 1.0) for _, _, d in G.in_edges(node, data=True))


def weighted_out_degree(G: nx.DiGraph, node) -> float:
    """Sum of outgoing edge weights (e.g. total sent amount)."""
    if node not in G:
        return 0.0
    return sum(d.get("weight", 1.0) for _, _, d in G.out_edges(node, data=True))


def pagerank_score(G: nx.DiGraph, node, alpha: float = 0.85) -> float:
    """
    Directed PageRank — authority flows with money.
    High score = many high-value nodes send to this node.
    """
    if node not in G or G.number_of_edges() == 0:
        return 0.0
    pr = nx.pagerank(G, alpha=alpha, weight="weight")
    return pr.get(node, 0.0)


def hub_score(G: nx.DiGraph, node) -> float:
    """
    HITS hub score — high score means this node sends
    to many authoritative nodes. Useful for detecting
    accounts that fan out to many merchants.
    """
    if node not in G or G.number_of_edges() == 0:
        return 0.0
    hubs, _ = nx.hits(G, max_iter=100)
    return hubs.get(node, 0.0)


def authority_score(G: nx.DiGraph, node) -> float:
    """
    HITS authority score — high score means this node
    is pointed to by many hub nodes. Useful for flagging
    merchants receiving from many accounts.
    """
    if node not in G or G.number_of_edges() == 0:
        return 0.0
    _, authorities = nx.hits(G, max_iter=100)
    return authorities.get(node, 0.0)


def neighbor_fraud_rate(G: nx.DiGraph, node, fraud_nodes: set) -> float:
    """
    Fraction of immediate neighbors (in + out) that are known fraud nodes.
    Requires a set of confirmed fraud node IDs.
    """
    if node not in G:
        return 0.0
    neighbors = set(G.predecessors(node)) | set(G.successors(node))
    if not neighbors:
        return 0.0
    return len(neighbors & fraud_nodes) / len(neighbors)


def cycle_score(G: nx.DiGraph, node) -> float:
    """
    Returns 1.0 if the node participates in any directed cycle
    (money carousel / ring pattern), else 0.0.
    Uses simple cycle detection on the node's local subgraph.
    """
    if node not in G:
        return 0.0
    neighbors = set(G.successors(node)) | set(G.predecessors(node)) | {node}
    subgraph = G.subgraph(neighbors)
    try:
        cycles = list(nx.simple_cycles(subgraph))
        return 1.0 if any(node in c for c in cycles) else 0.0
    except Exception:
        return 0.0


def compute_all_features(
    G: nx.DiGraph,
    node,
    fraud_nodes: Optional[set] = None,
) -> dict:
    """
    Compute all directed graph features for a single node.
    Returns a flat dict ready to be added as a DataFrame row.
    """
    if fraud_nodes is None:
        fraud_nodes = set()
    return {
        "in_degree":           in_degree(G, node),
        "out_degree":          out_degree(G, node),
        "weighted_in_degree":  weighted_in_degree(G, node),
        "weighted_out_degree": weighted_out_degree(G, node),
        "pagerank":            pagerank_score(G, node),
        "hub_score":           hub_score(G, node),
        "authority_score":     authority_score(G, node),
        "neighbor_fraud_rate": neighbor_fraud_rate(G, node, fraud_nodes),
        "cycle_score":         cycle_score(G, node),
    }

def compute_all_features_batch(G: nx.DiGraph, nodes: list, fraud_nodes: set = None) -> pd.DataFrame:
    """
    Compute graph features for all nodes at once.
    Much faster than calling compute_all_features per row.
    """
    if fraud_nodes is None:
        fraud_nodes = set()

    # Compute expensive metrics once for entire graph
    if G.number_of_edges() > 0:
        pr    = nx.pagerank(G, alpha=0.85, weight="weight")
        hubs, auths = nx.hits(G, max_iter=100)
    else:
        pr = {}; hubs = {}; auths = {}

    rows = []
    for node in nodes:
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        fraud_neighbors = len(neighbors & fraud_nodes)
        neighbor_fraud  = fraud_neighbors / len(neighbors) if neighbors else 0.0

        # Cycle score only on local subgraph
        local = G.subgraph(neighbors | {node})
        try:
            cycles = list(nx.simple_cycles(local))
            cyc = 1.0 if any(node in c for c in cycles) else 0.0
        except Exception:
            cyc = 0.0

        rows.append({
            "in_degree":           G.in_degree(node)  if node in G else 0,
            "out_degree":          G.out_degree(node) if node in G else 0,
            "weighted_in_degree":  sum(d.get("weight",1) for _,_,d in G.in_edges(node,  data=True)) if node in G else 0.0,
            "weighted_out_degree": sum(d.get("weight",1) for _,_,d in G.out_edges(node, data=True)) if node in G else 0.0,
            "pagerank":            pr.get(node, 0.0),
            "hub_score":           hubs.get(node, 0.0),
            "authority_score":     auths.get(node, 0.0),
            "neighbor_fraud_rate": neighbor_fraud,
            "cycle_score":         cyc,
        })

    return pd.DataFrame(rows)