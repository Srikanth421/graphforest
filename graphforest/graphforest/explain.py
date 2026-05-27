import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional

from graphforest.graph_features import compute_all_features
from graphforest.neighborhood import NeighborhoodRiskEstimator
from graphforest.classifier import FRAUD_TYPES


class GraphForestExplainer:
    """
    Produces human-readable explanations for a single transaction.

    Explanation has three parts:
      1. tabular_rules   — top RF feature importances for this prediction
      2. graph_context   — directed graph features for the source node
      3. neighbor_context — neighborhood risk and nearby fraud nodes
    """

    def __init__(self, model):
        """Pass a fitted DirectedGraphForestClassifier."""
        self._model = model

    def explain(self, df: pd.DataFrame, row_index) -> dict:
        """
        Explain a single transaction identified by its DataFrame index.

        Returns a dict with:
          - transaction     : raw transaction values
          - predicted_class : integer class
          - predicted_type  : human-readable fraud type
          - class_probas    : probability per class
          - graph_context   : all directed graph features for source node
          - neighbor_context: neighborhood risk score + nearby fraud nodes
          - tabular_rules   : top 5 tabular feature importances (if applicable)
        """
        model = self._model
        assert model._fitted, "Model must be fitted before explaining."

        row = df.loc[row_index]
        single = df.loc[[row_index]]

        # Predicted class and probabilities
        proba = model.predict_proba(single)[0]
        pred_class = model.classes_[np.argmax(proba)]
        pred_type = FRAUD_TYPES.get(int(pred_class), "unknown")
        class_probas = {
            FRAUD_TYPES.get(int(c), str(c)): round(float(p), 4)
            for c, p in zip(model.classes_, proba)
        }

        # Graph context for source node
        source_node = row[model.source_col]
        graph_feats = compute_all_features(
            model._G, source_node, fraud_nodes=model._fraud_nodes
        )
        graph_context = {k: round(float(v), 4) for k, v in graph_feats.items()}

        # Neighborhood context
        neighborhood_score = model._neighborhood.score(model._G, source_node)
        nearby_fraud = self._nearby_fraud_nodes(model._G, source_node, model._fraud_nodes)

        neighbor_context = {
            "neighborhood_risk_score": round(float(neighborhood_score), 4),
            "nearby_fraud_nodes": nearby_fraud,
            "hops_checked": model._neighborhood.hops,
        }

        # Tabular rules (feature importances from tabular RF)
        tabular_rules = []
        if model.tabular_features:
            importances = model._tabular_rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            tabular_rules = [
                {
                    "feature": model.tabular_features[i],
                    "importance": round(float(importances[i]), 4),
                    "value": round(float(row[model.tabular_features[i]]), 4),
                }
                for i in top_idx
            ]

        # Graph RF top features
        graph_importance = model._graph_rf.feature_importances_
        graph_feature_names = model.graph_features
        top_graph_idx = np.argsort(graph_importance)[::-1][:5]
        graph_rules = [
            {
                "feature": graph_feature_names[i],
                "importance": round(float(graph_importance[i]), 4),
                "value": graph_context.get(graph_feature_names[i], 0.0),
            }
            for i in top_graph_idx
        ]

        return {
            "transaction": row.to_dict(),
            "predicted_class": int(pred_class),
            "predicted_type": pred_type,
            "class_probas": class_probas,
            "graph_context": graph_context,
            "neighbor_context": neighbor_context,
            "tabular_rules": tabular_rules,
            "graph_rules": graph_rules,
        }

    def _nearby_fraud_nodes(
        self, G: nx.DiGraph, node, fraud_nodes: set, hops: int = 2
    ) -> list:
        """Return fraud nodes within `hops` of the given node."""
        visited = {node}
        frontier = {node}
        found = []

        for h in range(1, hops + 1):
            next_frontier = set()
            for n in frontier:
                next_frontier |= set(G.successors(n)) | set(G.predecessors(n))
            next_frontier -= visited

            for n in next_frontier:
                if n in fraud_nodes:
                    found.append({"node": n, "hops_away": h})

            visited |= next_frontier
            frontier = next_frontier

        return found