import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from graphforest.graph_builder import TransactionGraphBuilder
from graphforest.graph_features import compute_all_features
from graphforest.neighborhood import NeighborhoodRiskEstimator


FRAUD_TYPES = {
    0: "normal",
    1: "friendly_fraud",
    2: "identity_fraud",
    3: "account_takeover",
    4: "mule_ring",
    5: "synthetic_identity",
    6: "merchant_collusion",
}


class DirectedGraphForestClassifier:
    """
    Dual-branch Random Forest with directed graph features
    for explainable fraud detection.

    Branch 1 — TabularRF   : trained on raw transaction features
    Branch 2 — GraphRF     : trained on directed graph features
    Branch 3 — Neighborhood: proximity-to-fraud risk score
    Meta-learner           : logistic regression combining all three
    """

    def __init__(
        self,
        source_col: str,
        target_col: str,
        timestamp_col: str,
        amount_col: str = None,
        tabular_features: list = None,
        graph_features: list = None,
        n_estimators: int = 100,
        hops: int = 2,
        decay: float = 0.5,
        random_state: int = 42,
    ):
        self.source_col = source_col
        self.target_col = target_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.tabular_features = tabular_features or []
        self.graph_features = graph_features or list(self._default_graph_features())
        self.n_estimators = n_estimators
        self.hops = hops
        self.decay = decay
        self.random_state = random_state

        self._builder = None
        self._neighborhood = NeighborhoodRiskEstimator(hops=hops, decay=decay)
        self._tabular_rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
        )
        self._graph_rf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, class_weight="balanced"
        )
        self._meta = LogisticRegression(max_iter=1000, random_state=random_state)
        self._scaler = StandardScaler()
        self._fitted = False
        self.classes_ = None

    @staticmethod
    def _default_graph_features():
        return [
            "in_degree", "out_degree",
            "weighted_in_degree", "weighted_out_degree",
            "pagerank", "hub_score", "authority_score",
            "neighbor_fraud_rate", "cycle_score",
        ]

    def _build_graph_feature_matrix(
        self, df: pd.DataFrame, G: nx.DiGraph, fraud_nodes: set
    ) -> pd.DataFrame:
        from graphforest.graph_features import compute_all_features_batch
        nodes = df[self.source_col].tolist()
        result = compute_all_features_batch(G, nodes, fraud_nodes=fraud_nodes)
        return result[[f for f in self.graph_features if f in result.columns]]

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "DirectedGraphForestClassifier":
        """
        Fit the dual-branch forest + meta-learner.

        df : transaction DataFrame (must include source, target, timestamp cols)
        y  : integer labels (see FRAUD_TYPES)
        """
        self.classes_ = np.unique(y)

        # Build directed graph from training data
        self._builder = TransactionGraphBuilder(
            source_col=self.source_col,
            target_col=self.target_col,
            timestamp_col=self.timestamp_col,
            amount_col=self.amount_col,
        )
        self._builder.fit(df)
        G = self._builder.full_graph()

        # Known fraud nodes = source nodes where y == fraud
        fraud_mask = y > 0
        fraud_nodes = set(df.loc[fraud_mask, self.source_col].unique())
        self._neighborhood.fit(fraud_nodes)

        # Branch 1: tabular RF
        if self.tabular_features:
            X_tab = df[self.tabular_features].fillna(0)
            self._tabular_rf.fit(X_tab, y)
            tab_proba = self._tabular_rf.predict_proba(X_tab)
        else:
            tab_proba = np.zeros((len(df), len(self.classes_)))

        # Branch 2: graph RF
        X_graph = self._build_graph_feature_matrix(df, G, fraud_nodes)
        self._graph_rf.fit(X_graph, y)
        graph_proba = self._graph_rf.predict_proba(X_graph)

        # Branch 3: neighborhood risk (single score per row)
        neighborhood_scores = self._neighborhood.score_nodes(
            G, df[self.source_col].tolist()
        ).reshape(-1, 1)

        # Meta-learner input: concatenate all branch outputs
        meta_input = np.hstack([tab_proba, graph_proba, neighborhood_scores])
        meta_input_scaled = self._scaler.fit_transform(meta_input)
        self._meta.fit(meta_input_scaled, y)

        self._fitted = True
        self._fraud_nodes = fraud_nodes
        self._G = G
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit() first."

        if self.tabular_features:
            X_tab = df[self.tabular_features].fillna(0)
            tab_proba = self._tabular_rf.predict_proba(X_tab)
        else:
            tab_proba = np.zeros((len(df), len(self.classes_)))

        X_graph = self._build_graph_feature_matrix(df, self._G, self._fraud_nodes)
        graph_proba = self._graph_rf.predict_proba(X_graph)

        neighborhood_scores = self._neighborhood.score_nodes(
            self._G, df[self.source_col].tolist()
        ).reshape(-1, 1)

        meta_input = np.hstack([tab_proba, graph_proba, neighborhood_scores])
        meta_input_scaled = self._scaler.transform(meta_input)
        return self._meta.predict_proba(meta_input_scaled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_fraud_type(self, df: pd.DataFrame) -> list:
        """Returns human-readable fraud type labels."""
        preds = self.predict(df)
        return [FRAUD_TYPES.get(int(p), "unknown") for p in preds]