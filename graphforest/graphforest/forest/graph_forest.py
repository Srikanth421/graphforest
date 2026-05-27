import numpy as np
import pandas as pd
from typing import Optional
from joblib import Parallel, delayed

from graphforest.forest.graph_tree import GraphDecisionTreeClassifier


class GraphRandomForestClassifier:
    """
    An ensemble of GraphDecisionTreeClassifiers.

    Each tree sees a bootstrap sample of the data and a random
    subset of features. The graph-aware split criterion is used
    at every node in every tree.

    Drop-in compatible with sklearn's fit/predict/predict_proba API.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 6,
        min_samples_leaf: int = 50,
        max_features: str = "sqrt",
        max_thresholds: int = 32,
        alpha: float = 0.6,
        beta: float = 0.25,
        gamma: float = 0.1,
        lam: float = 0.05,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features     = max_features
        self.max_thresholds   = max_thresholds
        self.alpha        = alpha
        self.beta         = beta
        self.gamma        = gamma
        self.lam          = lam
        self.n_jobs       = n_jobs
        self.random_state = random_state
        self.trees_:    list       = []
        self.classes_:  np.ndarray = np.array([])
        self.n_classes_: int       = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        graph_risk: Optional[np.ndarray] = None,
        cycle_risk: Optional[np.ndarray] = None,
    ) -> "GraphRandomForestClassifier":

        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if graph_risk is None:
            graph_risk = np.zeros(len(y))
        if cycle_risk is None:
            cycle_risk = np.zeros(len(y))

        seeds = np.random.SeedSequence(self.random_state).spawn(self.n_estimators)

        def _fit_one(seed):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y), size=len(y), replace=True)
            tree = GraphDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_thresholds=self.max_thresholds,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                lam=self.lam,
                random_state=int(rng.integers(1e6)),
                classes=self.classes_,
            )
            tree.fit(
                X[idx], y[idx],
                graph_risk=graph_risk[idx],
                cycle_risk=cycle_risk[idx],
            )
            return tree

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(s) for s in seeds
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        avg_proba = np.zeros((n_samples, self.n_classes_))

        for tree in self.trees_:
            tree_proba = tree.predict_proba(X)
            for local_idx, cls in enumerate(tree.classes_):
                global_idxs = np.where(self.classes_ == cls)[0]
                if len(global_idxs) == 0:
                    continue
                global_idx = global_idxs[0]
                avg_proba[:, global_idx] += tree_proba[:, local_idx]

        with np.errstate(invalid='ignore'):
            result = avg_proba / len(self.trees_)
        return np.nan_to_num(result, nan=0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.trees_:
            return np.array([])
        counts = {}
        for tree in self.trees_:
            self._count_splits(tree.root, counts)
        if not counts:
            return np.array([])
        max_feat = max(counts.keys()) + 1
        importances = np.zeros(max_feat)
        for feat_idx, count in counts.items():
            importances[feat_idx] = count
        return importances / importances.sum()

    def _count_splits(self, node, counts: dict):
        if node is None or node.is_leaf:
            return
        counts[node.feature_index] = counts.get(node.feature_index, 0) + 1
        self._count_splits(node.left, counts)
        self._count_splits(node.right, counts)