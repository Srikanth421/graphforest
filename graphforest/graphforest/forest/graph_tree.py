import numpy as np
import pandas as pd
from typing import Optional


class SplitNode:
    """A node in the graph-aware decision tree."""

    def __init__(self):
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["SplitNode"] = None
        self.right: Optional["SplitNode"] = None
        self.is_leaf: bool = False
        self.class_probs: Optional[np.ndarray] = None
        self.n_samples: int = 0
        self.gini_gain: float = 0.0
        self.graph_gain: float = 0.0


class GraphDecisionTreeClassifier:
    """
    A decision tree that uses a graph-aware split criterion.

    At each node, the split score combines:
        score = alpha * gini_gain
              + beta  * graph_risk_separation
              + gamma * cycle_risk_separation
              - lam   * complexity_penalty

    This is the core research contribution of graphforest v0.2.
    Unlike sklearn trees which only optimize Gini/entropy,
    this tree explicitly separates high-risk graph neighborhoods.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 50,
        max_features: str = "sqrt",
        max_thresholds: int = 32,
        alpha: float = 0.6,    # weight for gini gain
        beta: float = 0.25,    # weight for graph risk separation
        gamma: float = 0.1,    # weight for cycle/ring separation
        lam: float = 0.05,     # complexity penalty weight
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.random_state = random_state
        self.root: Optional[SplitNode] = None
        self.n_classes_: int = 0
        self.classes_: np.ndarray = np.array([])
        self._rng = np.random.default_rng(random_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        graph_risk: Optional[np.ndarray] = None,
        cycle_risk: Optional[np.ndarray] = None,
    ) -> "GraphDecisionTreeClassifier":
        """
        X          : feature matrix (n_samples, n_features)
        y          : integer class labels
        graph_risk : per-sample neighborhood risk score [0,1]
        cycle_risk : per-sample cycle/ring score [0,1]
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        y_idx = np.array([self._class_to_idx[c] for c in y])

        if graph_risk is None:
            graph_risk = np.zeros(len(y))
        if cycle_risk is None:
            cycle_risk = np.zeros(len(y))

        self.root = self._grow(
            X, y_idx, graph_risk, cycle_risk, depth=0
        )
        return self

    def _grow(self, X, y, graph_risk, cycle_risk, depth) -> SplitNode:
        node = SplitNode()
        node.n_samples = len(y)
        node.class_probs = self._class_probs(y)

        # Stop conditions
        if (
            depth >= self.max_depth
            or len(y) < 2 * self.min_samples_leaf
            or len(np.unique(y)) == 1
        ):
            node.is_leaf = True
            return node

        # Sample features
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            n_try = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            n_try = max(1, int(np.log2(n_features)))
        else:
            n_try = n_features

        feature_indices = self._rng.choice(n_features, size=n_try, replace=False)

        best_score = -np.inf
        best_feat  = None
        best_thresh = None
        best_left_mask = None

        parent_gini = self._gini(y)

        for feat_idx in feature_indices:
            col = X[:, feat_idx]
            thresholds = self._candidate_thresholds(col)

            for thresh in thresholds:
                left_mask  = col <= thresh
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf:
                    continue
                if right_mask.sum() < self.min_samples_leaf:
                    continue

                score = self._split_score(
                    y, graph_risk, cycle_risk,
                    left_mask, right_mask,
                    parent_gini,
                )

                if score > best_score:
                    best_score     = score
                    best_feat      = feat_idx
                    best_thresh    = thresh
                    best_left_mask = left_mask

        if best_feat is None:
            node.is_leaf = True
            return node

        node.feature_index = best_feat
        node.threshold     = best_thresh

        left_mask  = best_left_mask
        right_mask = ~best_left_mask

        node.left  = self._grow(
            X[left_mask],  y[left_mask],
            graph_risk[left_mask],  cycle_risk[left_mask],
            depth + 1,
        )
        node.right = self._grow(
            X[right_mask], y[right_mask],
            graph_risk[right_mask], cycle_risk[right_mask],
            depth + 1,
        )
        return node

    def _split_score(
        self, y, graph_risk, cycle_risk,
        left_mask, right_mask, parent_gini
    ) -> float:
        n = len(y)
        n_l = left_mask.sum()
        n_r = right_mask.sum()

        # 1. Gini gain
        gini_l = self._gini(y[left_mask])
        gini_r = self._gini(y[right_mask])
        gini_gain = parent_gini - (n_l/n * gini_l + n_r/n * gini_r)

        # 2. Graph risk separation
        # Good split = high-risk samples go left, low-risk go right (or vice versa)
        mean_risk_l = graph_risk[left_mask].mean()
        mean_risk_r = graph_risk[right_mask].mean()
        graph_separation = abs(mean_risk_l - mean_risk_r)

        # 3. Cycle/ring separation
        mean_cycle_l = cycle_risk[left_mask].mean()
        mean_cycle_r = cycle_risk[right_mask].mean()
        cycle_separation = abs(mean_cycle_l - mean_cycle_r)

        # 4. Complexity penalty — discourages tiny unstable splits
        balance = min(n_l, n_r) / max(n_l, n_r)
        complexity_penalty = 1.0 - balance

        score = (
            self.alpha * gini_gain
            + self.beta  * graph_separation
            + self.gamma * cycle_separation
            - self.lam   * complexity_penalty
        )
        return score

    def _candidate_thresholds(self, col: np.ndarray) -> np.ndarray:
        unique_vals = np.unique(col)
        if len(unique_vals) <= self.max_thresholds:
            midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
        else:
            percentiles = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
            midpoints = np.percentile(col, percentiles)
        return midpoints

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        probs = np.bincount(y, minlength=self.n_classes_) / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _class_probs(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / counts.sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(x, self.root) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _traverse(self, x: np.ndarray, node: SplitNode) -> np.ndarray:
        if node.is_leaf:
            return node.class_probs
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)