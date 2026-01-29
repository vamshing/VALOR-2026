"""Tree-based models for uplift modeling.

This module provides an implementation of the Robust ZILN Forest.
"""

from typing import Optional, List, Any
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor


class ZILNNode:
    """A node in the ZILN Uplift Tree."""

    def __init__(self, depth: int = 0):
        self.depth = depth
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional['ZILNNode'] = None
        self.right: Optional['ZILNNode'] = None
        self.is_leaf: bool = False
        self.value: float = 0.0


class ZILNUpliftTree:
    """A single ZILN Uplift Tree."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 25,
        max_features: Optional[int] = None,
        split_criterion: str = 'ziln',
        use_smoothing: bool = True
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.split_criterion = split_criterion
        self.use_smoothing = use_smoothing
        self.root: Optional[ZILNNode] = None

        # Adaptive Priors
        self.prior_p = 0.0
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        self.alpha_p = 1.0 if use_smoothing else 0.0
        self.alpha_reg = 5.0 if use_smoothing else 0.0

    def _calc_robust_ziln_params(self, y: np.ndarray) -> tuple:
        n = len(y)
        if n == 0:
            return 0.0, 0.0, 1.0

        n_pos = np.sum(y > 0)
        p = (n_pos + self.alpha_p * self.prior_p) / (n + self.alpha_p + 1e-9)

        if n_pos > 1:
            log_y = np.log(np.maximum(y[y > 0], 1e-6))
            sample_mu, sample_sigma = np.mean(log_y), np.std(log_y)
            if self.use_smoothing:
                w = n_pos / (n_pos + self.alpha_reg)
                mu = w * sample_mu + (1 - w) * self.prior_mu
                sigma = w * sample_sigma + (1 - w) * self.prior_sigma
            else:
                mu, sigma = sample_mu, sample_sigma
        else:
            mu, sigma = self.prior_mu, self.prior_sigma

        return p, mu, np.clip(sigma, 0.1, 4.0)

    def _get_uplift(self, y: np.ndarray, t: np.ndarray) -> float:
        pt, mt, st = self._calc_robust_ziln_params(y[t == 1])
        ev_t = pt * np.exp(mt + 0.5 * st**2)
        pc, mc, sc = self._calc_robust_ziln_params(y[t == 0])
        ev_c = pc * np.exp(mc + 0.5 * sc**2)
        return ev_t - ev_c

    def _calculate_split_gain(self, y: np.ndarray, t: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> float:
        if self.split_criterion == 'mse':
            z = y * (2 * t - 1)
            def mse(arr): return 0.0 if len(arr) == 0 else np.var(arr) * len(arr)
            return mse(z[left_mask | right_mask]) - (mse(z[left_mask]) + mse(z[right_mask]))

        y_l, t_l = y[left_mask], t[left_mask]
        y_r, t_r = y[right_mask], t[right_mask]

        if (np.sum(t_l == 1) < 3 or np.sum(t_l == 0) < 3 or
                np.sum(t_r == 1) < 3 or np.sum(t_r == 0) < 3):
            return -1.0

        tau_l, tau_r = self._get_uplift(y_l, t_l), self._get_uplift(y_r, t_r)
        n_l, n_r = len(y_l), len(y_r)
        return (n_l * n_r / (n_l + n_r + 1e-9)) * (tau_l - tau_r)**2

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        pos_mask = y > 0
        if np.sum(pos_mask) > 0:
            self.prior_p = np.mean(pos_mask)
            log_y = np.log(np.maximum(y[pos_mask], 1e-6))
            self.prior_mu, self.prior_sigma = np.mean(log_y), np.std(log_y)
        self.root = self._build_tree(X, y, t, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, t: np.ndarray, depth: int) -> ZILNNode:
        node = ZILNNode(depth)
        if depth >= self.max_depth or len(y) < self.min_samples_leaf:
            node.is_leaf, node.value = True, self._get_uplift(y, t)
            return node

        n_features = X.shape[1]
        n_select = self.max_features if self.max_features else int(np.sqrt(n_features))
        features = np.random.choice(n_features, n_select, replace=False)

        best_gain, best_split = -1.0, None
        for feat_idx in features:
            values = X[:, feat_idx]
            thresholds = np.unique(np.percentile(values, [20, 40, 60, 80]))
            for thresh in thresholds:
                l_mask = values <= thresh
                r_mask = ~l_mask
                if np.sum(l_mask) < self.min_samples_leaf or np.sum(r_mask) < self.min_samples_leaf:
                    continue
                gain = self._calculate_split_gain(y, t, l_mask, r_mask)
                if gain > best_gain:
                    best_gain, best_split = gain, (feat_idx, thresh)

        if best_split is None:
            node.is_leaf, node.value = True, self._get_uplift(y, t)
            return node

        node.feature_idx, node.threshold = best_split
        l_mask = X[:, node.feature_idx] <= node.threshold
        node.left = self._build_tree(X[l_mask], y[l_mask], t[l_mask], depth + 1)
        node.right = self._build_tree(X[~l_mask], y[~l_mask], t[~l_mask], depth + 1)
        return node

    def predict_row(self, row: np.ndarray, node: ZILNNode) -> float:
        if node.is_leaf:
            return node.value
        if row[node.feature_idx] <= node.threshold:
            return self.predict_row(row, node.left)
        return self.predict_row(row, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_row(row, self.root) for row in X])


class ZILNForestLearner:
    """A collection of ZILN Uplift Trees (Random Forest)."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 10, split_criterion: str = 'ziln', use_smoothing: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.use_smoothing = use_smoothing
        self.trees: List[ZILNUpliftTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray):
        def train_tree(_):
            idx = np.random.choice(len(y), len(y), replace=True)
            tree = ZILNUpliftTree(self.max_depth, split_criterion=self.split_criterion, use_smoothing=self.use_smoothing)
            tree.fit(X[idx], y[idx], t[idx])
            return tree

        with ThreadPoolExecutor(max_workers=4) as ex:
            self.trees = list(ex.map(train_tree, range(self.n_estimators)))

    def forward(self, X: Any) -> torch.Tensor:
        if hasattr(X, 'cpu'): X = X.cpu().numpy()
        preds = np.zeros((len(X), len(self.trees)))
        for i, t in enumerate(self.trees):
            preds[:, i] = t.predict(X)
        return torch.tensor(np.mean(preds, axis=1).reshape(-1, 1), dtype=torch.float32)
