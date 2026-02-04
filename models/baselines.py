"""Baseline models for uplift modeling.

This module provides implementations for S-Learner, T-Learner, TARNet, DragonNet,
CFRNet (MMD/Wasserstein), UniTE, and EUEN.
"""

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from VALOR.models.layers import MMOE


class Net(nn.Module):
    """Simple multi-layer perceptron block."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, use_gpu: bool = False, decrease: bool = False):
        super(Net, self).__init__()
        if decrease:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, out_dim),
            )
        if use_gpu:
            self.net.to(torch.device("cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_data_with_treatment_type(data: torch.Tensor, treatment: torch.Tensor):
    """Splits data into treatment and control groups."""
    treatment = treatment.squeeze()
    mask = (treatment == 1)
    data_treated = data[mask]
    data_control = data[~mask]
    return data_treated, data_control


# IPM Helpers for CFRNet
def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    dist = torch.abs(sample_1.unsqueeze(1).expand(n_1, n_2, -1) -
                     sample_2.unsqueeze(0).expand(n_1, n_2, -1))
    dist = torch.pow(dist, norm)
    dist = torch.sum(dist, dim=2)
    return torch.sqrt(dist + eps)

def maximum_mean_discrepancy_loss(sample_1, sample_2, scale=[0.1, 1.0, 10.0]):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    if n_1 == 0 or n_2 == 0:
        return torch.tensor(0.0).to(sample_1.device)

    def gaussian_kernel(x, y, sigma):
        dist = pdist(x, y)
        return torch.exp(-dist**2 / (2 * sigma**2))

    loss = 0
    for s in scale:
        loss += torch.mean(gaussian_kernel(sample_1, sample_1, s))
        loss += torch.mean(gaussian_kernel(sample_2, sample_2, s))
        loss -= 2 * torch.mean(gaussian_kernel(sample_1, sample_2, s))
    return loss

def wasserstein_distance(X_treat, X_control, p=0.5, lam=1, iterations=10):
    """
    Computes Wasserstein distance (Sinkhorn algorithm) for CFRNet-WASS.
    """
    n_t = X_treat.size(0)
    n_c = X_control.size(0)
    if n_t == 0 or n_c == 0:
        return torch.tensor(0.0).to(X_treat.device)

    dist_mat = torch.cdist(X_treat, X_control, p=2) ** 2
    a = p * torch.ones((n_t, 1), device=X_treat.device) / n_t
    b = (1 - p) * torch.ones((n_c, 1), device=X_treat.device) / n_c
    K = torch.exp(-lam * dist_mat)
    K_tilde = K / a
    u = a
    for _ in range(iterations):
        v = b / (torch.matmul(K.t(), u) + 1e-10)
        u = 1.0 / (torch.matmul(K_tilde, v) + 1e-10)
    T = u * K * v.t()
    E = T * dist_mat
    return 2 * torch.sum(E)


class TLearner(nn.Module):
    """T-Learner baseline."""

    def __init__(self, config: Any):
        super().__init__()
        self.t0_net = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.t1_net = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.criterion = nn.MSELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        feat_t, feat_c = get_data_with_treatment_type(features, treatment)
        lab_t, lab_c = get_data_with_treatment_type(labels, treatment)
        loss = 0
        if feat_c.size(0) > 0: loss += self.criterion(self.t0_net(feat_c).squeeze(), lab_c.float())
        if feat_t.size(0) > 0: loss += self.criterion(self.t1_net(feat_t).squeeze(), lab_t.float())
        return loss

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        y0 = self.t0_net(features)
        y1 = self.t1_net(features)
        return y1 - y0


class TARNet(nn.Module):
    """TARNet baseline."""

    def __init__(self, config: Any):
        super().__init__()
        self.rep = Net(config.input_dim, config.input_dim, config.rep_hidden_dim, config.use_gpu)
        self.h1 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.h0 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.criterion = nn.MSELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        rep_t, rep_c = get_data_with_treatment_type(rep, treatment)
        lab_t, lab_c = get_data_with_treatment_type(labels, treatment)

        if rep_c.size(0) > 0:
            y0 = self.h0(rep_c).squeeze()
            loss_h0 = self.criterion(y0, lab_c.float())
        else:
            loss_h0 = torch.tensor(0.0, device=features.device)

        if rep_t.size(0) > 0:
            y1 = self.h1(rep_t).squeeze()
            loss_h1 = self.criterion(y1, lab_t.float())
        else:
            loss_h1 = torch.tensor(0.0, device=features.device)

        if (rep_c.size(0) > 0) and (rep_t.size(0) > 0):
            pred_loss = (loss_h0 + loss_h1) / 2
        else:
            pred_loss = loss_h0 + loss_h1
        return pred_loss

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        return self.h1(rep) - self.h0(rep)


class CFRNet(nn.Module):
    """CFRNet baseline."""

    def __init__(self, config: Any):
        super(CFRNet, self).__init__()
        input_dim = config.input_dim
        hidden_dim_rep = config.rep_hidden_dim
        hidden_dim_hypo = config.hidden_dim
        self.ipm_function = getattr(config, 'ipm_function', 'mmd')
        self.alpha = config.alpha
        self.rep = Net(input_dim, input_dim, hidden_dim_rep, config.use_gpu)
        self.h1 = Net(input_dim, 1, hidden_dim_hypo, config.use_gpu, True)
        self.h0 = Net(input_dim, 1, hidden_dim_hypo, config.use_gpu, True)
        self.criterion = nn.MSELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        rep_t, rep_c = get_data_with_treatment_type(rep, treatment)
        lab_t, lab_c = get_data_with_treatment_type(labels, treatment)

        if rep_c.size(0) > 0:
            loss_h0 = self.criterion(self.h0(rep_c).squeeze(), lab_c.float())
        else:
            loss_h0 = torch.tensor(0.0, device=features.device)

        if rep_t.size(0) > 0:
            loss_h1 = self.criterion(self.h1(rep_t).squeeze(), lab_t.float())
        else:
            loss_h1 = torch.tensor(0.0, device=features.device)

        if (rep_c.size(0) > 0) and (rep_t.size(0) > 0):
            pred_loss = (loss_h0 + loss_h1) / 2
        else:
            pred_loss = loss_h0 + loss_h1

        ipm_loss = torch.tensor(0.0, device=features.device)
        if (rep_c.size(0) > 0) and (rep_t.size(0) > 0):
            if self.ipm_function == "mmd":
                ipm_loss = maximum_mean_discrepancy_loss(rep_t, rep_c)
            elif self.ipm_function == "wasserstein":
                ipm_loss = wasserstein_distance(rep_t, rep_c)
        return pred_loss + (self.alpha * ipm_loss)

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        return self.h1(rep) - self.h0(rep)


class DragonNet(nn.Module):
    """DragonNet baseline."""

    def __init__(self, config: Any):
        super().__init__()
        self.rep = Net(config.input_dim, config.input_dim, config.rep_hidden_dim, config.use_gpu)
        self.treat_out = nn.Linear(config.input_dim, 1)
        self.y0 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.y1 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.epsilon = nn.Linear(1, 1)

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        t_pred = torch.sigmoid(self.treat_out(rep))
        y0_pred = self.y0(rep)
        y1_pred = self.y1(rep)
        loss_t = F.binary_cross_entropy(t_pred.squeeze(), treatment.float())
        mask = treatment.squeeze().bool()
        loss_y = 0
        if (~mask).sum() > 0: loss_y += torch.sum(((labels[~mask] - y0_pred[~mask].squeeze())**2))
        if mask.sum() > 0: loss_y += torch.sum(((labels[mask] - y1_pred[mask].squeeze())**2))
        return loss_y + loss_t

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        return self.y1(rep) - self.y0(rep)


class CFRNetWass(nn.Module):
    """Same as CFRNet but uses Wasserstein Distance instead of MMD."""
    def __init__(self, config: Any):
        super().__init__()
        self.alpha = config.alpha
        self.rep = Net(config.input_dim, config.input_dim, config.rep_hidden_dim, config.use_gpu)
        self.h1 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.h0 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.criterion = nn.MSELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        rep_t, rep_c = get_data_with_treatment_type(rep, treatment)
        lab_t, lab_c = get_data_with_treatment_type(labels, treatment)
        pred_loss = 0
        if rep_c.size(0) > 0: pred_loss += self.criterion(self.h0(rep_c).squeeze(), lab_c.float())
        if rep_t.size(0) > 0: pred_loss += self.criterion(self.h1(rep_t).squeeze(), lab_t.float())
        wass_loss = 0
        if rep_c.size(0) > 0 and rep_t.size(0) > 0:
            wass_loss = wasserstein_distance(rep_t, rep_c, p=0.5, lam=1, iterations=10)
        return pred_loss + self.alpha * wass_loss

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep(features)
        return self.h1(rep) - self.h0(rep)


class EUEN(nn.Module):
    """EUEN (Exposure-Utility Embedding Network) baseline."""

    def __init__(self, config: Any):
        super().__init__()
        self.h0 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.h1 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.criterion = nn.MSELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        feats_t, feats_c = get_data_with_treatment_type(features, treatment)
        lab_t, lab_c = get_data_with_treatment_type(labels, treatment)

        if feats_c.size(0) > 0:
            y_pred_h0_t0 = self.h0(feats_c).squeeze()
            loss_h0 = self.criterion(y_pred_h0_t0, lab_c.float())
        else:
            loss_h0 = torch.tensor(0.0, device=features.device)

        if feats_t.size(0) > 0:
            y_pred_h0_t1 = self.h0(feats_t).squeeze()
            y_pred_h1 = self.h1(feats_t).squeeze() + y_pred_h0_t1
            loss_h1 = self.criterion(y_pred_h1, lab_t.float())
        else:
            loss_h1 = torch.tensor(0.0, device=features.device)
        return loss_h0 + loss_h1

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        return self.h1(features)


class UniTE(nn.Module):
    """UniTE (Unified Treatment Effect) baseline."""

    def __init__(self, config: Any):
        super().__init__()
        input_dim = config.input_dim
        self.base_MMOE = MMOE(input_size=input_dim, num_experts=3, experts_hidden=64, experts_out=32, towers_hidden=8, tasks=3, tower_types=[False, False, True], tower_out=128)
        self.Outcome_MMOE = MMOE(input_size=128, num_experts=3, experts_hidden=32, experts_out=16, towers_hidden=8, tasks=1, tower_types=[False], tower_out=1)
        self.Uplift_MMOE = MMOE(input_size=128, num_experts=3, experts_hidden=32, experts_out=16, towers_hidden=8, tasks=1, tower_types=[False], tower_out=1)
        self.IPS_MMOE = MMOE(input_size=128, num_experts=3, experts_hidden=32, experts_out=16, towers_hidden=8, tasks=1, tower_types=[True], tower_out=1)
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        self.ce_loss_fn = nn.BCELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        treatment = treatment.float()
        labels = labels.float()
        base_out = self.base_MMOE(features)
        outcome_out = self.Outcome_MMOE(base_out[0])[0].squeeze()
        uplift_out = self.Uplift_MMOE(base_out[1])[0].squeeze()
        ips_out = self.IPS_MMOE(base_out[2])[0].squeeze()
        
        treatment = treatment.squeeze()
        labels = labels.squeeze()

        loss_outcome = self.mse_loss_fn(outcome_out, labels)
        loss_ips = self.ce_loss_fn(ips_out, treatment)
        r_learner_pred = outcome_out + uplift_out * (treatment - ips_out)
        loss_uplift = self.mse_loss_fn(r_learner_pred, labels)
        return loss_outcome + loss_ips + loss_uplift

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        base_out = self.base_MMOE(features)
        return self.Uplift_MMOE(base_out[1])[0]


from sklearn.ensemble import RandomForestRegressor

class CausalForestLearner(nn.Module):
    """PyTorch-compatible wrapper for an R-Learner (Causal Forest) using Scikit-Learn."""
    def __init__(self, n_estimators=100, max_depth=10, min_samples_leaf=10):
        super().__init__()
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'n_jobs': -1,
            'random_state': 42
        }
        self.model_y = RandomForestRegressor(**self.params)
        self.model_t = RandomForestRegressor(**self.params)
        self.model_cate = RandomForestRegressor(**self.params)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor): return x.cpu().detach().numpy()
        if isinstance(x, list): return np.array(x)
        return x

    def fit(self, X, y, t):
        X = self._to_numpy(X)
        y = self._to_numpy(y).ravel()
        t = self._to_numpy(t).ravel()
        self.model_y.fit(X, y)
        self.model_t.fit(X, t)
        y_pred = self.model_y.predict(X)
        t_pred = self.model_t.predict(X)
        y_res = y - y_pred
        t_res = t - t_pred
        mask = (np.abs(t_res) > 0.01)
        if np.sum(mask) > 0:
            X_final = X[mask]
            target = y_res[mask] / t_res[mask]
            weights = t_res[mask]**2
            self.model_cate.fit(X_final, target, sample_weight=weights)
        else:
            print("Warning: Treatment is deterministic. Causal Forest cannot learn.")

    def forward(self, X):
        X_np = self._to_numpy(X)
        uplift = self.model_cate.predict(X_np)
        return torch.tensor(uplift.reshape(-1, 1), dtype=torch.float32)
