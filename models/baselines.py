"""Baseline models for uplift modeling.

This module provides implementations for S-Learner, T-Learner, TARNet, DragonNet,
CFRNet (MMD/Wasserstein), UniTE, and EUEN.
"""

from typing import Any
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


class SLearner(nn.Module):
    """S-Learner baseline."""

    def __init__(self, config: Any):
        super().__init__()
        self.net1 = Net(config.input_dim, 1, config.hidden_dim, config.use_gpu, True)
        self.criterion = nn.MSELoss(reduction='mean')

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        if treatment.dim() == 1: treatment = treatment.unsqueeze(1)
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, treatment], dim=1)
        pred = self.net1(features)
        loss = self.criterion(pred.squeeze(), labels.float())
        return loss

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        feats = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        t1 = torch.ones((feats.size(0), 1), device=feats.device)
        y1 = self.net1(torch.cat([feats, t1], dim=1))
        t0 = torch.zeros((feats.size(0), 1), device=feats.device)
        y0 = self.net1(torch.cat([feats, t0], dim=1))
        return y1 - y0


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
        return self.t1_net(features) - self.t0_net(features)


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
        loss_h0 = self.criterion(self.h0(rep_c).squeeze(), lab_c.float()) if rep_c.size(0) > 0 else 0
        loss_h1 = self.criterion(self.h1(rep_t).squeeze(), lab_t.float()) if rep_t.size(0) > 0 else 0
        return (loss_h0 + loss_h1) / 2 if (rep_c.size(0) > 0 and rep_t.size(0) > 0) else (loss_h0 + loss_h1)

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
        loss_h0 = self.criterion(self.h0(feats_c).squeeze(), lab_c.float()) if feats_c.size(0) > 0 else 0
        if feats_t.size(0) > 0:
            y_pred_h1 = self.h1(feats_t).squeeze() + self.h0(feats_t).squeeze()
            loss_h1 = self.criterion(y_pred_h1, lab_t.float())
        else:
            loss_h1 = 0
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
        base_out = self.base_MMOE(features)
        outcome_out = self.Outcome_MMOE(base_out[0])[0].squeeze()
        uplift_out = self.Uplift_MMOE(base_out[1])[0].squeeze()
        ips_out = self.IPS_MMOE(base_out[2])[0].squeeze()
        loss_outcome = self.mse_loss_fn(outcome_out, labels.float().squeeze())
        loss_ips = self.ce_loss_fn(ips_out, treatment.float().squeeze())
        r_learner_pred = outcome_out + uplift_out * (treatment.squeeze() - ips_out)
        loss_uplift = self.mse_loss_fn(r_learner_pred, labels.float().squeeze())
        return loss_outcome + loss_ips + loss_uplift

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        base_out = self.base_MMOE(features)
        return self.Uplift_MMOE(base_out[1])[0]
