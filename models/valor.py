"""VALOR (Value-Aware Learning for Optimized Revenue) model architectures.

This module provides VALOR-enhanced versions of TARNet, DragonNet, UniTE, and EUEN.
"""

from typing import Any
import torch
import torch.nn as nn
from VALOR.models.layers import TreatmentGatedInteraction, ZILNHead, StandardHead
from VALOR.training.losses import valor_loss_function, compute_mmd_loss, compute_wasserstein_loss


class VALOR_TARNet(nn.Module):
    """VALOR-enhanced TARNet."""

    def __init__(self, config: Any, prior_log_mean: float = 1.0):
        super(VALOR_TARNet, self).__init__()
        self.config = config

        # Standard Rep (No BatchNorm)
        self.rep = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU()
        )
        self.gate = TreatmentGatedInteraction(config.hidden_dim, use_gti=config.use_gti)
        
        # Toggle Heads based on Config
        if config.use_ziln:
            self.h0 = ZILNHead(config.hidden_dim, config.hidden_dim // 2, prior_mu=prior_log_mean)
            self.h1 = ZILNHead(config.hidden_dim, config.hidden_dim // 2, prior_mu=prior_log_mean)
        else:
            self.h0 = StandardHead(config.hidden_dim, config.hidden_dim // 2)
            self.h1 = StandardHead(config.hidden_dim, config.hidden_dim // 2)

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        treatment = treatment.float().view(-1, 1)
        labels = labels.float().view(-1, 1)

        phi = self.rep(features)
        
        # IPM Loss Calculation
        ipm_loss = torch.tensor(0.0, device=features.device)
        if self.config.use_ipm:
            mask_t = (treatment > 0).flatten()
            mask_c = ~mask_t
            if mask_t.sum() > 0 and mask_c.sum() > 0:
                if self.config.ipm_type == 'wass':
                    ipm_loss = compute_wasserstein_loss(phi[mask_t], phi[mask_c])
                else:
                    ipm_loss = compute_mmd_loss(phi[mask_t], phi[mask_c])

        rep_c, rep_t = self.gate(phi, treatment)
        c_out = self.h0(rep_c)
        t_out = self.h1(rep_t)

        loss = valor_loss_function(c_out, t_out, labels, treatment,
                                   use_focal=self.config.use_focal,
                                   use_ranking=self.config.use_ranking,
                                   use_ziln=self.config.use_ziln)
        return loss + (self.config.alpha_ipm * ipm_loss)

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        dummy_t = torch.zeros((features.size(0), 1), device=features.device)
        phi = self.rep(features)
        rep_c, rep_t = self.gate(phi, dummy_t)
        
        if self.config.use_ziln:
            c_l, c_m, _ = self.h0(rep_c)
            t_l, t_m, _ = self.h1(rep_t)
            def get_expect(logits, mu):
                return torch.sigmoid(logits) * torch.exp(torch.clamp(mu, -10, 10))
            return get_expect(t_l, t_m) - get_expect(c_l, c_m)
        else:
            return self.h1(rep_t) - self.h0(rep_c)


class VALOR_DragonNet(nn.Module):
    """VALOR-enhanced DragonNet."""

    def __init__(self, config: Any, prior_log_mean: float = 1.0):
        super(VALOR_DragonNet, self).__init__()
        self.config = config
        
        # Standard Rep (No BatchNorm)
        self.rep = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU()
        )
        self.propensity = nn.Linear(config.hidden_dim, 1)
        self.gate = TreatmentGatedInteraction(config.hidden_dim, use_gti=config.use_gti)
        
        # Toggle Heads
        if config.use_ziln:
            self.h0 = ZILNHead(config.hidden_dim, config.hidden_dim // 2, prior_mu=prior_log_mean)
            self.h1 = ZILNHead(config.hidden_dim, config.hidden_dim // 2, prior_mu=prior_log_mean)
        else:
            self.h0 = StandardHead(config.hidden_dim, config.hidden_dim // 2)
            self.h1 = StandardHead(config.hidden_dim, config.hidden_dim // 2)
            
        self.bce = nn.BCEWithLogitsLoss()

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        treatment = treatment.float().view(-1, 1)
        labels = labels.float().view(-1, 1)

        phi = self.rep(features)
        loss_p = self.bce(self.propensity(phi), treatment)

        rep_c, rep_t = self.gate(phi, treatment)
        
        loss_v = valor_loss_function(
            self.h0(rep_c), self.h1(rep_t), labels, treatment,
            use_focal=self.config.use_focal,
            use_ranking=self.config.use_ranking,
            use_ziln=self.config.use_ziln
        )
        return loss_v + loss_p

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        dummy_t = torch.zeros((features.size(0), 1), device=features.device)
        phi = self.rep(features)
        rep_c, rep_t = self.gate(phi, dummy_t)

        if self.config.use_ziln:
            c_l, c_m, _ = self.h0(rep_c)
            t_l, t_m, _ = self.h1(rep_t)
            def get_expect(logits, mu):
                return torch.sigmoid(logits) * torch.exp(torch.clamp(mu, -10, 10))
            return get_expect(t_l, t_m) - get_expect(c_l, c_m)
        else:
            return self.h1(rep_t) - self.h0(rep_c)


class VALOR_UniTE(nn.Module):
    """VALOR-enhanced UniTE."""

    def __init__(self, config: Any, prior_log_mean: float = 1.0):
        super(VALOR_UniTE, self).__init__()
        self.config = config

        # Standard Rep (BatchNorm removed)
        self.rep = nn.Sequential(
            nn.Linear(config.input_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU()
        )
        self.gate = TreatmentGatedInteraction(128, use_gti=config.use_gti)
        
        # Toggle Heads based on ZILN flag
        if config.use_ziln:
            self.h0 = ZILNHead(128, 64, prior_log_mean)
            self.h1 = ZILNHead(128, 64, prior_log_mean)
        else:
            self.h0 = StandardHead(128, 64)
            self.h1 = StandardHead(128, 64)

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        treatment = treatment.float().view(-1, 1)
        labels = labels.float().view(-1, 1)

        phi = self.rep(features)
        rep_c, rep_t = self.gate(phi, treatment)
        
        return valor_loss_function(self.h0(rep_c), self.h1(rep_t), labels, treatment,
                                   use_focal=self.config.use_focal,
                                   use_ranking=self.config.use_ranking,
                                   use_ziln=self.config.use_ziln)

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        dummy_t = torch.zeros((features.size(0), 1), device=features.device)
        phi = self.rep(features)
        rep_c, rep_t = self.gate(phi, dummy_t)

        if self.config.use_ziln:
            def get_expect(logits, mu): 
                return torch.sigmoid(logits) * torch.exp(torch.clamp(mu, -10, 10))
            c_l, c_m, _ = self.h0(rep_c)
            t_l, t_m, _ = self.h1(rep_t)
            return get_expect(t_l, t_m) - get_expect(c_l, c_m)
        else:
            return self.h1(rep_t) - self.h0(rep_c)


class VALOR_EUEN(nn.Module):
    """
    VALOR-Enhanced EUEN.
    Correctly toggles between ZILNHead and StandardHead.
    """
    def __init__(self, config: Any, prior_log_mean: float = 1.0):
        super(VALOR_EUEN, self).__init__()
        self.config = config

        # Shared Feature Encoding (No BatchNorm)
        self.rep = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU()
        )

        # Explicit Interaction Mechanism
        self.gate = TreatmentGatedInteraction(config.hidden_dim, use_gti=config.use_gti)

        # Conditional Heads
        if config.use_ziln:
            self.h0 = ZILNHead(config.hidden_dim, config.hidden_dim // 2, prior_mu=prior_log_mean)
            self.h1 = ZILNHead(config.hidden_dim, config.hidden_dim // 2, prior_mu=prior_log_mean)
        else:
            self.h0 = StandardHead(config.hidden_dim, config.hidden_dim // 2)
            self.h1 = StandardHead(config.hidden_dim, config.hidden_dim // 2)

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        treatment = treatment.float().view(-1, 1)
        labels = labels.float().view(-1, 1)

        phi = self.rep(features)
        rep_c, rep_t = self.gate(phi, treatment)

        c_out = self.h0(rep_c)
        t_out = self.h1(rep_t)

        loss = valor_loss_function(c_out, t_out, labels, treatment,
                                   use_focal=self.config.use_focal,
                                   use_ranking=self.config.use_ranking,
                                   use_ziln=self.config.use_ziln)
        return loss

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster):
        features = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat, pid_cluster], dim=1)
        dummy_t = torch.zeros((features.size(0), 1), device=features.device)

        phi = self.rep(features)
        rep_c, rep_t = self.gate(phi, dummy_t)

        if self.config.use_ziln:
            c_l, c_m, _ = self.h0(rep_c)
            t_l, t_m, _ = self.h1(rep_t)
            def get_expect(logits, mu):
                return torch.sigmoid(logits) * torch.exp(torch.clamp(mu, -10, 10))
            return get_expect(t_l, t_m) - get_expect(c_l, c_m)
        else:
            return self.h1(rep_t) - self.h0(rep_c)
