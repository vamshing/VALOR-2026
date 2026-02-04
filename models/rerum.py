"""RERUM Implementation.

This module provides the RERUM model architecture and its specific loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from VALOR.training.losses import zero_inflated_lognormal_pred, zero_inflated_lognormal_loss, uplift_ranking_loss

class RERUM(nn.Module):
    def __init__(self, config):
        super(RERUM, self).__init__()
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        rep_dim = config.rep_hidden_dim

        self.rep_layer = nn.Sequential(
            nn.Linear(input_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU(),
            nn.Linear(rep_dim, rep_dim),
            nn.ELU()
        )

        self.propensity_head = nn.Linear(rep_dim, 1)

        # Heads output 3 values: [logit_prob, mu, sigma_raw]
        self.head_y0 = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 3)
        )

        self.head_y1 = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 3)
        )

        self.alpha_rank = getattr(config, 'alpha_rank', 1.0)

    def forward(self, uid_cont, uid_cat, pid_cont, pid_cat, *args):
        x = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep_layer(x)

        logits_y0 = self.head_y0(rep)
        logits_y1 = self.head_y1(rep)

        pred_y0 = zero_inflated_lognormal_pred(logits_y0)
        pred_y1 = zero_inflated_lognormal_pred(logits_y1)

        return pred_y1 - pred_y0

    def calculate_loss(self, uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, *args):
        x = torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1)
        rep = self.rep_layer(x)

        t_logits = self.propensity_head(rep)
        t_prob = torch.sigmoid(t_logits)

        logits_y0 = self.head_y0(rep)
        logits_y1 = self.head_y1(rep)

        loss_t = F.binary_cross_entropy_with_logits(t_logits.squeeze(), treatment.float())

        mask_t = (treatment == 1)
        mask_c = (treatment == 0)

        loss_y0 = 0.0
        if mask_c.sum() > 0:
            loss_y0 = zero_inflated_lognormal_loss(labels[mask_c].view(-1, 1), logits_y0[mask_c])

        loss_y1 = 0.0
        if mask_t.sum() > 0:
            loss_y1 = zero_inflated_lognormal_loss(labels[mask_t].view(-1, 1), logits_y1[mask_t])

        loss_ziln = loss_y0 + loss_y1

        loss_rank = uplift_ranking_loss(
            y_true=labels.float(),
            t_true=treatment.float(),
            t_pred=t_prob,
            y0_logits=logits_y0,
            y1_logits=logits_y1
        )

        total_loss = loss_ziln + loss_t + (self.alpha_rank * loss_rank)
        return total_loss
