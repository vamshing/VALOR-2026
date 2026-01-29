"""Loss functions for VALOR models.

This module provides ZILN, Focal, Ranking, MMD, and Wasserstein loss functions.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mmd_loss(rep_t: torch.Tensor, rep_c: torch.Tensor) -> torch.Tensor:
    """Computes Maximum Mean Discrepancy (MMD) loss."""
    if rep_t.size(0) == 0 or rep_c.size(0) == 0:
        return torch.tensor(0.0, device=rep_t.device)
    return torch.sum((rep_t.mean(0) - rep_c.mean(0))**2)


def compute_wasserstein_loss(rep_t: torch.Tensor, rep_c: torch.Tensor) -> torch.Tensor:
    """Computes an approximation of Wasserstein Distance."""
    if rep_t.size(0) == 0 or rep_c.size(0) == 0:
        return torch.tensor(0.0, device=rep_t.device)
    return torch.norm(rep_t.mean(0) - rep_c.mean(0), p=2)


def valor_loss_function(
    c_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    t_out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    treatment: torch.Tensor,
    lambda_rank: float = 0.1,
    alpha: float = 0.75,
    gamma: float = 2.0,
    use_focal: bool = True,
    use_ranking: bool = True,
    use_ziln: bool = True
) -> torch.Tensor:
    """Combined loss function for VALOR.

    Includes Distribution Loss (Log-Normal + Focal) and Ranking Loss.
    """
    if use_ziln:
        c_logits, c_mu, c_sigma = c_out
        t_logits, t_mu, t_sigma = t_out

        def get_dist_loss(y_true, logits, mu, sigma):
            target_cls = (y_true > 0).float()
            p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)

            if use_focal:
                pt = (p * target_cls) + ((1 - p) * (1 - target_cls))
                alpha_t = (alpha * target_cls) + ((1 - alpha) * (1 - target_cls))
                prop_loss = -alpha_t * torch.pow(1 - pt, gamma) * torch.log(pt)
            else:
                prop_loss = F.binary_cross_entropy(p, target_cls, reduction='none')

            reg_loss = torch.zeros_like(prop_loss)
            nonzero_mask = (y_true > 0).view(-1)
            if nonzero_mask.sum() > 0:
                y_pos, mu_pos, sigma_pos = y_true[nonzero_mask], mu[nonzero_mask], sigma[nonzero_mask]
                safe_y = torch.clamp(y_pos, 1e-7)
                reg_term = 0.5 * ((torch.log(safe_y) - mu_pos) / sigma_pos) ** 2 + torch.log(sigma_pos)
                reg_loss.view(-1)[nonzero_mask] = reg_term.view(-1)
            return prop_loss + reg_loss

        loss_c = get_dist_loss(labels, c_logits, c_mu, c_sigma)
        loss_t = get_dist_loss(labels, t_logits, t_mu, t_sigma)
        dist_loss = torch.mean(loss_c * (1 - treatment) + loss_t * treatment)

        rank_loss_val = torch.tensor(0.0, device=labels.device)
        if use_ranking and lambda_rank > 0.0:
            prop_c, prop_t = torch.sigmoid(c_logits), torch.sigmoid(t_logits)
            tau_hat = prop_t - prop_c
            z_true = labels * (2.0 * treatment - 1.0)

            if len(labels) > 500:
                idx = torch.randperm(len(labels))[:500]
                sub_z, sub_tau = z_true[idx], tau_hat[idx]
            else:
                sub_z, sub_tau = z_true, tau_hat

            diff_z, diff_tau = sub_z - sub_z.t(), sub_tau - sub_tau.t()
            weights = torch.log1p(torch.abs(diff_z)).detach()
            signs = torch.sign(diff_z)
            valid_pairs = (torch.abs(diff_z) > 1e-6).float()
            pair_loss = weights * torch.log1p(torch.exp(-signs * diff_tau))
            rank_loss_val = torch.sum(pair_loss * valid_pairs) / (torch.sum(valid_pairs) + 1e-6)

        return dist_loss + (lambda_rank * rank_loss_val)
    else:
        y_pred = treatment * t_out + (1 - treatment) * c_out
        return F.mse_loss(y_pred, labels)
