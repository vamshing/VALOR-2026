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


def valor_loss_function(c_out, t_out, labels, treatment,
                        lambda_rank=0.10, alpha=0.75, gamma=2.0,
                        use_focal=True, use_ranking=True, use_ziln=True):
    """
    Robust Loss Function with Log-Value Hinge Ranking.
    Ensures a return value in all paths.
    """
    if use_ziln:
        c_logits, c_mu, c_sigma = c_out
        t_logits, t_mu, t_sigma = t_out

        # --- 1. Distribution Loss (ZILN + Focal) ---
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
                y_pos = y_true[nonzero_mask]; mu_pos = mu[nonzero_mask]; sigma_pos = sigma[nonzero_mask]
                safe_y = torch.clamp(y_pos, 1e-7)
                reg_term = 0.5 * ((torch.log(safe_y) - mu_pos) / sigma_pos) ** 2 + torch.log(sigma_pos)
                reg_loss.view(-1)[nonzero_mask] = reg_term.view(-1)

            return prop_loss + reg_loss

        loss_c = get_dist_loss(labels, c_logits, c_mu, c_sigma)
        loss_t = get_dist_loss(labels, t_logits, t_mu, t_sigma)
        dist_loss = torch.mean(loss_c * (1 - treatment) + loss_t * treatment)

        # --- 2. Ranking Loss: Log-Value Hinge Ranking ---
        rank_loss_val = torch.tensor(0.0, device=labels.device)

        if use_ranking and lambda_rank > 0.0:
            # Clamp mu to prevent outliers
            mu_c_safe = torch.clamp(c_mu, -10, 10)
            mu_t_safe = torch.clamp(t_mu, -10, 10)

            # Log-Value Score = log(Prob) + log(Value) -> Linear stability
            score_c = F.logsigmoid(c_logits) + mu_c_safe
            score_t = F.logsigmoid(t_logits) + mu_t_safe
            
            tau_hat = score_t - score_c
            z_true = labels * (2.0 * treatment - 1.0)

            # Sampling
            if len(labels) > 500:
                idx = torch.randperm(len(labels))[:500]
                sub_z = z_true[idx]
                sub_tau = tau_hat[idx]
            else:
                sub_z = z_true
                sub_tau = tau_hat

            diff_z = sub_z.unsqueeze(1) - sub_z.unsqueeze(0)
            diff_tau = sub_tau.unsqueeze(1) - sub_tau.unsqueeze(0)

            # Only rank pairs with meaningful dollar differences
            valid_mask = (torch.abs(diff_z) > 1.0).float()
            
            # Weight: Log Dollar Difference
            weights = torch.log1p(torch.abs(diff_z))
            
            # Hinge Loss: Ensure correct order with a margin
            # If Z_i > Z_j, we want Tau_i > Tau_j
            signs = torch.sign(diff_z)
            margin = 0.1
            
            if valid_mask.sum() > 0:
                # Loss = ReLU(margin - sign * diff_tau)
                hinge_error = F.relu(margin - (signs * diff_tau))
                weighted_hinge = hinge_error * weights
                rank_loss_val = torch.sum(weighted_hinge * valid_mask) / (torch.sum(valid_mask) + 1e-6)

        return dist_loss + (lambda_rank * rank_loss_val)

    else:
        # Standard MSE for Baselines
        y_pred = treatment * t_out + (1 - treatment) * c_out
        return F.mse_loss(y_pred, labels)


# ==========================================
# RERUM SPECIFIC FUNCTIONS
# ==========================================

import torch.distributions as tdist

def zero_inflated_lognormal_pred(logits):
    """Calculates predicted mean of zero inflated lognormal logits.
    Numerically stable version.
    """
    positive_probs = torch.sigmoid(logits[..., :1])
    loc = logits[..., 1:2]
    scale = F.softplus(logits[..., 2:])

    # --- STABILITY FIX ---
    # Clamp the exponent to prevent overflow (exp(88) is ~max float32)
    exponent = loc + 0.5 * scale**2
    exponent = torch.clamp(exponent, max=80.0)

    preds = positive_probs * torch.exp(exponent)
    return preds

def zero_inflated_lognormal_loss(labels, logits):
    """Computes the zero inflated lognormal loss."""
    positive = (labels > 0).float()

    positive_logits = logits[..., :1]
    classification_loss = F.binary_cross_entropy_with_logits(
        positive_logits, positive, reduction='mean')

    loc = logits[..., 1:2]
    scale = torch.max(
        F.softplus(logits[..., 2:]),
        torch.sqrt(torch.tensor(1e-6))
    )

    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    log_prob = tdist.LogNormal(loc=loc, scale=scale).log_prob(safe_labels)

    # Use sum/count instead of mean to handle empty batches gracefully
    regression_loss = -torch.sum(positive * log_prob) / (torch.sum(positive) + 1e-6)

    return classification_loss + regression_loss

def uplift_ranking_loss(y_true, t_true, t_pred, y0_logits, y1_logits):
    """Listwise ranking loss for uplift (Numerically Stable Version)."""
    # 1. Get Predictions (using clamped function)
    y0_pred = zero_inflated_lognormal_pred(y0_logits)
    y1_pred = zero_inflated_lognormal_pred(y1_logits)

    # 2. Predicted Uplift
    tau_pred = y1_pred - y0_pred

    # 3. Separate Treatment and Control groups
    t_mask = (t_true == 1).view(-1)
    c_mask = (t_true == 0).view(-1)

    tau_pred_t = tau_pred[t_mask].view(-1, 1)
    tau_pred_c = tau_pred[c_mask].view(-1, 1)

    treated_y = y_true[t_mask].view(-1, 1)
    control_y = y_true[c_mask].view(-1, 1)

    N1 = treated_y.shape[0]
    N0 = control_y.shape[0]

    # --- STABILITY FIX ---
    # Use log_softmax instead of log(softmax) for better numerical stability

    term_t = 0.0
    if N1 > 0:
        log_softmax_t = F.log_softmax(tau_pred_t, dim=0)
        term_t = (1.0 / N1) * torch.sum(treated_y * log_softmax_t)

    term_c = 0.0
    if N0 > 0:
        log_softmax_c = F.log_softmax(tau_pred_c, dim=0)
        term_c = (1.0 / N0) * torch.sum(control_y * log_softmax_c)

    # Standard implementation: maximize the correlation -> minimize negative correlation
    loss = - (N1 + N0) * (term_t - term_c)
    return loss
