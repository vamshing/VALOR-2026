"""Reusable layers and components for VALOR models.

This module provides custom PyTorch modules like TreatmentGatedInteraction and ZILNHead.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TreatmentGatedInteraction(nn.Module):
    """Applies a gating mechanism based on treatment assignment.

    Attributes:
        use_gti: Whether to enable the gating interaction.
        wx: Linear transformation for features.
        wt: Linear transformation for treatment to compute gate values.
    """

    def __init__(self, feature_dim: int, use_gti: bool = True):
        super(TreatmentGatedInteraction, self).__init__()
        self.use_gti = use_gti
        self.wx = nn.Linear(feature_dim, feature_dim)
        self.wt = nn.Linear(1, feature_dim)
        self.wt.bias.data.fill_(2.0)  # Initial bias to keep gate open
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for TreatmentGatedInteraction."""
        feature_proj = self.wx(x)
        if not self.use_gti:
            return feature_proj, feature_proj

        t_c = torch.zeros_like(t)
        gate_c = self.sigmoid(self.wt(t_c))
        t_t = torch.ones_like(t)
        gate_t = self.sigmoid(self.wt(t_t))

        rep_c = feature_proj * gate_c
        rep_t = feature_proj * gate_t
        return rep_c, rep_t


class ZILNHead(nn.Module):
    """Prediction head for Zero-Inflated Log-Normal distribution.

    Outputs logits for conversion probability, mu (log-mean), and sigma (scale).
    """

    def __init__(self, input_dim: int, hidden_dim: int, prior_mu: float = 0.0):
        super(ZILNHead, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU()
        )
        self.out_logits = nn.Linear(hidden_dim, 1)
        self.out_mu = nn.Linear(hidden_dim, 1)
        self.out_sigma = nn.Linear(hidden_dim, 1)

        self.out_mu.bias.data.fill_(prior_mu)
        self.out_sigma.bias.data.fill_(0.0)
        self.out_logits.bias.data.fill_(-1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for ZILNHead."""
        h = self.hidden(x)
        logits = self.out_logits(h)
        mu = self.out_mu(h)
        sigma = torch.clamp(F.softplus(self.out_sigma(h)) + 1e-4, 1e-4, 3.0)
        return logits, mu, sigma


class Expert(nn.Module):
    """Expert module for Mixture-of-Experts (MoE) architectures."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Tower(nn.Module):
    """Tower module for MoE architectures, supporting regression or classification."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int, is_classification: bool = False):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.is_classification = is_classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.is_classification:
            out = self.sigmoid(out)
        return out


class MMOE(nn.Module):
    """Multi-gate Mixture-of-Experts module."""

    def __init__(
        self,
        input_size: int,
        num_experts: int,
        experts_hidden: int,
        experts_out: int,
        towers_hidden: int,
        tasks: int,
        tower_types: list,
        tower_out: int
    ):
        super(MMOE, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.experts_out = experts_out
        self.experts = nn.ModuleList([Expert(input_size, experts_out, experts_hidden) for _ in range(num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for _ in range(tasks)])
        self.towers = nn.ModuleList([Tower(experts_out, tower_out, towers_hidden, is_classification=tower_types[i]) for i in range(tasks)])

    def forward(self, x: torch.Tensor) -> list:
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)
        gates_o = [self.softmax(x @ g) for g in self.w_gates]
        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]
        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return final_output
