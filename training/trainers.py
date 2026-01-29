"""Training and evaluation loops for VALOR models.

This module provides functions for epoch training and model validation.
"""

from typing import Dict, Any
import time
import numpy as np
import torch
from VALOR.utils.metrics import get_eval_score, get_causalml_auuc


def train_epoch(model, loader, device, optimizer) -> float:
    """Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training.
        device: Device to train on.
        optimizer: Optimizer.

    Returns:
        The average loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch in loader:
        # Unpack items from DataLoader
        uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster, _ = [b.to(device) for b in batch]

        optimizer.zero_grad()
        loss = model.calculate_loss(uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device) -> Dict[str, float]:
    """Evaluates the model on a validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation.
        device: Device to evaluate on.

    Returns:
        A dictionary containing AUUC, Qini, Kendall, and Lift@30.
    """
    model.eval()
    pred_uplift, true_labels, treatments, true_taus = [], [], [], []

    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster, batch_tau = [b.to(device) for b in batch]
            tau_pred = model(uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster)

            pred_uplift.extend(tau_pred.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy().flatten())
            treatments.extend(treatment.cpu().numpy().flatten())
            true_taus.extend(batch_tau.cpu().numpy().flatten())
    end_time = time.time()

    pred_uplift, true_labels, treatments, true_taus = np.array(pred_uplift), np.array(true_labels), np.array(treatments), np.array(true_taus)

    auuc_raw = get_causalml_auuc(true_labels, treatments, pred_uplift)
    qini, kendall, lift_30 = get_eval_score(true_labels, pred_uplift, treatments, true_taus)

    return {
        'auuc': auuc_raw,
        'qini': qini,
        'kendall': kendall,
        'lift_30': lift_30,
        'inference_time': end_time - start_time
    }
