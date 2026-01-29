"""Synthetic data generation logic for VALOR experiments.

This module provides functions to generate synthetic datasets compatible with
ZILN (Zero-Inflated Log-Normal) distributions.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def set_random_seed(seed: int):
    """Sets random seeds for reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_dataframe(
    n_samples: int,
    n_binary: int,
    p_binary: float,
    n_continuous: int,
    mean_gaussian: float,
    std_gaussian: float,
    n_classes: int,
    str_name: str,
    cluster: bool = False
) -> pd.DataFrame:
    """Generates a base dataframe with binary and continuous features."""
    columns = []
    data = []

    if n_binary > 0:
        binary_data = np.random.binomial(1, p_binary, (n_samples, n_binary))
        binary_columns = [f'{str_name}_binary_{i+1}' for i in range(n_binary)]
        columns.extend(binary_columns)
        data.append(binary_data)

    if n_continuous > 0:
        continuous_data = np.random.normal(mean_gaussian, std_gaussian, (n_samples, n_continuous))
        continuous_columns = [f'{str_name}_continuous_{i+1}' for i in range(n_continuous)]
        columns.extend(continuous_columns)
        data.append(continuous_data)

    if n_classes > 0:
        class_labels = np.random.randint(0, n_classes, n_samples) / n_classes
        columns.append(f'{str_name}_class')
        data.append(class_labels.reshape(-1, 1))

    if cluster:
        cluster_labels = np.random.randint(0, 6, n_samples)
        columns.append('data_cluster')
        data.append(cluster_labels.reshape(-1, 1))

    return pd.DataFrame(np.column_stack(data), columns=columns)


def create_synthetic_data(
    n_uid: int = 5000,
    n_pid: int = 50000,
    seed: int = 123
) -> pd.DataFrame:
    """Generates Synthetic Data compatible with ZILN distribution.

    Args:
        n_uid: Number of users.
        n_pid: Number of products.
        seed: Random seed.

    Returns:
        A DataFrame containing features, treatment, labels, and true tau.
    """
    print("Generating Synthetic Data (ZILN Compatible)...")
    set_random_seed(seed)

    df_uid = generate_dataframe(
        n_samples=n_uid, n_binary=23, p_binary=0.5,
        n_continuous=83, mean_gaussian=0, std_gaussian=1,
        n_classes=10, str_name='uid'
    )

    df_pid = generate_dataframe(
        n_samples=n_pid, n_binary=0, p_binary=0.5,
        n_continuous=106, mean_gaussian=0, std_gaussian=1,
        n_classes=300, str_name='pid', cluster=True
    )

    df_uid_columns = df_uid.columns
    df_pid_columns = df_pid.columns

    joined_data = []
    for _, row in df_uid.iterrows():
        n = np.random.randint(30, 60)
        selected_rows = df_pid.sample(n=n, replace=True)
        repeated_row = pd.concat([pd.DataFrame([row])] * n, ignore_index=True)
        joined_row = pd.concat([repeated_row, selected_rows.reset_index(drop=True)], axis=1)
        joined_data.append(joined_row)

    result_df = pd.concat(joined_data, ignore_index=True)

    u_id_sum = result_df[df_uid_columns].sum(axis=1)
    p_id_sum = result_df[df_pid_columns].sum(axis=1)
    u_id_squared_sum = result_df[df_uid_columns].pow(2).sum(axis=1)

    def normalize(x):
        return (x - x.mean()) / (x.std() + 1e-5)

    feat_interaction = normalize(u_id_sum * p_id_sum)
    feat_user = normalize(u_id_squared_sum)
    feat_item = normalize(p_id_sum)

    # Propensity Generation
    base_logits = -0.5 + 0.5 * feat_user + 0.2 * feat_item
    prob_c = 1 / (1 + np.exp(-base_logits))
    prob_t = 1 / (1 + np.exp(-(base_logits + 0.3 + 0.1 * feat_interaction)))

    # Revenue Generation (Mu)
    base_mu = 3.0 + 0.3 * feat_interaction + 0.2 * feat_user
    mu_c = base_mu
    mu_t = base_mu + 0.2 + 0.2 * feat_interaction
    sigma = 0.5

    # Sample Outcomes
    conv_c = np.random.binomial(1, prob_c)
    conv_t = np.random.binomial(1, prob_t)
    rev_c = np.exp(np.random.normal(mu_c, sigma))
    rev_t = np.exp(np.random.normal(mu_t, sigma))

    y0 = conv_c * rev_c
    y1 = conv_t * rev_t

    # True Tau (Expected Value Difference)
    ev_c = prob_c * np.exp(mu_c + 0.5 * sigma**2)
    ev_t = prob_t * np.exp(mu_t + 0.5 * sigma**2)
    true_tau = ev_t - ev_c

    result_df['y0'] = y0
    result_df['y1'] = y1
    result_df['true_tau'] = true_tau
    result_df['treatment'] = np.random.choice([0, 1], size=len(result_df))
    result_df['label'] = np.where(result_df['treatment'] == 0, result_df['y0'], result_df['y1'])

    print(f"Data Generated. Shape: {result_df.shape}")
    return result_df
