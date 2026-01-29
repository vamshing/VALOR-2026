"""Data loading utilities for VALOR experiments.

This module provides functions to prepare DataLoaders for PyTorch models.
"""

from typing import Tuple, List
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_gcp_data():
    raise NotImplementedError("This dataset is proprietary and cannot be released. Use synthetic_data instead.")


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    batch_size: int = 512,
    train_on_sample: bool = False
) -> Tuple[DataLoader, DataLoader, int, List[str], List[str], List[str], List[str], List[str]]:
    """Preprocesses data and creates Train/Val DataLoaders.

    Args:
        df: Input DataFrame.
        test_size: Fraction of data for validation.
        batch_size: Batch size for loaders.
        train_on_sample: If True, uses only 10% of data for debugging.

    Returns:
        A tuple of (train_loader, valid_loader, input_dim, feature_groups...).
    """
    feature_cols = [c for c in df.columns if c not in ['y0', 'y1', 'treatment', 'label', 'true_tau']]

    uid_cont_cols = [c for c in feature_cols if 'uid_continuous' in c]
    uid_cat_cols = [c for c in feature_cols if 'uid_binary' in c or 'uid_class' in c]
    pid_cont_cols = [c for c in feature_cols if 'pid_continuous' in c]
    pid_cat_cols = [c for c in feature_cols if 'pid_class' in c]
    cluster_cols = [c for c in feature_cols if 'data_cluster' in c]

    scaler = StandardScaler()
    df[uid_cont_cols + pid_cont_cols] = scaler.fit_transform(df[uid_cont_cols + pid_cont_cols])

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    if train_on_sample:
        train_df = train_df.sample(frac=0.1, random_state=42)
        val_df = val_df.sample(frac=0.1, random_state=42)

    def create_loader(dataframe):
        t_uid_cont = torch.tensor(dataframe[uid_cont_cols].values, dtype=torch.float32)
        t_uid_cat = torch.tensor(dataframe[uid_cat_cols].values, dtype=torch.float32)
        t_pid_cont = torch.tensor(dataframe[pid_cont_cols].values, dtype=torch.float32)
        t_pid_cat = torch.tensor(dataframe[pid_cat_cols].values, dtype=torch.float32)
        t_treat = torch.tensor(dataframe['treatment'].values, dtype=torch.float32)
        t_label = torch.tensor(dataframe['label'].values, dtype=torch.float32)
        t_clust = torch.tensor(dataframe[cluster_cols].values, dtype=torch.float32)
        t_tau = torch.tensor(dataframe['true_tau'].values, dtype=torch.float32)

        dataset = TensorDataset(t_uid_cont, t_uid_cat, t_pid_cont, t_pid_cat, t_treat, t_label, t_clust, t_tau)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_loader(train_df)
    valid_loader = create_loader(val_df)

    input_dim = len(uid_cont_cols) + len(uid_cat_cols) + len(pid_cont_cols) + len(pid_cat_cols)

    return (
        train_loader, valid_loader, input_dim,
        uid_cont_cols, uid_cat_cols, pid_cont_cols, pid_cat_cols, cluster_cols
    )
