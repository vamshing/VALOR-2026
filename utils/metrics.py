"""Metric functions for uplift modeling evaluation.

This module provides implementations for AUUC, Qini coefficient, Lift@k, and Kendall correlation.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def get_true_kendall(preds: np.ndarray, true_tau: np.ndarray) -> float:
    """Calculates Kendall Rank Correlation between Predicted and Ground Truth Uplift.

    Args:
        preds: Predicted uplift values.
        true_tau: Ground truth uplift (ITE).

    Returns:
        The Kendall correlation coefficient.
    """
    if len(preds) > 5000:
        indices = np.random.choice(len(preds), 5000, replace=False)
        return kendalltau(preds[indices], true_tau[indices])[0]
    return kendalltau(preds, true_tau)[0]


def lift_h_metric(
    df: pd.DataFrame,
    h: float = 0.3,
    outcome_col: str = 'label',
    treatment_col: str = 'treatment',
    predict_col: str = '1'
) -> float:
    """Calculates Lift@h% metric.

    The difference in mean outcome between Treatment and Control in the top h%
    of samples ranked by predicted uplift.

    Args:
        df: DataFrame containing predictions and outcomes.
        h: Fraction of the population to consider (top h%).
        outcome_col: Name of the outcome column.
        treatment_col: Name of the treatment column.
        predict_col: Name of the prediction column.

    Returns:
        The lift value.
    """
    df = df.sort_values(predict_col, ascending=False)
    top_n = int(len(df) * h)
    if top_n == 0:
        return 0.0

    df_top = df.iloc[:top_n]
    mask_t = (df_top[treatment_col] == 1)
    mask_c = (df_top[treatment_col] == 0)

    mean_t = df_top.loc[mask_t, outcome_col].mean() if mask_t.sum() > 0 else 0
    mean_c = df_top.loc[mask_c, outcome_col].mean() if mask_c.sum() > 0 else 0

    return mean_t - mean_c


def get_cumlift(df: pd.DataFrame) -> pd.Series:
    """Computes the cumulative uplift curve.

    Args:
        df: DataFrame with 'model' (preds), 'y' (outcome), and 'w' (treatment).

    Returns:
        A Series representing the cumulative uplift.
    """
    df = df.sort_values('model', ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df['cumsum_tr'] = df['w'].cumsum()
    df['cumsum_ct'] = df.index.values - df['cumsum_tr']
    df['cumsum_tr_y'] = (df['y'] * df['w']).cumsum()
    df['cumsum_ct_y'] = (df['y'] * (1 - df['w'])).cumsum()

    df['cumsum_tr'] = df['cumsum_tr'].replace(0, 1e-10)
    df['cumsum_ct'] = df['cumsum_ct'].replace(0, 1e-10)

    df['uplift'] = df['cumsum_tr_y'] / df['cumsum_tr'] - df['cumsum_ct_y'] / df['cumsum_ct']
    return df['uplift']


def get_causalml_auuc(y: np.ndarray, t: np.ndarray, ite_pred: np.ndarray) -> float:
    """Calculates Area Under Uplift Curve (AUUC).

    Args:
        y: Outcome labels.
        t: Treatment assignments.
        ite_pred: Predicted ITE.

    Returns:
        The AUUC score.
    """
    metric_df = pd.DataFrame({
        'model': ite_pred.flatten(),
        'y': y.flatten(),
        'w': t.flatten()
    })
    uplift_curve = get_cumlift(metric_df)
    uplift_rank_gain = uplift_curve * uplift_curve.index.values
    total_gain = np.abs(uplift_rank_gain.iloc[-1])
    if total_gain == 0:
        total_gain = 1e-10

    uplift_rank_gain_normalized = uplift_rank_gain / total_gain
    return uplift_rank_gain_normalized.mean()


def calc_qini(
    df: pd.DataFrame,
    outcome_col: str = 'label',
    treatment_col: str = 'treatment',
    predict_col: str = '1'
) -> float:
    """Calculates the Qini Coefficient.

    Args:
        df: DataFrame with predictions and labels.
        outcome_col: Label column.
        treatment_col: Treatment column.
        predict_col: Prediction column.

    Returns:
        The Qini coefficient.
    """
    df = df.sort_values(predict_col, ascending=False).reset_index(drop=True)
    n_t = df[treatment_col].sum()
    n_c = len(df) - n_t
    if n_c == 0:
        return 0.0

    df['cumsum_tr'] = df[treatment_col].cumsum()
    df['cumsum_ct'] = (df.index + 1) - df['cumsum_tr']
    df['cumsum_y_tr'] = (df[outcome_col] * df[treatment_col]).cumsum()
    df['cumsum_y_ct'] = (df[outcome_col] * (1 - df[treatment_col])).cumsum()

    uplift_curve = df['cumsum_y_tr'] - (df['cumsum_y_ct'] * (n_t / n_c))
    random_curve = (uplift_curve.iloc[-1] / len(df)) * (df.index + 1)

    actual_lift = uplift_curve.iloc[-1]
    if actual_lift == 0:
        actual_lift = 1.0
    return (uplift_curve - random_curve).mean() / abs(actual_lift)


def get_eval_score(
    labels: np.ndarray,
    preds: np.ndarray,
    treatment: np.ndarray,
    true_tau: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """Computes a comprehensive set of evaluation metrics.

    Args:
        labels: Ground truth outcome.
        preds: Predicted uplift.
        treatment: Treatment assignment.
        true_tau: Optional ground truth ITE.

    Returns:
        A tuple of (Qini, Kendall/Binned-Kendall, Lift@30).
    """
    df_eval = pd.DataFrame({
        'label': labels.flatten(),
        '1': preds.flatten(),
        'treatment': treatment.flatten()
    })

    qini_coef = calc_qini(df_eval)

    # Binned Kendall
    df_eval['1_noise'] = df_eval['1'] + np.random.normal(0, 1e-10, size=len(df_eval))
    num_bucket = 50
    try:
        df_eval['score_bucket'] = pd.qcut(df_eval['1_noise'], num_bucket, duplicates='drop', labels=False)
    except Exception:
        df_eval['score_bucket'] = pd.cut(df_eval['1_noise'], num_bucket, labels=False)

    res = df_eval.groupby(['score_bucket', 'treatment'])['label'].mean().reset_index()
    t0 = res[res.treatment == 0].rename(columns={'label': 'mean_0'})
    t1 = res[res.treatment == 1].rename(columns={'label': 'mean_1'})

    comp = pd.merge(t0, t1, on='score_bucket', how='inner')
    comp['uplift'] = comp['mean_1'] - comp['mean_0']

    binned_kendall = comp[['score_bucket', 'uplift']].corr(method='kendall').iloc[0, 1] if len(comp) > 1 else 0.0

    lift_30 = lift_h_metric(df_eval, h=0.3)
    final_kendall = get_true_kendall(preds, true_tau.flatten()) if true_tau is not None else binned_kendall

    return qini_coef, final_kendall, lift_30
