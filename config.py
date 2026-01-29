"""Configuration for VALOR experiments.

This module defines hyperparameters for different models and dataset configurations.
"""

from typing import Dict, Any

# Model hyperparameter mappings
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'slearner': {
        'hidden_dim': 256,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'tlearner': {
        'hidden_dim': 256,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'tarnet': {
        'hidden_dim': 256,
        'rep_hidden_dim': 256,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'dragonnet': {
        'hidden_dim': 256,
        'rep_hidden_dim': 256,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'cfrnet': {
        'hidden_dim': 256,
        'rep_hidden_dim': 256,
        'alpha': 1.0,
        'ipm_type': 'mmd',
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'unite': {
        'hidden_dim': 256,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'euen': {
        'hidden_dim': 256,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'valor_tarnet': {
        'hidden_dim': 256,
        'use_gti': True,
        'use_ranking': True,
        'use_focal': True,
        'use_ipm': False,
        'alpha_ipm': 1.0,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'valor_dragonnet': {
        'hidden_dim': 256,
        'use_gti': True,
        'use_ranking': True,
        'use_focal': True,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'valor_unite': {
        'hidden_dim': 256,
        'use_gti': True,
        'use_ranking': True,
        'use_focal': True,
        'lr': 0.0005,
        'epochs': 30,
        'batch_size': 512
    },
    'ziln_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'split_criterion': 'ziln',
        'use_smoothing': True
    }
}

# Dataset specific settings
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'synthetic': {
        'n_samples': 5000,
        'n_uid': 5000,
        'n_pid': 50000,
    }
}
