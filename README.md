# VALOR: Value-Aware Learning for Optimized Revenue

VALOR is a research framework for uplift modeling, specifically designed for revenue optimization in B2B and e-commerce scenarios. It leverages Zero-Inflated Log-Normal (ZILN) distributions, Treatment-Gated Interactions (GTI), and Value-Weighted Ranking losses to accurately estimate Individual Treatment Effects (ITE) on revenue.

## Repository Structure

```text
VALOR/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── main.py                 # CLI entry point for experiments
├── config.py               # Hyperparameters and model mappings
├── data/
│   ├── __init__.py
│   ├── loaders.py          # Data preprocessing and DataLoader utilities
│   └── generators.py       # Synthetic data generation logic (ZILN-compatible)
├── models/
│   ├── __init__.py
│   ├── layers.py           # Reusable components (GTI, ZILNHead, MoE)
│   ├── baselines.py        # Baseline models (S/T-Learner, TARNet, DragonNet, etc.)
│   ├── valor.py            # VALOR-specific architectures
│   └── trees.py            # Robust ZILN Forest implementation
├── training/
│   ├── __init__.py
│   ├── losses.py           # ZILN, Focal, and Ranking loss functions
│   └── trainers.py         # Training and validation loops
└── utils/
    ├── __init__.py
    └── metrics.py          # Uplift metrics (AUUC, Qini, Lift@k, Kendall)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run an experiment with the VALOR-enhanced TARNet model on synthetic data:

```bash
python main.py --model valor_tarnet --dataset synthetic --runs 5
```

### Supported Models

- **Baselines**: `tlearner`, `tarnet`, `dragonnet`, `cfrnet`, `cfrnet_wass`, `unite`, `euen`, `causal_forest`
- **RERUM**: `rerum` (Robust Evaluation and Ranking for Uplift Modeling)
- **VALOR Variants**: `valor_tarnet`, `valor_dragonnet`, `valor_unite`, `valor_euen`
- **Tree Models**: `ziln_forest`

## Key Features

- **ZILN Loss**: Specifically designed to handle zero-inflated continuous outcomes like revenue.
- **Treatment-Gated Interaction (GTI)**: A mechanism to learn feature interactions that are sensitive to treatment assignment.
- **Value-Weighted Ranking Loss**: Optimizes the ranking of users by their expected revenue uplift.
- **Bayesian Smoothing**: Used in ZILN Forest to handle sparse data in tree leaves.

## Citation

If you use this code in your research, please cite the VALOR paper:

```bibtex
@article{valor2026,
  title={VALOR: Value-Aware Learning for Optimized Revenue},
  author={...},
  journal={...},
  year={2026}
}
```
