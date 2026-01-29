"""Main entry point for running VALOR experiments.

Usage:
    python main.py --model valor_tarnet --dataset synthetic --runs 5
"""

import argparse
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any

from VALOR.config import MODEL_CONFIGS
from VALOR.data.generators import create_synthetic_data, set_random_seed
from VALOR.data.loaders import prepare_data
from VALOR.models.baselines import SLearner, TLearner, TARNet, DragonNet, UniTE, EUEN
from VALOR.models.valor import VALOR_TARNet, VALOR_DragonNet, VALOR_UniTE
from VALOR.models.trees import ZILNForestLearner
from VALOR.training.trainers import train_epoch, validate


def get_model(model_name: str, config: Any, prior_log_mean: float = 1.0):
    """Factory function to instantiate the requested model."""
    if model_name == 'slearner':
        return SLearner(config)
    elif model_name == 'tlearner':
        return TLearner(config)
    elif model_name == 'tarnet':
        return TARNet(config)
    elif model_name == 'dragonnet':
        return DragonNet(config)
    elif model_name == 'unite':
        return UniTE(config)
    elif model_name == 'euen':
        return EUEN(config)
    elif model_name == 'valor_tarnet':
        return VALOR_TARNet(config, prior_log_mean)
    elif model_name == 'valor_dragonnet':
        return VALOR_DragonNet(config, prior_log_mean)
    elif model_name == 'valor_unite':
        return VALOR_UniTE(config, prior_log_mean)
    elif model_name == 'ziln_forest':
        return ZILNForestLearner(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            split_criterion=config.split_criterion,
            use_smoothing=config.use_smoothing
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='VALOR Uplift Experiment Runner')
    parser.add_argument('--model', type=str, default='valor_tarnet', help='Model to run')
    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset to use')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--sample', action='store_true', help='Run on 10%% sample for debugging')
    args = parser.parse_args()

    # 1. Load Config
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Model {args.model} not in config.py")
    
    model_cfg_dict = MODEL_CONFIGS[args.model]
    
    # 2. Generate/Load Data
    if args.dataset == 'synthetic':
        df = create_synthetic_data()
    else:
        raise NotImplementedError(f"Dataset {args.dataset} loading not implemented yet.")

    # 3. Independent Runs
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run in range(args.runs):
        print(f"\n>>> Starting Run {run+1}/{args.runs} for {args.model}")
        set_random_seed(42 + run)

        train_loader, valid_loader, input_dim, *feature_groups = prepare_data(
            df, batch_size=model_cfg_dict.get('batch_size', 512), train_on_sample=args.sample
        )

        # Build Config object for models
        class SimpleConfig:
            def __init__(self, d, input_dim):
                for k, v in d.items():
                    setattr(self, k, v)
                self.input_dim = input_dim
                self.use_gpu = torch.cuda.is_available()
        
        config = SimpleConfig(model_cfg_dict, input_dim)
        if args.model == 'slearner': config.input_dim += 1

        # Prior for ZILN
        pos_labels = df[df['label'] > 0]['label']
        prior_log_mean = float(np.mean(np.log(pos_labels.values + 1e-5))) if len(pos_labels) > 0 else 0.0

        model = get_model(args.model, config, prior_log_mean)
        
        if args.model == 'ziln_forest':
            # Tree-based model fit
            X_train = []
            y_train = []
            t_train = []
            for batch in train_loader:
                uid_cont, uid_cat, pid_cont, pid_cat, treatment, labels, pid_cluster, _ = batch
                X_train.append(torch.cat([uid_cont, uid_cat, pid_cont, pid_cat], dim=1).numpy())
                y_train.append(labels.numpy())
                t_train.append(treatment.numpy())
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            t_train = np.concatenate(t_train)
            
            start_t = time.time()
            model.fit(X_train, y_train, t_train)
            train_time = time.time() - start_t
        else:
            # PyTorch model fit
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=model_cfg_dict.get('lr', 0.0005))
            
            start_t = time.time()
            for epoch in range(args.epochs):
                loss = train_epoch(model, train_loader, device, optimizer)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
            train_time = time.time() - start_t

        metrics = validate(model, valid_loader, device)
        metrics['train_time'] = train_time
        metrics['model'] = args.model
        metrics['run'] = run + 1
        results.append(metrics)

        print(f"Run {run+1} Complete: AUUC={metrics['auuc']-0.5:.4f}, Qini={metrics['qini']:.4f}")

    # 4. Final Summary
    res_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f" EXPERIMENT SUMMARY: {args.model} on {args.dataset} ")
    print("="*50)
    print(res_df.drop(columns=['model', 'run']).mean())


if __name__ == '__main__':
    import time
    main()
