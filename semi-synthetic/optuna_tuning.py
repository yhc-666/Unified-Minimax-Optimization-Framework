#!/usr/bin/env python
"""
Optuna hyperparameter tuning for semi-synthetic evaluation models.
Optimizes models (MF_DR_JL, MF_MRDR_JL, MF_Minimax, MF_DR_BIAS, MF_DR_BMSE, MF_DR_DCE)
for different objectives (ECE, BMSE, DR_Bias, DR_Variance).
"""

import argparse
import os
import json
import time
import csv
from typing import Dict, Any
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
import pandas as pd

# Import functions from evaluation.py
from evaluation import (
    load_data, 
    create_train_test_split,
    train_and_evaluate_model,
    compute_ece_torch,
    compute_bmse_torch,
    compute_dr_bias_torch,
    compute_dr_variance_torch,
    get_phi_normalized,
    get_model_propensity_scores
)

# Hyperparameter search ranges
HYPERPARAM_RANGES = {
    # Common parameters for all models
    'common': {
        'epochs': [150],
        'prop_epochs': [100],
        'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
        'prop_lr': [0.001, 0.005, 0.01, 0.05, 0.1],
        'batch_size': [64, 128, 256],
        'embedding_k': [4, 8, 16, 32],
        'gamma': [0.01, 0.05, 0.1, 0.2],
        'G': [1, 2, 4, 6, 8, 10]
    },
    # Model-specific parameters
    'MF_Minimax': {
        'beta': [0.1, 0.5, 1, 5, 10]
    },
    'MF_DR_BMSE': {
        'bmse_weight': [0.1, 0.5, 1, 5, 10, 15, 20]
    },
    'MF_DR_DCE': {
        'ece_weight': [1, 5, 10, 20, 50],
        'n_bins': [5, 10, 15, 20]
    }
}


def create_trial_params(trial: optuna.Trial, model_name: str, args) -> Dict[str, Any]:
    """Create hyperparameters for a trial based on the model."""
    
    # Common parameters
    params = {
        'epochs': trial.suggest_categorical('epochs', HYPERPARAM_RANGES['common']['epochs']),
        'prop_epochs': trial.suggest_categorical('prop_epochs', HYPERPARAM_RANGES['common']['prop_epochs']),
        'lr': trial.suggest_categorical('lr', HYPERPARAM_RANGES['common']['lr']),
        'prop_lr': trial.suggest_categorical('prop_lr', HYPERPARAM_RANGES['common']['prop_lr']),
        'batch_size': trial.suggest_categorical('batch_size', HYPERPARAM_RANGES['common']['batch_size']),
        'embedding_k': trial.suggest_categorical('embedding_k', HYPERPARAM_RANGES['common']['embedding_k']),
        'gamma': trial.suggest_categorical('gamma', HYPERPARAM_RANGES['common']['gamma']),
        'G': trial.suggest_categorical('G', HYPERPARAM_RANGES['common']['G']),
        'verbose': False,  # Disable verbose during optimization
        'test_ratio': args.test_ratio,
        'unbiased_test_ratio': args.unbiased_test_ratio,
        'seed': args.seed,
        'save_results': None  # Don't save individual results during optimization
    }
    
    # Model-specific parameters
    if model_name == 'MF_Minimax':
        params['beta'] = trial.suggest_categorical('beta', HYPERPARAM_RANGES['MF_Minimax']['beta'])
    elif model_name == 'MF_DR_BMSE':
        params['bmse_weight'] = trial.suggest_categorical('bmse_weight', HYPERPARAM_RANGES['MF_DR_BMSE']['bmse_weight'])
    elif model_name == 'MF_DR_DCE':
        params['ece_weight'] = trial.suggest_categorical('ece_weight', HYPERPARAM_RANGES['MF_DR_DCE']['ece_weight'])
        params['n_bins'] = trial.suggest_categorical('n_bins', HYPERPARAM_RANGES['MF_DR_DCE']['n_bins'])
    
    # Add default values for parameters not being optimized
    if model_name != 'MF_Minimax':
        params['beta'] = 1.0  # Default value
    if model_name != 'MF_DR_BMSE':
        params['bmse_weight'] = 1.0  # Default value
    if model_name != 'MF_DR_DCE':
        params['ece_weight'] = 10.0  # Default value
        params['n_bins'] = 10  # Default value
    
    return params


def objective(trial: optuna.Trial, args, data_splits: Dict) -> float:
    """Objective function for Optuna optimization."""
    
    # Create hyperparameters for this trial
    trial_params = create_trial_params(trial, args.model, args)
    
    # Convert params to args-like object
    class Args:
        pass
    
    trial_args = Args()
    for key, value in trial_params.items():
        setattr(trial_args, key, value)
    
    # Add model list for train_and_evaluate_model
    trial_args.models = [args.model]
    
    try:
        # Train and evaluate model
        start_time = time.time()
        model, predictions, metrics = train_and_evaluate_model(args.model, data_splits, trial_args)
        training_time = time.time() - start_time
        
        # Get the objective metric
        objective_value = metrics[args.objective]
        
        # Store all metrics as user attributes
        for metric_name, value in metrics.items():
            if metric_name != 'model':
                trial.set_user_attr(metric_name, value)
        trial.set_user_attr('training_time', training_time)
        
        # Save trial to CSV if requested
        if args.save_all_trials:
            save_trial_to_csv(trial, metrics, args, trial_params)
        
        return objective_value
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return worst possible value
        return float('inf') if args.direction == 'minimize' else float('-inf')


def save_trial_to_csv(trial: optuna.Trial, metrics: Dict, args, params: Dict):
    """Save trial results to CSV file."""
    
    output_dir = os.path.join('optuna_results', args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f'{args.model}_{args.objective}_all_trials.csv')
    
    # Prepare row data
    row_data = {
        'trial_number': trial.number,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'objective_value': trial.value if hasattr(trial, 'value') and trial.value is not None else None,
        **params,
        **{k: v for k, v in metrics.items() if k != 'model'}
    }
    
    # Write header if file doesn't exist
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def save_best_params(study: optuna.Study, args):
    """Save best parameters to JSON file."""
    
    output_dir = os.path.join('optuna_results', args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    best_params_path = os.path.join(output_dir, f'{args.model}_{args.objective}_best_params.json')
    
    best_trial = study.best_trial
    best_params = {
        'model': args.model,
        'objective': args.objective,
        'best_value': best_trial.value,
        'trial_number': best_trial.number,
        'params': best_trial.params,
        'all_metrics': best_trial.user_attrs
    }
    
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)


def save_summary(study: optuna.Study, args, data_splits: Dict):
    """Save optimization summary to text file."""
    
    output_dir = os.path.join('optuna_results', args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, f'{args.model}_{args.objective}_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"Optuna Hyperparameter Optimization Summary\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Objective: {args.objective} ({args.direction})\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best value: {study.best_value:.6f}\n")
        f.write(f"Best trial: {study.best_trial.number}\n\n")
        
        f.write("Best Parameters:\n")
        f.write("-"*40 + "\n")
        for param, value in study.best_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\nAll Metrics for Best Trial:\n")
        f.write("-"*40 + "\n")
        for metric, value in study.best_trial.user_attrs.items():
            if isinstance(value, float):
                f.write(f"{metric}: {value:.6f}\n")
            else:
                f.write(f"{metric}: {value}\n")
        
        # Add top 5 trials
        f.write("\nTop 5 Trials:\n")
        f.write("-"*40 + "\n")
        trials_df = study.trials_dataframe()
        top_trials = trials_df.nsmallest(5, 'value') if args.direction == 'minimize' else trials_df.nlargest(5, 'value')
        
        for idx, trial_row in top_trials.iterrows():
            f.write(f"\nTrial {int(trial_row['number'])}:\n")
            f.write(f"  {args.objective}: {trial_row['value']:.6f}\n")
            # Show key parameters
            for param in ['epochs', 'lr', 'embedding_k', 'gamma', 'G']:
                if f'params_{param}' in trial_row:
                    f.write(f"  {param}: {trial_row[f'params_{param}']}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter tuning for semi-synthetic evaluation models')
    
    # Model and objective
    parser.add_argument('--model', type=str, required=True,
                       choices=['MF_DR_JL', 'MF_MRDR_JL', 'MF_Minimax', 'MF_DR_BIAS', 'MF_DR_BMSE', 'MF_DR_DCE'],
                       help='Model to optimize')
    parser.add_argument('--objective', type=str, required=True,
                       choices=['ECE', 'BMSE', 'DR_Bias', 'DR_Variance'],
                       help='Objective metric to optimize')
    parser.add_argument('--direction', type=str, default='minimize',
                       choices=['minimize', 'maximize'],
                       help='Optimization direction')
    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                       help='Database URL for distributed optimization')
    
    # Data settings (same as evaluation.py)
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    parser.add_argument('--unbiased_test_ratio', type=float, default=0.2,
                       help='Ratio of test set to use as unbiased sample')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')
    
    # Output settings
    parser.add_argument('--save_all_trials', action='store_true',
                       help='Save all trial results to CSV')
    parser.add_argument('--output_dir', type=str, default='optuna_results',
                       help='Base directory for saving results')
    
    args = parser.parse_args()
    
    # Create default study name if not provided
    if args.study_name is None:
        args.study_name = f'{args.model}_{args.objective}_{time.strftime("%Y%m%d_%H%M%S")}'
    
    return args


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"Optuna Hyperparameter Optimization")
    print(f"="*60)
    print(f"Model: {args.model}")
    print(f"Objective: {args.objective} ({args.direction})")
    print(f"Number of trials: {args.n_trials}")
    print(f"Study name: {args.study_name}")
    
    # Load data once
    print("\nLoading data...")
    ground_truth, propensity, num_users, num_items = load_data(verbose=False)
    
    # Create train/test splits once
    print("Creating train/test splits...")
    data_splits = create_train_test_split(
        ground_truth, propensity, num_users, num_items,
        test_ratio=args.test_ratio,
        unbiased_test_ratio=args.unbiased_test_ratio,
        random_state=args.seed,
        verbose=False
    )
    
    # Create Optuna study
    if args.storage:
        # For distributed optimization
        study = optuna.create_study(
            study_name=args.study_name,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=True,
            sampler=TPESampler(seed=args.seed)
        )
    else:
        # Local optimization
        study = optuna.create_study(
            study_name=args.study_name,
            direction=args.direction,
            sampler=TPESampler(seed=args.seed)
        )
    
    # Run optimization
    print(f"\nStarting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, args, data_splits),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Save results
    print("\nSaving results...")
    save_best_params(study, args)
    save_summary(study, args, data_splits)
    
    # Print summary
    print(f"\nOptimization completed!")
    print(f"Best {args.objective}: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"\nBest parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    output_dir = os.path.join('optuna_results', args.model)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

#   python semi-synthetic/optuna_tuning.py --model MF_DR_JL --objective ECE --n_trials 100 --save_all_trials
#   python semi-synthetic/optuna_tuning.py --model MF_DR_BMSE --objective BMSE --n_trials 60 --save_all_trials
