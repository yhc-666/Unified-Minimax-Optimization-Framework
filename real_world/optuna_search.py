#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna hyperparameter search for MF_Minimax model
Evaluates directly on test set with customizable hyperparameter ranges
"""

import optuna
from optuna import create_study
from optuna.samplers import TPESampler
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from matrix_factorization_DT import MF_Minimax
from dataset import load_data
from utils import ndcg_func, recall_func, precision_func, rating_mat_to_sample, binarize, shuffle
from matrix_factorization_DT import generate_total_sample
import csv
import os
import time
import scipy.sparse as sps

# Metric functions
mse_func = lambda x,y: np.mean((x-y)**2)
mae_func = lambda x,y: np.mean(np.abs(x-y))

# Hyperparameter search ranges for each dataset
HYPERPARAM_RANGES = {
    'coat': {
        'embedding_k': [4, 8, 16],
        'embedding_k1': [4, 8, 16],
        'pred_lr': (0.005, 0.01),
        'impu_lr': (0.005, 0.01),
        'prop_lr': (0.005, 0.01),
        'dis_lr': (0.005, 0.01),
        'lamb_pred': (1e-7, 10),
        'lamb_imp': (1e-7, 10),
        'lamb_prop': (1e-7, 10),
        'dis_lamb': (1e-7, 10),
        'gamma': (0.01, 0.2),
        'beta': (0.01, 1.0),
        'G': [2, 4, 6, 8],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [64, 128, 256],
        'abc_model_name': ['logistic_regression', 'mlp']
    },
    'yahoo': {
        'embedding_k': [4, 8, 16, 32, 64, 256],
        'embedding_k1': [4, 8, 16, 32, 64, 256],
        'pred_lr': (0.005, 0.05),
        'impu_lr': (0.005, 0.05),
        'prop_lr': (0.005, 0.05),
        'dis_lr': (0.005, 0.05),
        'lamb_pred': (1e-7, 10),
        'lamb_imp': (1e-7, 10),
        'lamb_prop': (1e-7, 10),
        'dis_lamb': (1e-7, 10),
        'gamma': (0.01, 0.2),
        'beta': (1e-6, 10),
        'G': [2, 4, 6, 8],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [2048, 4096, 8192],
        'abc_model_name': ['logistic_regression', 'mlp']
    },
    'kuai': {
        'embedding_k': [4, 8, 16, 32, 64, 256],
        'embedding_k1': [4, 8, 16, 32, 64, 256],
        'pred_lr': (0.005, 0.05),
        'impu_lr': (0.005, 0.05),
        'prop_lr': (0.005, 0.05),
        'dis_lr': (0.005, 0.05),
        'lamb_pred': (1e-7, 10),
        'lamb_imp': (1e-7, 10),
        'lamb_prop': (1e-7, 10),
        'dis_lamb': (1e-7, 10),
        'gamma': (0.01, 0.2),
        'beta': (1e-6, 10),
        'G': [2, 4, 6, 8],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [2048, 4096, 8192],
        'abc_model_name': ['logistic_regression', 'mlp']
    }
}

def train_and_eval_with_params(dataset_name, train_args, model_args):
    """Train and evaluate model with given hyperparameters"""
    
    # Set up data based on dataset
    top_k_list = [5, 10]
    
    if dataset_name == "coat":
        train_mat, test_mat = load_data("coat")        
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]

    elif dataset_name == "yahoo":
        x_train, y_train, x_test, y_test = load_data("yahoo")
        x_train, y_train = shuffle(x_train, y_train)
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1

    elif dataset_name == "kuai":
        x_train, y_train, x_test, y_test = load_data("kuai")
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1
        top_k_list = [50, 100]

    # Binarize ratings
    if dataset_name == "kuai":
        y_train = binarize(y_train, 1)
        y_test = binarize(y_test, 1)
    else:
        y_train = binarize(y_train)
        y_test = binarize(y_test)

    # Create model
    mf = MF_Minimax(num_user, num_item, 
                    batch_size=train_args['batch_size'], 
                    batch_size_prop=train_args['batch_size_prop'],
                    embedding_k=model_args['embedding_k'], 
                    embedding_k1=model_args['embedding_k1'],
                    abc_model_name=model_args.get('abc_model_name', 'logistic_regression'),
                    copy_model_pred=model_args.get('copy_model_pred', 1))
    
    # First compute propensity scores
    mf._compute_IPS(x_train, 
                    num_epoch=200,  # Fixed for propensity pre-training
                    lr=model_args.get('prop_lr', 0.01), 
                    lamb=model_args.get('prop_lamb', model_args['lamb_prop']),
                    verbose=False)
    
    # Then train the full model
    mf.fit(x_train, y_train, 
           pred_lr=model_args['pred_lr'], 
           impu_lr=model_args['impu_lr'],
           prop_lr=model_args['prop_lr'],
           dis_lr=model_args['dis_lr'],
           alpha=train_args.get('alpha', 0.5), 
           beta=train_args['beta'], 
           theta=train_args.get('theta', 1),
           lamb_prop=model_args['lamb_prop'],
           lamb_pred=model_args['lamb_pred'],
           lamb_imp=model_args['lamb_imp'],
           dis_lamb=model_args['dis_lamb'],
           G=train_args["G"],
           gamma=train_args['gamma'],
           num_bins=train_args.get('num_bins', 10),
           verbose=False)

    # Evaluate
    test_pred = mf.predict(x_test)
    mse = mse_func(y_test, test_pred)
    mae = mae_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)
    
    # Build results dictionary with all metrics
    results = {
        'mse': mse,
        'mae': mae,
        'auc': auc,
    }
    
    # Add metrics for each k value
    for k in top_k_list:
        precision_key = f"precision_{k}"
        recall_key = f"recall_{k}"
        ndcg_key = f"ndcg_{k}"
        
        # Calculate F1 for this k
        f1_k = 2 / (1 / np.mean(precisions[precision_key]) + 1 / np.mean(recalls[recall_key]))
        
        results[f'ndcg_{k}'] = np.mean(ndcgs[ndcg_key])
        results[f'precision_{k}'] = np.mean(precisions[precision_key])
        results[f'recall_{k}'] = np.mean(recalls[recall_key])
        results[f'f1_{k}'] = f1_k
    
    # For backward compatibility, also include the first k value without suffix
    results['ndcg'] = results[f'ndcg_{top_k_list[0]}']
    results['precision'] = results[f'precision_{top_k_list[0]}']
    results['recall'] = results[f'recall_{top_k_list[0]}']
    results['f1'] = results[f'f1_{top_k_list[0]}']
    
    return results


def objective(trial, args):
    """Optuna objective function for multi-objective optimization"""
    
    # Get hyperparameter ranges for the dataset
    ranges = HYPERPARAM_RANGES[args.dataset]
    
    # Suggest hyperparameters
    train_args = {
        'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
        'batch_size_prop': trial.suggest_categorical('batch_size', ranges['batch_size']),  # Same as batch_size
        'gamma': trial.suggest_float('gamma', *ranges['gamma']),
        'G': trial.suggest_categorical('G', ranges['G']),
        'beta': trial.suggest_float('beta', *ranges['beta'], log=True),
        'num_bins': trial.suggest_categorical('num_bins', ranges['num_bins']),
        'alpha': 0.5,  # Unused parameter
        'theta': 1     # Unused parameter
    }
    
    model_args = {
        'embedding_k': trial.suggest_categorical('embedding_k', ranges['embedding_k']),
        'embedding_k1': trial.suggest_categorical('embedding_k1', ranges['embedding_k1']),
        'pred_lr': trial.suggest_float('pred_lr', *ranges['pred_lr'], log=True),
        'impu_lr': trial.suggest_float('impu_lr', *ranges['impu_lr'], log=True),
        'prop_lr': trial.suggest_float('prop_lr', *ranges['prop_lr'], log=True),
        'dis_lr': trial.suggest_float('dis_lr', *ranges['dis_lr'], log=True),
        'lamb_pred': trial.suggest_float('lamb_pred', *ranges['lamb_pred'], log=True),
        'lamb_imp': trial.suggest_float('lamb_imp', *ranges['lamb_imp'], log=True),
        'lamb_prop': trial.suggest_float('lamb_prop', *ranges['lamb_prop'], log=True),
        'dis_lamb': trial.suggest_float('dis_lamb', *ranges['dis_lamb']),
        'abc_model_name': trial.suggest_categorical('abc_model_name', ranges['abc_model_name']),
        'copy_model_pred': 1
    }
    
    try:
        # Train and evaluate
        start_time = time.time()
        results = train_and_eval_with_params(args.dataset, train_args, model_args)
        training_time = time.time() - start_time
        
        # Log all metrics
        for metric_name, value in results.items():
            trial.set_user_attr(metric_name, value)
        trial.set_user_attr('training_time', training_time)
        
        # Save to CSV
        if args.save_all_trials:
            save_trial_to_csv(trial, results, args)
        
        # Return tuple of metrics to optimize
        objective_values = []
        for metric in args.metrics:
            objective_values.append(results[metric])
        
        return tuple(objective_values)
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return worst values for each objective
        worst_values = []
        for i, direction in enumerate(args.directions):
            worst_values.append(float('-inf') if direction == 'maximize' else float('inf'))
        return tuple(worst_values)


def save_trial_to_csv(trial, results, args):
    """Save trial results to CSV file"""
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    csv_path = os.path.join(dataset_output_dir, f'{args.dataset}_all_trials.csv')
    
    # Prepare row data
    row_data = {
        'trial_number': trial.number,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **trial.params,
        **results,
        'training_time': trial.user_attrs.get('training_time', -1)
    }
    
    # Add objective values for multi-objective optimization
    if hasattr(trial, 'values') and trial.values is not None:
        for i, (metric, value) in enumerate(zip(args.metrics, trial.values)):
            row_data[f'objective_{metric}'] = value
    
    # Write header if file doesn't exist
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search for MF_Minimax')
    parser.add_argument('--dataset', type=str, default='coat', 
                        choices=['coat', 'yahoo', 'kuai'],
                        help='Dataset to use')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optuna trials')
    parser.add_argument('--metrics', type=str, nargs='+', default=['auc'],
                        choices=['auc', 'ndcg', 'recall', 'f1', 'mse', 'mae', 'precision',
                                 'ndcg_5', 'ndcg_10', 'ndcg_50', 'ndcg_100',
                                 'precision_5', 'precision_10', 'precision_50', 'precision_100',
                                 'recall_5', 'recall_10', 'recall_50', 'recall_100',
                                 'f1_5', 'f1_10', 'f1_50', 'f1_100'],
                        help='Metrics to optimize (can specify multiple)')
    parser.add_argument('--directions', type=str, nargs='+', default=['maximize'],
                        choices=['maximize', 'minimize'],
                        help='Optimization directions for each metric')
    parser.add_argument('--output_dir', type=str, default='./optuna_results',
                        help='Directory for saving results')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')
    parser.add_argument('--save_all_trials', action='store_true',
                        help='Save all trial results to CSV')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database URL for distributed optimization')
    
    args = parser.parse_args()
    
    # Validate that metrics and directions have the same length
    if len(args.metrics) != len(args.directions):
        parser.error(f"Number of metrics ({len(args.metrics)}) must match number of directions ({len(args.directions)})")
    
    return args


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectory for this dataset
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Create study name
    if args.study_name is None:
        metrics_str = '_'.join(args.metrics)
        args.study_name = f'{args.dataset}_{metrics_str}_{time.strftime("%Y%m%d_%H%M%S")}'
    
    # Create optuna study for multi-objective optimization
    if args.storage:
        # For distributed optimization
        study = create_study(
            study_name=args.study_name,
            directions=args.directions,
            storage=args.storage,
            load_if_exists=True,
            sampler=TPESampler(seed=args.seed)
        )
    else:
        # Local optimization
        study = create_study(
            study_name=args.study_name,
            directions=args.directions,
            sampler=TPESampler(seed=args.seed)
        )
    
    # Run optimization
    print(f"Starting multi-objective hyperparameter search for {args.dataset} dataset")
    print(f"Optimizing metrics: {', '.join([f'{m} ({d})' for m, d in zip(args.metrics, args.directions)])}")
    print(f"Running {args.n_trials} trials...")
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    # Get Pareto optimal trials
    pareto_trials = study.best_trials
    
    # Print results
    print("\n" + "="*50)
    print(f"Found {len(pareto_trials)} Pareto optimal solutions")
    
    # Save Pareto optimal parameters
    pareto_params_path = os.path.join(dataset_output_dir, f'{args.dataset}_pareto_optimal_params.csv')
    with open(pareto_params_path, 'w', newline='') as f:
        if pareto_trials:
            # Prepare fieldnames
            fieldnames = ['trial_number', 'pareto_rank']
            fieldnames.extend(args.metrics)
            fieldnames.extend(sorted(pareto_trials[0].params.keys()))
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, trial in enumerate(pareto_trials):
                row = {
                    'trial_number': trial.number,
                    'pareto_rank': idx + 1
                }
                # Add metric values
                for i, metric in enumerate(args.metrics):
                    row[metric] = trial.values[i]
                # Add parameters
                row.update(trial.params)
                writer.writerow(row)
    
    # Save study summary
    summary_path = os.path.join(dataset_output_dir, f'{args.dataset}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Metrics: {', '.join([f'{m} ({d})' for m, d in zip(args.metrics, args.directions)])}\n")
        f.write(f"Number of trials: {args.n_trials}\n")
        f.write(f"Number of Pareto optimal solutions: {len(pareto_trials)}\n\n")
        
        if pareto_trials:
            f.write("="*60 + "\n")
            f.write("PARETO OPTIMAL SOLUTIONS\n")
            f.write("="*60 + "\n\n")
            
            for idx, trial in enumerate(pareto_trials[:10]):  # Show top 10
                f.write(f"Pareto Solution #{idx+1} (Trial {trial.number}):\n")
                f.write("-"*40 + "\n")
                
                # Write objective values
                f.write("Objective values:\n")
                for i, (metric, value) in enumerate(zip(args.metrics, trial.values)):
                    f.write(f"  {metric}: {value:.6f}\n")
                
                # Write all metrics
                f.write("\nAll metrics:\n")
                for key, value in trial.user_attrs.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                
                # Write parameters
                f.write("\nParameters:\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            if len(pareto_trials) > 10:
                f.write(f"... and {len(pareto_trials) - 10} more Pareto optimal solutions\n")
    
    print(f"\nResults saved to {dataset_output_dir}")
    print(f"Pareto optimal parameters saved to: {pareto_params_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()