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
from Minimax import MF_Minimax
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
        'embedding_k': [8, 16, 32],
        'embedding_k1': [8, 16, 32],
        'pred_lr': (0.005, 0.05),
        'impu_lr': (0.005, 0.05),
        'prop_lr': (0.005, 0.05),
        'dis_lr': (0.005, 0.05),
        'lamb_pred': (1e-6, 1e-3),
        'lamb_imp': (1e-3, 1.0),
        'lamb_prop': (1e-4, 1e-2),
        'dis_lamb': (0.0, 1e-3),
        'gamma': (0.01, 0.1),
        'beta': (1e-6, 1e-3),
        'G': [2, 4, 6, 8],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [2048, 4096, 8192],
        'abc_model_name': ['logistic_regression', 'mlp']
    },
    'kuai': {
        'embedding_k': [8, 16, 32],
        'embedding_k1': [8, 16, 32],
        'pred_lr': (0.005, 0.05),
        'impu_lr': (0.005, 0.05),
        'prop_lr': (0.005, 0.05),
        'dis_lr': (0.005, 0.05),
        'lamb_pred': (1e-6, 1e-3),
        'lamb_imp': (1e-3, 0.1),
        'lamb_prop': (1e-3, 0.1),
        'dis_lamb': (0.0, 1e-3),
        'gamma': (0.01, 0.1),
        'beta': (1e-6, 1e-3),
        'G': [2, 4, 6, 8],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [2048, 4096, 8192],
        'abc_model_name': ['logistic_regression', 'mlp']
    }
}

def train_and_eval_with_params(dataset_name, train_args, model_args):
    """Train and evaluate model with given hyperparameters"""
    
    # Set up data based on dataset
    top_k_list = [5]
    top_k_names = ("precision_5", "recall_5", "ndcg_5", "f1_5")
    
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
        top_k_list = [50]
        top_k_names = ("precision_50", "recall_50", "ndcg_50", "f1_50")

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
    
    # Calculate F1
    f1 = 2 / (1 / np.mean(precisions[top_k_names[0]]) + 1 / np.mean(recalls[top_k_names[1]]))
    
    return {
        'mse': mse,
        'mae': mae,
        'auc': auc,
        'ndcg': np.mean(ndcgs[top_k_names[2]]),
        'precision': np.mean(precisions[top_k_names[0]]),
        'recall': np.mean(recalls[top_k_names[1]]),
        'f1': f1
    }


def objective(trial, args):
    """Optuna objective function"""
    
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
        
        # Return the metric to optimize
        return results[args.metric]
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf') if args.direction == 'maximize' else float('inf')


def save_trial_to_csv(trial, results, args):
    """Save trial results to CSV file"""
    csv_path = os.path.join(args.output_dir, f'{args.dataset}_trials.csv')
    
    # Prepare row data
    row_data = {
        'trial_number': trial.number,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **trial.params,
        **results,
        'training_time': trial.user_attrs.get('training_time', -1)
    }
    
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
    parser.add_argument('--metric', type=str, default='auc',
                        choices=['auc', 'ndcg', 'recall', 'f1', 'mse'],
                        help='Metric to optimize')
    parser.add_argument('--direction', type=str, default='maximize',
                        choices=['maximize', 'minimize'],
                        help='Optimization direction')
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Adjust direction based on metric
    if args.metric == 'mse':
        args.direction = 'minimize'
    
    # Create study name
    if args.study_name is None:
        args.study_name = f'{args.dataset}_{args.metric}_{time.strftime("%Y%m%d_%H%M%S")}'
    
    # Create optuna study
    if args.storage:
        # For distributed optimization
        study = create_study(
            study_name=args.study_name,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=True,
            sampler=TPESampler(seed=args.seed)
        )
    else:
        # Local optimization
        study = create_study(
            study_name=args.study_name,
            direction=args.direction,
            sampler=TPESampler(seed=args.seed)
        )
    
    # Run optimization
    print(f"Starting hyperparameter search for {args.dataset} dataset")
    print(f"Optimizing {args.metric} ({args.direction})")
    print(f"Running {args.n_trials} trials...")
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    # Print results
    print("\n" + "="*50)
    print("Best trial:")
    print(f"Value ({args.metric}): {study.best_value:.6f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params_path = os.path.join(args.output_dir, f'{args.dataset}_best_params.csv')
    with open(best_params_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['parameter', 'value'])
        writer.writeheader()
        for key, value in study.best_params.items():
            writer.writerow({'parameter': key, 'value': value})
    
    # Save study summary
    summary_path = os.path.join(args.output_dir, f'{args.dataset}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Metric: {args.metric} ({args.direction})\n")
        f.write(f"Number of trials: {args.n_trials}\n")
        f.write(f"Best value: {study.best_value:.6f}\n")
        f.write(f"\nBest parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        
        # Add all metrics from best trial
        best_trial = study.best_trial
        f.write(f"\nAll metrics from best trial:\n")
        for key, value in best_trial.user_attrs.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()