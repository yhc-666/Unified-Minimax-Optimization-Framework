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
from matrix_factorization_DT import MF_Minimax, MF_MinimaxV2, MF_MinimaxV3, MF_MinimaxV4, MF_DRv2_BMSE_Imp, MF_DRv2_BMSE
from dataset import load_data
from utils import ndcg_func, recall_func, precision_func, rating_mat_to_sample, binarize, shuffle, set_all_seeds, set_deterministic
from matrix_factorization_DT import generate_total_sample
import csv
import os
import time
import scipy.sparse as sps
import pandas as pd
from tqdm import tqdm

# Metric functions
mse_func = lambda x,y: np.mean((x-y)**2)
mae_func = lambda x,y: np.mean(np.abs(x-y))

# Hyperparameter search ranges for each dataset
HYPERPARAM_RANGES = {
    'coat': {
        'embedding_k': [16, 32, 64],
        'embedding_k1': [16, 32, 64],
        'pred_lr': [0.005, 0.01],
        'impu_lr': [0.005, 0.01],
        'prop_lr': [0.005, 0.01],
        'dis_lr': [0.005, 0.01],
        'lamb_pred': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'lamb_imp': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'lamb_prop': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'dis_lamb': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'gamma': (0.01, 0.05),
        'beta': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        'G': [1, 2, 4, 6, 8, 10, 12],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [128],
        'abc_model_name': ['logistic_regression', 'mlp']
    },
    'yahoo': {
        'embedding_k': [16, 32, 64],
        'embedding_k1': [16, 32, 64],
        'pred_lr': [0.005, 0.01],
        'impu_lr': [0.005, 0.01],
        'prop_lr': [0.005, 0.01],
        'dis_lr': [0.005, 0.01],
        'lamb_pred': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'lamb_imp': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'lamb_prop': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'dis_lamb': [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'gamma': (0.01, 0.05),
        'beta': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        'G': [1, 2, 4, 6, 8, 10, 12],
        'num_bins': [5, 10, 15, 20],
        'batch_size': [4096],
        'abc_model_name': ['logistic_regression', 'mlp']
    },
    'kuai': {
        'embedding_k': [16, 32, 64],
        'embedding_k1': [16, 32, 64],
        'pred_lr': [0.005, 0.01],
        'impu_lr': [0.005, 0.01],
        'prop_lr': [0.005, 0.01],
        'dis_lr': [0.005, 0.01],
        'lamb_pred': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01],
        'lamb_imp': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01],
        'lamb_prop': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01],
        'dis_lamb': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01],
        'gamma': (0.01, 0.1),
        'beta': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        'G': [1, 2, 4, 6, 8, 10, 12],
        'num_bins': [5, 10, 15, 20, 25, 30, 40, 50, 50],
        'batch_size': [4096],
        'abc_model_name': ['logistic_regression', 'mlp']
    }
}

# DR-V2 specific hyperparameter ranges
DRV2_HYPERPARAM_RANGES = {
    'coat': {
        'embedding_k': [16, 32, 64],  # For propensity model
        'embedding_k1': [16, 32, 64], # For prediction/imputation models
        'embedding_k_prop': [16, 32, 64], # For DRv2_Imp propensity model
        'pred_lr': [0.01, 0.05],
        'impu_lr': [0.005, 0.01],  # Only for DRv2_Imp
        'prop_lr': [0.01, 0.05],
        'lamb_pred': [0.001, 0.005, 0.01],
        'lamb_imp': [5e-5, 1e-4, 5e-4],  # Only for DRv2_Imp
        'lamb_prop': [5e-4, 0.001, 0.005],
        'alpha': [1],  # Fixed as it's unused
        'beta': [0.1, 0.5, 1, 2],
        'gamma': (0.01, 0.05),
        'imputation': [1e-4, 1e-3, 1e-2],  # Only for DRv2_Imp
        'batch_size': [128],
        'num_epoch': [500]
    },
    'yahoo': {
        'embedding_k': [64, 128, 256],
        'embedding_k1': [64, 128, 256],
        'embedding_k_prop': [32, 64, 128],
        'pred_lr': [0.001, 0.005],
        'impu_lr': [0.001, 0.005],  # Only for DRv2_Imp
        'prop_lr': [0.001, 0.005],
        'lamb_pred': [1e-6, 1e-5, 1e-4],
        'lamb_imp': [1e-6, 1e-5, 1e-4],  # Only for DRv2_Imp
        'lamb_prop': [1e-5, 1e-4, 1e-3],
        'alpha': [1],  # Fixed as it's unused
        'beta': [0.5, 1, 2, 5],
        'gamma': (0.01, 0.1),
        'imputation': [0.5, 1, 5],  # Only for DRv2_Imp
        'batch_size': [4096],
        'num_epoch': [500]
    },
    'kuai': {
        'embedding_k': [64, 128, 256],
        'embedding_k1': [64, 128, 256],
        'embedding_k_prop': [64, 128, 256],
        'pred_lr': [0.005, 0.01],
        'impu_lr': [0.005, 0.01],  # Only for DRv2_Imp
        'prop_lr': [0.005, 0.01],
        'lamb_pred': [5e-4, 1e-3, 5e-3],
        'lamb_imp': [5e-5, 1e-4, 5e-4],  # Only for DRv2_Imp
        'lamb_prop': [1e-4, 5e-4, 1e-3],
        'alpha': [1],  # Fixed as it's unused
        'beta': [1, 3, 5, 10],
        'gamma': (0.01, 0.1, 1, 5, 10),
        'imputation': [1e-4, 1e-3, 1e-2],  # Only for DRv2_Imp
        'batch_size': [4096],
        'num_epoch': [500]
    }
}

def train_and_eval_with_params(dataset_name, train_args, model_args, seed=2020):
    """Train and evaluate model with given hyperparameters"""
    
    # Set seeds for reproducibility
    set_all_seeds(seed)
    set_deterministic()
    
    # Set up data based on dataset
    top_k_list = [5]
    
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

    elif dataset_name == "kuai": # 改为使用CaliMR处理对齐
        # x_train, y_train, x_test, y_test = load_data("kuai")
        rdf_train = np.array(pd.read_table("real_world/data/CaliMR_kuai/user.txt", header = None, sep = ',')).astype(float)
        rdf_test = np.array(pd.read_table("real_world/data/CaliMR_kuai/random.txt", header = None, sep = ',')).astype(float)
        rdf_train_new = np.c_[rdf_train, np.ones(rdf_train.shape[0])]
        rdf_test_new = np.c_[rdf_test, np.zeros(rdf_test.shape[0])]
        rdf = np.r_[rdf_train_new, rdf_test_new]

        rdf = rdf[np.argsort(rdf[:, 0])]
        c = rdf.copy()
        for i in range(rdf.shape[0]):
            if i == 0:
                c[:, 0][i] = i
                temp = rdf[:, 0][0]
            else:
                if c[:, 0][i] == temp:
                    c[:, 0][i] = c[:, 0][i-1]
                else:
                    c[:, 0][i] = c[:, 0][i-1] + 1
                temp = rdf[:, 0][i]

        c = c[np.argsort(c[:, 1])]
        d = c.copy()
        for i in range(rdf.shape[0]):
            if i == 0:
                d[:, 1][i] = i
                temp = c[:, 1][0]
            else:
                if d[:, 1][i] == temp:
                    d[:, 1][i] = d[:, 1][i-1]
                else:
                    d[:, 1][i] = d[:, 1][i-1] + 1
                temp = c[:, 1][i]

        y_train = d[:, 2][d[:, 3] == 1].astype(int)
        y_test = d[:, 2][d[:, 3] == 0].astype(int)
        x_train = d[:, :2][d[:, 3] == 1].astype(int)
        x_test = d[:, :2][d[:, 3] == 0].astype(int)
        num_user = int(x_train[:,0].max()) + 1
        num_item = int(x_train[:,1].max()) + 1
        top_k_list = [20]

    # Binarize ratings
    if dataset_name == "kuai":
        y_train = binarize(y_train, 2)
        y_test = binarize(y_test, 2)
    else:
        y_train = binarize(y_train, 3)
        y_test = binarize(y_test, 3)

    # Create model
    if model_args.get('model_type') == 'minimaxv2':
        # For MinimaxV2, use unified embedding size
        mf = MF_MinimaxV2(num_user, num_item, 
                         batch_size=train_args['batch_size'], 
                         batch_size_prop=train_args['batch_size_prop'],
                         embedding_k=model_args['embedding_k'], 
                         embedding_k1=model_args['embedding_k'],  # Same as embedding_k for unified size
                         abc_model_name=model_args.get('abc_model_name', 'logistic_regression'),
                         copy_model_pred=model_args.get('copy_model_pred', 1))
    elif model_args.get('model_type') == 'minimaxv3':
        # For MinimaxV3, use enhanced architecture with dropout
        mf = MF_MinimaxV3(num_user, num_item, 
                         batch_size=train_args['batch_size'], 
                         batch_size_prop=train_args['batch_size_prop'],
                         embedding_k=model_args['embedding_k'], 
                         embedding_k1=model_args['embedding_k1'],
                         dropout_rate=model_args.get('dropout_rate', 0.2),
                         abc_model_name=model_args.get('abc_model_name', 'mlp_enhanced'),
                         copy_model_pred=model_args.get('copy_model_pred', 1))
    elif model_args.get('model_type') == 'minimaxv4':
        # For MinimaxV4, use standard models with V3's training improvements
        mf = MF_MinimaxV4(num_user, num_item, 
                         batch_size=train_args['batch_size'], 
                         batch_size_prop=train_args['batch_size_prop'],
                         embedding_k=model_args['embedding_k'], 
                         embedding_k1=model_args['embedding_k1'],
                         abc_model_name=model_args.get('abc_model_name', 'logistic_regression'),
                         copy_model_pred=model_args.get('copy_model_pred', 1))
    else:
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
    
    # Create progress bar for training
    pbar = tqdm(total=1000, desc="Training MF_Minimax", unit="epoch", leave=False)
    last_train_auc = None
    last_test_auc = None
    
    def progress_callback(epoch, total_epochs, loss, train_auc=None, test_auc=None):
        nonlocal last_train_auc, last_test_auc
        
        # Update stored values
        if train_auc is not None:
            last_train_auc = train_auc
        if test_auc is not None:
            last_test_auc = test_auc
        
        # Update progress bar
        pbar.n = epoch + 1
        pbar.total = total_epochs  # Update total if different
        
        # Build description
        desc_parts = [f"Loss: {loss:.1f}"]
        if last_train_auc is not None:
            desc_parts.append(f"Train: {last_train_auc:.4f}")
        if last_test_auc is not None:
            desc_parts.append(f"Test: {last_test_auc:.4f}")
            # Add overfitting indicator
            if last_train_auc is not None and last_train_auc - last_test_auc > 0.2:
                desc_parts.append("⚠️ OVERFIT")
        
        pbar.set_description(" | ".join(desc_parts))
        pbar.refresh()
    
    # Then train the full model with early stopping
    fit_params = {
        'x': x_train,
        'y': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'pred_lr': model_args['pred_lr'],
        'impu_lr': model_args['impu_lr'],
        'prop_lr': model_args['prop_lr'],
        'dis_lr': model_args['dis_lr'],
        'alpha': train_args.get('alpha', 0.5),
        'beta': train_args['beta'],
        'theta': train_args.get('theta', 1),
        'lamb_prop': model_args['lamb_prop'],
        'lamb_pred': model_args['lamb_pred'],
        'lamb_imp': model_args['lamb_imp'],
        'dis_lamb': model_args['dis_lamb'],
        'G': train_args["G"],
        'gamma': train_args['gamma'],
        'num_bins': train_args.get('num_bins', 10),
        'verbose': False,
        'early_stop_patience': 15,
        'early_stop_min_delta': 1e-5,
        'eval_freq': 1,
        'progress_callback': progress_callback
    }
    
    # Add extra parameters for MinimaxV3 and MinimaxV4
    if model_args.get('model_type') in ['minimaxv3', 'minimaxv4']:
        fit_params['grad_clip_norm'] = model_args.get('grad_clip_norm', 1.0)
    
    mf.fit(**fit_params)
    
    # Close progress bar
    pbar.close()

    # Evaluate on training set
    train_pred = mf.predict(x_train)
    train_mse = mse_func(y_train, train_pred)
    train_mae = mae_func(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_pred)
    
    # Evaluate on test set
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
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_auc': train_auc,
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


def train_and_eval_drv2(dataset_name, train_args, model_args, use_imputation=True, seed=2020):
    """Train and evaluate DR-V2 model with given hyperparameters"""
    
    # Set seeds for reproducibility
    set_all_seeds(seed)
    set_deterministic()
    
    # Set up data based on dataset
    top_k_list = [5]
    
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
        top_k_list = [20]

    # Binarize ratings
    if dataset_name == "kuai":
        y_train = binarize(y_train, 2)
        y_test = binarize(y_test, 2)
    else:
        y_train = binarize(y_train, 3)
        y_test = binarize(y_test, 3)

    # Create model based on imputation flag
    if use_imputation:
        mf = MF_DRv2_BMSE_Imp(
            num_user, num_item, 
            batch_size=train_args['batch_size'], 
            batch_size_prop=train_args['batch_size_prop'],
            embedding_k=model_args.get('embedding_k', model_args['embedding_k1']), 
            embedding_k1=model_args['embedding_k1'],
            embedding_k_prop=model_args.get('embedding_k_prop', model_args.get('embedding_k', model_args['embedding_k1']))
        )
        
        # Train with imputation parameters
        mf.fit(x_train, y_train,
               lr=model_args['pred_lr'],
               impu_lr=model_args['impu_lr'],
               prop_lr=model_args['prop_lr'],
               lamb=model_args['lamb_pred'],
               lamb_imp=model_args['lamb_imp'],
               lamb_prop=model_args['lamb_prop'],
               alpha=train_args.get('alpha', 1),
               beta=train_args['beta'],
               gamma=train_args['gamma'],
               imputation=train_args['imputation'],
               num_epoch=train_args.get('num_epoch', 500),
               verbose=False)
    else:
        mf = MF_DRv2_BMSE(
            num_user, num_item,
            batch_size=train_args['batch_size'],
            batch_size_prop=train_args['batch_size_prop'],
            embedding_k=model_args.get('embedding_k', model_args['embedding_k1']),
            embedding_k1=model_args['embedding_k1']
        )
        
        # Train without imputation parameters
        mf.fit(x_train, y_train,
               lr=model_args['pred_lr'],
               prop_lr=model_args['prop_lr'],
               lamb=model_args['lamb_pred'],
               lamb_prop=model_args['lamb_prop'],
               alpha=train_args.get('alpha', 1),
               beta=train_args['beta'],
               gamma=train_args['gamma'],
               num_epoch=train_args.get('num_epoch', 500),
               verbose=False)

    # Evaluate on training set
    train_pred = mf.predict(x_train)
    train_mse = mse_func(y_train, train_pred)
    train_mae = mae_func(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_pred)
    
    # Evaluate on test set
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
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_auc': train_auc,
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
    
    # 目前支持DRV2与minimax
    if args.model_type in ['minimax', 'minimaxv2', 'minimaxv3', 'minimaxv4']:
        ranges = HYPERPARAM_RANGES[args.dataset]
    else:
        ranges = DRV2_HYPERPARAM_RANGES[args.dataset]
    
    # Suggest hyperparameters based on model type
    if args.model_type == 'minimax':
        train_args = {
            'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
            'batch_size_prop': trial.suggest_categorical('batch_size', ranges['batch_size']),  # Same as batch_size
            'gamma': trial.suggest_float('gamma', *ranges['gamma']),
            'G': trial.suggest_categorical('G', ranges['G']),
            'beta': trial.suggest_categorical('beta', ranges['beta']),
            'num_bins': trial.suggest_categorical('num_bins', ranges['num_bins']),
            'alpha': 0.5,  # Unused parameter
            'theta': 1     # Unused parameter
        }
        
        model_args = {
            'embedding_k': trial.suggest_categorical('embedding_k', ranges['embedding_k']),
            'embedding_k1': trial.suggest_categorical('embedding_k1', ranges['embedding_k1']),
            'pred_lr': trial.suggest_categorical('pred_lr', ranges['pred_lr']),
            'impu_lr': trial.suggest_categorical('impu_lr', ranges['impu_lr']),
            'prop_lr': trial.suggest_categorical('prop_lr', ranges['prop_lr']),
            'dis_lr': trial.suggest_categorical('dis_lr', ranges['dis_lr']),
            'lamb_pred': trial.suggest_categorical('lamb_pred', ranges['lamb_pred']),
            'lamb_imp': trial.suggest_categorical('lamb_imp', ranges['lamb_imp']),
            'lamb_prop': trial.suggest_categorical('lamb_prop', ranges['lamb_prop']),
            'dis_lamb': trial.suggest_categorical('dis_lamb', ranges['dis_lamb']),
            'abc_model_name': trial.suggest_categorical('abc_model_name', ranges['abc_model_name']),
            'copy_model_pred': 1
        }
    elif args.model_type == 'minimaxv2':
        # MinimaxV2 with unified embedding size
        train_args = {
            'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
            'batch_size_prop': trial.suggest_categorical('batch_size', ranges['batch_size']),  # Same as batch_size
            'gamma': trial.suggest_float('gamma', *ranges['gamma']),
            'G': trial.suggest_categorical('G', ranges['G']),
            'beta': trial.suggest_categorical('beta', ranges['beta']),
            'num_bins': trial.suggest_categorical('num_bins', ranges['num_bins']),
            'alpha': 0.5,  # Unused parameter
            'theta': 1     # Unused parameter
        }
        
        # Use single embedding_k for all components
        embedding_k = trial.suggest_categorical('embedding_k', ranges['embedding_k'])
        
        model_args = {
            'embedding_k': embedding_k,
            'embedding_k1': embedding_k,  # Same as embedding_k for unified size
            'pred_lr': trial.suggest_categorical('pred_lr', ranges['pred_lr']),
            'impu_lr': trial.suggest_categorical('impu_lr', ranges['impu_lr']),
            'prop_lr': trial.suggest_categorical('prop_lr', ranges['prop_lr']),
            'dis_lr': trial.suggest_categorical('dis_lr', ranges['dis_lr']),
            'lamb_pred': trial.suggest_categorical('lamb_pred', ranges['lamb_pred']),
            'lamb_imp': trial.suggest_categorical('lamb_imp', ranges['lamb_imp']),
            'lamb_prop': trial.suggest_categorical('lamb_prop', ranges['lamb_prop']),
            'dis_lamb': trial.suggest_categorical('dis_lamb', ranges['dis_lamb']),
            'abc_model_name': trial.suggest_categorical('abc_model_name', ranges['abc_model_name']),
            'copy_model_pred': 1,
            'model_type': 'minimaxv2'  # Pass model type to train function
        }
    elif args.model_type == 'minimaxv3':
        # MinimaxV3 with enhanced architecture
        train_args = {
            'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
            'batch_size_prop': trial.suggest_categorical('batch_size', ranges['batch_size']),  # Same as batch_size
            'gamma': trial.suggest_float('gamma', *ranges['gamma']),
            'G': trial.suggest_categorical('G', ranges['G']),
            'beta': trial.suggest_categorical('beta', ranges['beta']),
            'num_bins': trial.suggest_categorical('num_bins', ranges['num_bins']),
            'alpha': 0.5,  # Unused parameter
            'theta': 1     # Unused parameter
        }
        
        model_args = {
            'embedding_k': trial.suggest_categorical('embedding_k', ranges['embedding_k']),
            'embedding_k1': trial.suggest_categorical('embedding_k1', ranges['embedding_k1']),
            'pred_lr': trial.suggest_categorical('pred_lr', ranges['pred_lr']),
            'impu_lr': trial.suggest_categorical('impu_lr', ranges['impu_lr']),
            'prop_lr': trial.suggest_categorical('prop_lr', ranges['prop_lr']),
            'dis_lr': trial.suggest_categorical('dis_lr', ranges['dis_lr']),
            'lamb_pred': trial.suggest_categorical('lamb_pred', ranges['lamb_pred']),
            'lamb_imp': trial.suggest_categorical('lamb_imp', ranges['lamb_imp']),
            'lamb_prop': trial.suggest_categorical('lamb_prop', ranges['lamb_prop']),
            'dis_lamb': trial.suggest_categorical('dis_lamb', ranges['dis_lamb']),
            'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4]),
            'grad_clip_norm': trial.suggest_categorical('grad_clip_norm', [0.5, 1.0, 2.0]),
            'abc_model_name': trial.suggest_categorical('abc_model_name', ['mlp_enhanced', 'logistic_regression']),
            'copy_model_pred': 1,
            'model_type': 'minimaxv3'  # Pass model type to train function
        }
    elif args.model_type == 'minimaxv4':
        # MinimaxV4 with standard models + V3 training improvements
        train_args = {
            'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
            'batch_size_prop': trial.suggest_categorical('batch_size', ranges['batch_size']),  # Same as batch_size
            'gamma': trial.suggest_float('gamma', *ranges['gamma']),
            'G': trial.suggest_categorical('G', ranges['G']),
            'beta': trial.suggest_categorical('beta', ranges['beta']),
            'num_bins': trial.suggest_categorical('num_bins', ranges['num_bins']),
            'alpha': 0.5,  # Unused parameter
            'theta': 1     # Unused parameter
        }
        
        model_args = {
            'embedding_k': trial.suggest_categorical('embedding_k', ranges['embedding_k']),
            'embedding_k1': trial.suggest_categorical('embedding_k1', ranges['embedding_k1']),
            'pred_lr': trial.suggest_categorical('pred_lr', ranges['pred_lr']),
            'impu_lr': trial.suggest_categorical('impu_lr', ranges['impu_lr']),
            'prop_lr': trial.suggest_categorical('prop_lr', ranges['prop_lr']),
            'dis_lr': trial.suggest_categorical('dis_lr', ranges['dis_lr']),
            'lamb_pred': trial.suggest_categorical('lamb_pred', ranges['lamb_pred']),
            'lamb_imp': trial.suggest_categorical('lamb_imp', ranges['lamb_imp']),
            'lamb_prop': trial.suggest_categorical('lamb_prop', ranges['lamb_prop']),
            'dis_lamb': trial.suggest_categorical('dis_lamb', ranges['dis_lamb']),
            'grad_clip_norm': trial.suggest_categorical('grad_clip_norm', [0.5, 1.0, 2.0]),
            'abc_model_name': trial.suggest_categorical('abc_model_name', ['logistic_regression', 'mlp']),
            'copy_model_pred': 1,
            'model_type': 'minimaxv4'  # Pass model type to train function
        }
    else:
        # DR-V2 parameters
        train_args = {
            'batch_size': trial.suggest_categorical('batch_size', ranges['batch_size']),
            'batch_size_prop': trial.suggest_categorical('batch_size', ranges['batch_size']),  # Same as batch_size
            'gamma': trial.suggest_float('gamma', *ranges['gamma']),
            'beta': trial.suggest_categorical('beta', ranges['beta']),
            'alpha': trial.suggest_categorical('alpha', ranges['alpha']),
            'num_epoch': trial.suggest_categorical('num_epoch', ranges['num_epoch'])
        }
        
        # Add imputation parameter only for drv2_imp
        if args.model_type == 'drv2_imp':
            train_args['imputation'] = trial.suggest_categorical('imputation', ranges['imputation'])
        
        model_args = {
            'embedding_k': trial.suggest_categorical('embedding_k', ranges['embedding_k']),
            'embedding_k1': trial.suggest_categorical('embedding_k1', ranges['embedding_k1']),
            'pred_lr': trial.suggest_categorical('pred_lr', ranges['pred_lr']),
            'prop_lr': trial.suggest_categorical('prop_lr', ranges['prop_lr']),
            'lamb_pred': trial.suggest_categorical('lamb_pred', ranges['lamb_pred']),
            'lamb_prop': trial.suggest_categorical('lamb_prop', ranges['lamb_prop'])
        }
        
        # Add embedding_k_prop if specified in ranges
        if 'embedding_k_prop' in ranges:
            model_args['embedding_k_prop'] = trial.suggest_categorical('embedding_k_prop', ranges['embedding_k_prop'])
        
        # Add imputation-specific parameters only for drv2_imp
        if args.model_type == 'drv2_imp':
            model_args['impu_lr'] = trial.suggest_categorical('impu_lr', ranges['impu_lr'])
            model_args['lamb_imp'] = trial.suggest_categorical('lamb_imp', ranges['lamb_imp'])
    
    try:
        start_time = time.time()
        if args.model_type in ['minimax', 'minimaxv2', 'minimaxv3', 'minimaxv4']:
            results = train_and_eval_with_params(args.dataset, train_args, model_args, seed=args.seed)
        else:
            use_imputation = (args.model_type == 'drv2_imp')
            results = train_and_eval_drv2(args.dataset, train_args, model_args, use_imputation, seed=args.seed)
        training_time = time.time() - start_time
        
        for metric_name, value in results.items():
            trial.set_user_attr(metric_name, value)
        trial.set_user_attr('training_time', training_time)
        
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
    csv_path = os.path.join(dataset_output_dir, f'{args.dataset}_{args.model_type}_all_trials.csv')
    
    row_data = {
        'trial_number': trial.number,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **trial.params,
        **results,
        'training_time': trial.user_attrs.get('training_time', -1)
    }
    
    if hasattr(trial, 'values') and trial.values is not None:
        for i, (metric, value) in enumerate(zip(args.metrics, trial.values)):
            row_data[f'objective_{metric}'] = value
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search for MF models')
    parser.add_argument('--dataset', type=str, default='coat', 
                        choices=['coat', 'yahoo', 'kuai'],
                        help='Dataset to use')
    parser.add_argument('--model_type', type=str, default='minimax',
                        choices=['minimax', 'minimaxv2', 'minimaxv3', 'minimaxv4', 'drv2', 'drv2_imp'],
                        help='Model type to use: minimax (MF_Minimax), minimaxv2 (MF_MinimaxV2 with fixed binning), minimaxv3 (MF_MinimaxV3 with enhanced architecture), minimaxv4 (MF_MinimaxV4 with standard models + V3 training), drv2 (MF_DRv2_BMSE), drv2_imp (MF_DRv2_BMSE_Imp)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optuna trials')
    parser.add_argument('--metrics', type=str, nargs='+', default=['auc'],
                        choices=['auc', 'ndcg', 'recall', 'f1', 'mse', 'mae', 'precision',
                                 'ndcg_5', 'ndcg_10', 'ndcg_20', 'ndcg_50', 'ndcg_100',
                                 'precision_5', 'precision_10', 'precision_20', 'precision_50', 'precision_100',
                                 'recall_5', 'recall_10', 'recall_20', 'recall_50', 'recall_100',
                                 'f1_5', 'f1_10', 'f1_20', 'f1_50', 'f1_100'],
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
    
    # check that metrics and directions have the same length
    if len(args.metrics) != len(args.directions):
        parser.error(f"Number of metrics ({len(args.metrics)}) must match number of directions ({len(args.directions)})")
    
    return args


def main():
    args = parse_args()
    
    set_all_seeds(args.seed)
    set_deterministic()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    if args.study_name is None:
        metrics_str = '_'.join(args.metrics)
        args.study_name = f'{args.dataset}_{args.model_type}_{metrics_str}_{time.strftime("%Y%m%d_%H%M%S")}'
    
    if args.storage:
        study = create_study(
            study_name=args.study_name,
            directions=args.directions,
            storage=args.storage,
            load_if_exists=True,
            sampler=TPESampler(seed=args.seed)
        )
    else:
        study = create_study(
            study_name=args.study_name,
            directions=args.directions,
            sampler=TPESampler(seed=args.seed)
        )
    
    print(f"Starting multi-objective hyperparameter search for {args.dataset} dataset using {args.model_type} model")
    print(f"Optimizing metrics: {', '.join([f'{m} ({d})' for m, d in zip(args.metrics, args.directions)])}")
    print(f"Running {args.n_trials} trials...")
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    pareto_trials = study.best_trials
    
    print("\n" + "="*50)
    print(f"Found {len(pareto_trials)} Pareto optimal solutions")
    
    pareto_params_path = os.path.join(dataset_output_dir, f'{args.dataset}_{args.model_type}_version2_pareto_optimal_params.csv')
    with open(pareto_params_path, 'w', newline='') as f:
        if pareto_trials:
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
                for i, metric in enumerate(args.metrics):
                    row[metric] = trial.values[i]
                row.update(trial.params)
                writer.writerow(row)
    
    summary_path = os.path.join(dataset_output_dir, f'{args.dataset}_{args.model_type}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Type: {args.model_type}\n")
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
                
                f.write("Objective values:\n")
                for i, (metric, value) in enumerate(zip(args.metrics, trial.values)):
                    f.write(f"  {metric}: {value:.6f}\n")
                
                f.write("\nAll metrics:\n")
                for key, value in trial.user_attrs.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                
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



  # For DR-V2 with imputation
  # python real_world/optuna_search.py --dataset coat --model_type drv2_imp --n_trials 100 --metrics auc ndcg

  # For DR-V2 without imputation
  # python real_world/optuna_search.py --dataset yahoo --model_type drv2 --n_trials 100 --metrics auc

  # Original Minimax model (default)
  # python real_world/optuna_search.py --dataset kuai --model_type minimax --n_trials 100