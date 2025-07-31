#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for Minimax model with different hyperparameter values
on ECE, BMSE, DR_Bias, DR_Variance metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sps
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
import time
import pandas as pd
import os
from typing import Dict, List, Tuple

from dataset import load_data
from matrix_factorization_DT import generate_total_sample, MF_Minimax
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))

# ----------- Metric functions copied from semi-synthetic/evaluation.py ------------

def get_bins(hat_p: torch.Tensor, M: int, mode: str = 'equal_freq'):
    """Helper function for binning propensity scores."""
    device = hat_p.device
    if mode == 'equi_width':
        edges = torch.linspace(0., 1., M + 1, device=device)
        idx = torch.bucketize(hat_p, edges, right=False) - 1
    elif mode == 'equal_freq':
        q = torch.linspace(0., 1., M + 1, device=device)[1:-1]
        qv = torch.quantile(hat_p, q)
        full_edges = torch.cat([torch.tensor([-float('inf')], device=device),
                                qv,
                                torch.tensor([float('inf')], device=device)])
        idx = torch.bucketize(hat_p, full_edges, right=True) - 1
    else:
        raise ValueError(f"unknown bin mode {mode}")
    idx = torch.clamp(idx, 0, M - 1)
    one_hot = torch.nn.functional.one_hot(idx, M).float()
    return idx, one_hot


def compute_ece_torch(hat_p: torch.Tensor,
                     obs: torch.Tensor,
                     M: int = 10,
                     mode: str = 'equal_freq') -> torch.Tensor:
    """PyTorch implementation of Expected Calibration Error."""
    idx, one_hot = get_bins(hat_p, M, mode)
    n = float(len(hat_p))
    
    obs_sum_per_bin = torch.matmul(obs.unsqueeze(0), one_hot).squeeze(0)
    hat_p_sum_per_bin = torch.matmul(hat_p.unsqueeze(0), one_hot).squeeze(0)
    
    samples_per_bin = one_hot.sum(dim=0)
    
    ece = 0.0
    for m in range(M):
        if samples_per_bin[m] > 0:
            avg_obs = obs_sum_per_bin[m] / samples_per_bin[m]
            avg_hat_p = hat_p_sum_per_bin[m] / samples_per_bin[m]
            weight = samples_per_bin[m] / n
            ece += weight * torch.abs(avg_obs - avg_hat_p)
    
    return ece


def compute_bmse_torch(phi: torch.Tensor,
                      hat_p: torch.Tensor,
                      obs: torch.Tensor) -> torch.Tensor:
    """Batch-wise MSE."""
    term = (obs / hat_p - (1 - obs) / (1 - hat_p)).unsqueeze(1) * phi
    bmse = term.mean(dim=0).norm(p=2) ** 2
    return bmse


def compute_dr_bias_torch(p_true: torch.Tensor,
                         hat_p: torch.Tensor,
                         e_true: torch.Tensor,
                         e_hat: torch.Tensor) -> torch.Tensor:
    """Theoretical DR Bias calculation."""
    n = float(len(hat_p))
    bias = torch.abs((((p_true - hat_p) / hat_p) * (e_true - e_hat)).sum()) / n
    return bias


def compute_dr_variance_torch(p_true: torch.Tensor,
                             hat_p: torch.Tensor,
                             e_true: torch.Tensor,
                             e_hat: torch.Tensor) -> torch.Tensor:
    """Theoretical DR Variance calculation."""
    n2 = float(len(hat_p)) ** 2
    term1 = p_true * (1 - p_true)
    term2 = (e_true - e_hat).pow(2)
    term3 = hat_p.pow(2)
    
    var = (term1 * term2 / term3).sum() / n2
    return var


def get_phi_normalized(model, x, device='cuda'):
    """Extract model prediction features (phi) for BMSE calculation.
    
    Uses sigmoid(logits) to map to [0,1] range.
    """
    model.eval()
    with torch.no_grad():
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # Convert to tensor
        x_tensor = torch.LongTensor(x).to(device)
        user_idx = x_tensor[:, 0]
        item_idx = x_tensor[:, 1]
        
        # Get raw logits from prediction model
        U_emb = model.model_pred.W(user_idx)
        V_emb = model.model_pred.H(item_idx)
        logits = torch.sum(U_emb.mul(V_emb), 1)
        
        # Ensure logits is on correct device
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, device=device, dtype=torch.float32)
        else:
            logits = logits.to(device)
        
        if len(logits.shape) > 1:
            logits = logits.squeeze()
        
        # Use sigmoid to map logits to [0, 1]
        phi = torch.sigmoid(logits)
        
        return phi.unsqueeze(1)


def get_model_propensity_scores(model, x_test: np.ndarray, device) -> np.ndarray:
    """Extract propensity scores from trained Minimax model."""
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.LongTensor(x_test).to(device)
        prop_scores = model.model_prop(x_test_tensor)
        return prop_scores.cpu().numpy()

# ------------ End of metric functions ------------


def count_parameters(model):
    """Count trainable parameters in the model and its components"""
    total_params = 0
    param_details = {}
    
    # Count parameters for each component
    if hasattr(model, 'model_pred'):
        pred_params = sum(p.numel() for p in model.model_pred.parameters() if p.requires_grad)
        param_details['Prediction Model'] = pred_params
        total_params += pred_params
    
    if hasattr(model, 'model_impu'):
        impu_params = sum(p.numel() for p in model.model_impu.parameters() if p.requires_grad)
        param_details['Imputation Model'] = impu_params
        total_params += impu_params
    
    if hasattr(model, 'model_prop'):
        prop_params = sum(p.numel() for p in model.model_prop.parameters() if p.requires_grad)
        param_details['Propensity Model'] = prop_params
        total_params += prop_params
    
    if hasattr(model, 'model_abc'):
        abc_params = sum(p.numel() for p in model.model_abc.parameters() if p.requires_grad)
        param_details['Discriminator Model'] = abc_params
        total_params += abc_params
    
    return total_params, param_details


def train_and_eval_minimax(dataset_name, train_args, model_args, x_test, y_test):
    """Train Minimax model and evaluate with metrics - using exact logic from Minimax.py"""
    
    # Set up top_k values based on dataset
    top_k_list = [5]
    if dataset_name == "kuai":
        top_k_list = [20]
    
    # Load data
    if dataset_name == "coat":
        train_mat, test_mat = load_data("coat")        
        x_train, y_train = rating_mat_to_sample(train_mat)
        x_test, y_test = rating_mat_to_sample(test_mat)
        num_user = train_mat.shape[0]
        num_item = train_mat.shape[1]
    elif dataset_name == "yahoo":
        x_train, y_train, _, _ = load_data("yahoo")
        x_train, y_train = shuffle(x_train, y_train)
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1
    elif dataset_name == "kuai":
        x_train, y_train, _, _ = load_data("kuai")
        num_user = x_train[:,0].max() + 1
        num_item = x_train[:,1].max() + 1

    # Set random seeds
    np.random.seed(2020)
    torch.manual_seed(2020)

    print("# user: {}, # item: {}".format(num_user, num_item))
    
    # Binarize labels
    if dataset_name == "kuai":
        y_train = binarize(y_train, 2)
        y_test = binarize(y_test, 2)
    else:
        y_train = binarize(y_train, 3)
        y_test = binarize(y_test, 3)

    "Minimax"
    # Start timing for model initialization
    init_start_time = time.time()
    
    # Create model
    mf = MF_Minimax(num_user, num_item, batch_size=train_args['batch_size'], batch_size_prop=train_args['batch_size_prop'],
                    embedding_k=model_args['embedding_k'], embedding_k1=model_args['embedding_k1'],
                    abc_model_name=model_args.get('abc_model_name', 'logistic_regression'),
                    copy_model_pred=model_args.get('copy_model_pred', 1))
    
    init_time = time.time() - init_start_time
    
    # Count parameters
    total_params, param_details = count_parameters(mf)
    
    # First compute propensity scores
    prop_start_time = time.time()
    mf._compute_IPS(x_train, 
                    num_epoch=200,  # Fixed for propensity pre-training
                    lr=model_args.get('prop_lr', 0.01), 
                    lamb=model_args.get('prop_lamb', 0),
                    verbose=False)
    prop_time = time.time() - prop_start_time
    
    # Then train the full model
    train_start_time = time.time()
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
           num_bins=train_args.get('num_bins', 10))
    train_time = time.time() - train_start_time
    
    total_time = init_time + prop_time + train_time

    # Get predictions
    test_pred = mf.predict(x_test)
    
    # Compute all original metrics from Minimax.py
    mse_mf = mse_func(y_test, test_pred)
    
    # Compute AUC if both classes are present
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, test_pred)
    else:
        auc = np.nan  # AUC is undefined when only one class is present
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)
    
    # Print original metrics
    print("***"*5 + "[Minimax]" + "***"*5)
    print("[Minimax] test mse:", mse_mf)
    print("[Minimax] test mae:", mae_mf)
    print("[Minimax] test auc:", auc)
    
    # Print results for each k value
    for k in top_k_list:
        precision_key = f"precision_{k}"
        recall_key = f"recall_{k}"
        ndcg_key = f"ndcg_{k}"
        
        f1_k = 2 / (1 / np.mean(precisions[precision_key]) + 1 / np.mean(recalls[recall_key]))
        
        print("[Minimax] {}:{:.6f}".format(
                ndcg_key.replace("_", "@"), np.mean(ndcgs[ndcg_key])))
        print("[Minimax] {}:{:.6f}".format(f"f1@{k}", f1_k))
        print("[Minimax] {}:{:.6f}".format(
                precision_key.replace("_", "@"), np.mean(precisions[precision_key])))
        print("[Minimax] {}:{:.6f}".format(
                recall_key.replace("_", "@"), np.mean(recalls[recall_key])))
    
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[Minimax]" + "***"*5)
    
    # Print complexity analysis
    print("\n" + "="*50)
    print("[Minimax] Complexity Analysis:")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    for component, params in param_details.items():
        print(f"  - {component}: {params:,}")
    
    print(f"\nTraining Time:")
    print(f"  - Model Initialization: {init_time:.2f} seconds")
    print(f"  - Propensity Pre-training: {prop_time:.2f} seconds")
    print(f"  - Main Training: {train_time:.2f} seconds")
    print(f"  - Total Time: {total_time:.2f} seconds")
    print("="*50 + "\n")
    
    # Now compute evaluation-specific metrics (ECE, BMSE, DR_Bias, DR_Variance)
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mf.to(device)
    
    # Get propensity scores
    hat_p_test = get_model_propensity_scores(mf, x_test, device)
    
    # For computing true propensity, we need to generate observation indicators
    # In real-world datasets, we don't have true propensities, so we'll compute metrics
    # that don't require them (ECE, BMSE) and approximate others
    
    # Create observation indicator for test set (all 1s since these are observed)
    obs_test = np.ones(len(y_test))
    
    # Convert to torch tensors
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)
    test_pred_torch = torch.tensor(test_pred, dtype=torch.float32).to(device)
    hat_p_test_torch = torch.tensor(hat_p_test, dtype=torch.float32).to(device)
    obs_test_torch = torch.tensor(obs_test, dtype=torch.float32).to(device)
    
    # Clamp propensity scores to avoid numerical issues
    hat_p_test_torch = torch.clamp(hat_p_test_torch, 1e-6, 1-1e-6)
    
    # Compute ECE
    ece = compute_ece_torch(hat_p_test_torch, obs_test_torch, M=10, mode='equal_freq')
    
    # Compute BMSE
    phi = get_phi_normalized(mf, x_test, device)
    bmse = compute_bmse_torch(phi, hat_p_test_torch, obs_test_torch)
    
    # For DR_Bias and DR_Variance, we approximate using empirical propensities
    # Since we don't have true propensities in real data, we use the observed frequency
    # as a proxy (this is a limitation but necessary for real-world evaluation)
    
    # Estimate "true" propensities from training data frequency
    train_obs = sps.csr_matrix((np.ones(x_train.shape[0]), (x_train[:, 0], x_train[:, 1])), 
                               shape=(num_user, num_item), dtype=np.float32).toarray()
    
    # Get empirical propensities for test samples
    p_true_test = []
    for i in range(len(x_test)):
        user, item = x_test[i]
        # Use a small constant to avoid division by zero
        p_true_test.append(max(train_obs[user, item], 0.01))
    
    p_true_test_torch = torch.tensor(p_true_test, dtype=torch.float32).to(device)
    p_true_test_torch = torch.clamp(p_true_test_torch, 1e-6, 1-1e-6)
    
    # Compute DR metrics
    dr_bias = compute_dr_bias_torch(p_true_test_torch, hat_p_test_torch, 
                                    y_test_torch, test_pred_torch)
    dr_variance = compute_dr_variance_torch(p_true_test_torch, hat_p_test_torch,
                                           y_test_torch, test_pred_torch)
    
    # Prepare metrics dictionary with all original metrics plus evaluation-specific ones
    metrics = {
        'MSE': mse_mf,
        'MAE': mae_mf,
        'AUC': auc,
        'ECE': ece.item(),
        'BMSE': bmse.item(),
        'DR_Bias': dr_bias.item(),
        'DR_Variance': dr_variance.item()
    }
    
    # Add NDCG, precision, recall, F1 metrics for each k
    for k in top_k_list:
        ndcg_key = f"ndcg_{k}"
        precision_key = f"precision_{k}"
        recall_key = f"recall_{k}"
        f1_key = f"f1_{k}"
        
        metrics[f'NDCG@{k}'] = np.mean(ndcgs[ndcg_key])
        metrics[f'Precision@{k}'] = np.mean(precisions[precision_key])
        metrics[f'Recall@{k}'] = np.mean(recalls[recall_key])
        
        # Calculate F1 score
        f1_k = 2 / (1 / np.mean(precisions[precision_key]) + 1 / np.mean(recalls[recall_key]))
        metrics[f'F1@{k}'] = f1_k
    
    # Add Gini indices
    metrics['Gini_Item'] = gi
    metrics['Gini_User'] = gu
    
    # Add parameter counts
    metrics['Total_Parameters'] = total_params
    
    # Add timing metrics
    metrics['Init_Time'] = init_time
    metrics['Prop_Training_Time'] = prop_time
    metrics['Main_Training_Time'] = train_time
    metrics['Total_Time'] = total_time
    
    return metrics


def evaluate_hyperparameter(dataset_name: str, hyperparam_name: str, hyperparam_values: List[float]):
    """Evaluate Minimax model with different values of a specific hyperparameter"""
    
    # Load test data once
    if dataset_name == "coat":
        train_mat, test_mat = load_data("coat")        
        x_test, y_test = rating_mat_to_sample(test_mat)
    elif dataset_name == "yahoo":
        _, _, x_test, y_test = load_data("yahoo")
    elif dataset_name == "kuai":
        _, _, x_test, y_test = load_data("kuai")
    
    # Don't binarize here - train_and_eval_minimax will handle it
    
    # Get default hyperparameters based on dataset
    if dataset_name == "coat":
        train_args = {
            "batch_size": 128,
            "batch_size_prop": 128,
            "gamma": 0.0174859545582588,
            "G": 1,
            "alpha": 0.5,
            "beta": 0.1,
            "theta": 1,
            "num_bins": 20
        }
        model_args = {
            "embedding_k": 32,
            "embedding_k1": 64,
            "pred_lr": 0.05,
            "impu_lr": 0.01,
            "prop_lr": 0.05,
            "dis_lr": 0.01,
            "lamb_prop": 1e-3,
            "prop_lamb": 1e-3,
            "lamb_pred": 0.005,
            "lamb_imp": 0.0001,
            "dis_lamb": 0.005,
            "abc_model_name": "logistic_regression",
            "copy_model_pred": 1
        }
    elif dataset_name == "yahoo":
        train_args = {
            "batch_size": 4096,
            "batch_size_prop": 4096,
            "gamma": 0.025320297702893,
            "G": 4,
            "alpha": 0.5,
            "beta": 1,
            "theta": 1,
            "num_bins": 20
        }
        model_args = {
            "embedding_k": 32,
            "embedding_k1": 64,
            "pred_lr": 0.005,
            "impu_lr": 0.01,
            "prop_lr": 0.005,
            "dis_lr": 0.01,
            "lamb_prop": 0.00990492184668211,
            "prop_lamb": 0.00990492184668211,
            "lamb_pred": 0.00011624950138819,
            "lamb_imp": 0.039023385901065,
            "dis_lamb": 0.0437005524910195,
            "abc_model_name": "logistic_regression",
            "copy_model_pred": 1
        }
    elif dataset_name == "kuai":
        train_args = {
            "batch_size": 4096,
            "batch_size_prop": 32764,
            "gamma": 0.05,
            "G": 4,
            "alpha": 0.5,
            "beta": 1e-5,
            "theta": 1,
            "num_bins": 10
        }
        model_args = {
            "embedding_k": 16,
            "embedding_k1": 16,
            "pred_lr": 0.01,
            "impu_lr": 0.01,
            "prop_lr": 0.01,
            "dis_lr": 0.01,
            "lamb_prop": 1e-2,
            "prop_lamb": 1e-2,
            "lamb_pred": 1e-5,
            "lamb_imp": 1e-2,
            "dis_lamb": 0.0,
            "abc_model_name": "logistic_regression",
            "copy_model_pred": 1
        }
    
    results = []
    
    for value in hyperparam_values:
        print(f"\nEvaluating {hyperparam_name} = {value}")
        
        # Update the specific hyperparameter
        if hyperparam_name in train_args:
            train_args_copy = train_args.copy()
            train_args_copy[hyperparam_name] = value
            model_args_copy = model_args.copy()
        elif hyperparam_name in model_args:
            train_args_copy = train_args.copy()
            model_args_copy = model_args.copy()
            model_args_copy[hyperparam_name] = value
        else:
            raise ValueError(f"Unknown hyperparameter: {hyperparam_name}")
        
        # Train and evaluate
        metrics = train_and_eval_minimax(dataset_name, train_args_copy, model_args_copy, x_test, y_test.copy())
        
        # Add hyperparameter value to results
        result = {hyperparam_name: value}
        result.update(metrics)
        results.append(result)
        
        # Print results
        print(f"Results for {hyperparam_name}={value}:")
        for metric, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric}: {metric_value:.10f}")
            else:
                print(f"  {metric}: {metric_value}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Minimax model with different hyperparameters')
    parser.add_argument('--dataset', type=str, default='yahoo', 
                        choices=['coat', 'yahoo', 'kuai'],
                        help='Dataset to use')
    parser.add_argument('--hyperparam', type=str, required=True,
                        help='Hyperparameter to vary (e.g., beta, gamma, G, etc.)')
    parser.add_argument('--values', nargs='+', type=float, required=True,
                        help='List of values to test for the hyperparameter')
    parser.add_argument('--output', type=str, default='minimax_evaluation_results.csv',
                        help='Output CSV file for results')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"Evaluating Minimax on {args.dataset} dataset")
    print(f"Testing {args.hyperparam} with values: {args.values}")
    
    # Run evaluation
    results_df = evaluate_hyperparameter(args.dataset, args.hyperparam, args.values)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Display summary
    print("\nSummary of results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

# python real_world/evaluate_minimax.py --dataset yahoo --hyperparam num_bins --values 1 3 5 8 10 12 15 18 20 25 30 35 40 45