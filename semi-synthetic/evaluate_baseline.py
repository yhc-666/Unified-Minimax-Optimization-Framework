#!/usr/bin/env python
"""
Usage:
    python evaluate_baseline.py --model MF_DR [options]
    
Available models:
    MF_DR, MF_MRDR_JL, MF_DR_BIAS, MF_DR_V2, dr_jl_abc
"""

import argparse
import sys
import os
import pickle
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import baseline methods
from baselines import MF_DR, MF_MRDR_JL, MF_DR_BIAS, MF_DR_V2, dr_jl_abc



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
    """Extract and normalize model prediction logits for BMSE calculation."""
    model.eval()
    with torch.no_grad():
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # Get raw logits (pre-sigmoid predictions)
        if hasattr(model, 'prediction_model'):
            # For models with separate prediction model (MF_MRDR_JL, MF_DR_BIAS, MF_DR_V2)
            logits = model.prediction_model.forward(x)
        elif hasattr(model, 'model_pred'):
            # For dr_jl_abc model
            logits = model.model_pred.forward_logit(x)
        else:
            # For direct models (MF_DR)
            logits = model.forward(x)

        # debug 奇怪的问题
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, device=device, dtype=torch.float32)
        else:
            logits = logits.to(device)
        
        if len(logits.shape) > 1:
            logits = logits.squeeze()
        
        # Normalize to [0, 1]
        min_val = logits.min()
        max_val = logits.max()
        
        # Handle edge case where all values are the same
        if max_val - min_val < 1e-8:
            # Return 0.5 for all values if there's no variation
            phi = torch.full_like(logits, 0.5)
        else:
            # Normalize to [0, 1]
            phi = (logits - min_val) / (max_val - min_val)
        
        return phi.unsqueeze(1)



def load_data(verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load ground truth data and dimensions, compute propensity scores."""
    with open("semi-synthetic/data/synthetic_data", "rb") as f:
        ground_truth = pickle.load(f)
    
    with open("semi-synthetic/data/predicted_matrix", "rb") as f:
        _ = pickle.load(f)  # predictions
        num_users = pickle.load(f)
        num_items = pickle.load(f)
    
    # Calculate propensity scores exactly as in synthetic.py
    propensity = np.copy(ground_truth)
    p = 0.5
    propensity[np.where(propensity == 1)] = p ** 1  
    propensity[np.where(propensity == 1)] = p ** 1  
    propensity[np.where(propensity == 1)] = p ** 1  
    propensity[np.where(propensity == 0)] = p ** 4  
    propensity[np.where(propensity == 0)] = p ** 4  
    # Actual result: positive samples get 0.5, negative samples get 0.0625
    
    if verbose:
        print(f"Loaded data: {num_users} users, {num_items} items")
        print(f"Ground truth shape: {ground_truth.shape}")
        print(f"Positive rate: {ground_truth.mean():.4f}")
        print(f"Propensity values: positive samples = {p**1:.4f}, negative samples = {p**4:.4f}")
    
    return ground_truth, propensity, num_users, num_items


def convert_to_pairs(ground_truth: np.ndarray, propensity: np.ndarray, 
                    num_users: int, num_items: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 1D ground truth and propensity to (user, item) pairs."""
    x_all = []
    for user in range(num_users):
        for item in range(num_items):
            x_all.append([user, item])
    
    x_all = np.array(x_all)
    y_all = ground_truth.flatten()
    p_all = propensity.flatten()
    
    return x_all, y_all, p_all


def generate_biased_observations(x_all: np.ndarray, y_all: np.ndarray, p_all: np.ndarray,
                                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate biased observed data using true propensity scores."""
    n_total = len(x_all)
    
    # Use the true propensity scores for sampling
    obs_mask = np.random.binomial(1, p_all).astype(bool)
    obs_idx = np.where(obs_mask)[0]
    
    x_obs = x_all[obs_idx]
    y_obs = y_all[obs_idx]
    p_obs = p_all[obs_idx]
    
    if verbose:
        print(f"\nGenerated {len(x_obs)} biased observations ({len(x_obs)/n_total:.2%})")
        print(f"Positive rate in observations: {y_obs.mean():.4f}")
        print(f"Average propensity in observations: {p_obs.mean():.4f}")
    
    return x_obs, y_obs, p_obs, obs_idx


def get_propensity_scores(model, model_name: str, x_test: np.ndarray, y_test: np.ndarray,
                         x_train: np.ndarray, y_train: np.ndarray, y_ips: np.ndarray,
                         num_users: int, num_items: int, device) -> np.ndarray:
    """Get propensity scores based on model type."""
    if model_name == 'MF_DR_V2':
        model.eval()
        with torch.no_grad():
            x_test_tensor = torch.LongTensor(x_test).to(device)
            prop_scores = model.propensity_model.forward(x_test_tensor)
            return prop_scores.cpu().numpy()
    elif model_name == 'dr_jl_abc':
        model.eval()
        with torch.no_grad():
            x_test_tensor = torch.LongTensor(x_test).to(device)
            prop_scores = model.model_prop(x_test_tensor)
            return prop_scores.cpu().numpy()
    else:
        # Other models use statistical estimation based on IPS method (no training)
        py1 = y_ips.sum() / len(y_ips) 
        py0 = 1 - py1  # True negative rate
        po1 = len(x_train) / (num_users * num_items)  
        py1o1 = y_train.sum() / len(y_train)  
        py0o1 = 1 - py1o1 
        
        propensity_est = np.zeros(len(y_test))
        propensity_est[y_test == 0] = (py0o1 * po1) / py0
        propensity_est[y_test == 1] = (py1o1 * po1) / py1
        
        return propensity_est


def create_data_splits(x_all: np.ndarray, y_all: np.ndarray, p_all: np.ndarray,
                      x_obs: np.ndarray, y_obs: np.ndarray, p_obs: np.ndarray,
                      obs_idx: np.ndarray,
                      test_ratio: float = 0.2,
                      y_ips_ratio: float = 0.05,
                      verbose: bool = True) -> Dict:
    """Create train/test splits and sample y_ips.
    
    Note: Current approach splits observed data into train/test,
    then samples y_ips from remaining data (excluding test).
    This differs from original framework which loads pre-split data.
    """
    x_train, x_test, y_train, y_test, p_train, p_test, idx_train, idx_test = train_test_split(
        x_obs, y_obs, p_obs, obs_idx, test_size=test_ratio, random_state=42, stratify=y_obs
    )
    
    # Sample y_ips (excluding test set)
    all_idx = np.arange(len(x_all))
    available_idx = np.setdiff1d(all_idx, idx_test)
    
    n_ips = int(len(x_all) * y_ips_ratio)
    ips_idx = np.random.choice(available_idx, size=n_ips, replace=False)
    
    x_ips = x_all[ips_idx]
    y_ips = y_all[ips_idx]
    
    if verbose:
        print(f"\nData splits:")
        print(f"Train: {len(x_train)} samples, positive rate: {y_train.mean():.4f}")
        print(f"Test: {len(x_test)} samples, positive rate: {y_test.mean():.4f}")
        print(f"Y_ips (unbiased): {len(x_ips)} samples, positive rate: {y_ips.mean():.4f}")
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'p_train': p_train,
        'x_test': x_test,
        'y_test': y_test,
        'p_test': p_test,
        'x_ips': x_ips,
        'y_ips': y_ips,
        'num_users': len(np.unique(x_all[:, 0])),
        'num_items': len(np.unique(x_all[:, 1]))
    }


def train_and_evaluate(model_name: str, data_splits: Dict, args, p_all_test=None, y_all_test=None) -> Dict[str, float]:
    """Train and evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    num_users = data_splits['num_users']
    num_items = data_splits['num_items']
    
    if model_name == 'MF_DR':
        model = MF_DR(num_users=num_users, num_items=num_items, 
                     embedding_k=args.embedding_k)
    elif model_name == 'MF_MRDR_JL':
        model = MF_MRDR_JL(num_users=num_users, num_items=num_items,
                          embedding_k=args.embedding_k)
    elif model_name == 'MF_DR_BIAS':
        model = MF_DR_BIAS(num_users=num_users, num_items=num_items,
                          embedding_k=args.embedding_k)
    elif model_name == 'MF_DR_V2':
        model = MF_DR_V2(num_users=num_users, num_items=num_items,
                        batch_size=args.batch_size, embedding_k=args.embedding_k)
    elif model_name == 'dr_jl_abc':
        model = dr_jl_abc(num_users=num_users, num_items=num_items,
                         embedding_k=args.embedding_k, batch_size=args.batch_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")
    model.to(device)
    
    x_train = data_splits['x_train']
    y_train = data_splits['y_train']
    x_test = data_splits['x_test']
    y_test = data_splits['y_test']
    p_test = data_splits['p_test']
    y_ips = data_splits['y_ips']
    
    if args.verbose:
        print(f"Training with {args.epochs} epochs, lr={args.lr}")
    
    if model_name == 'MF_DR_V2':
        # MF_DR_V2 不用 y_ips 
        model.fit(x_train, y_train,
                 num_epoch=args.epochs, lr=args.lr, verbose=args.verbose)
    else:
        model.fit(x_train, y_train, y_ips=y_ips,
                 num_epoch=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, verbose=args.verbose)
    
    y_pred = model.predict(x_test)
    if torch.is_tensor(y_pred):
        y_pred = torch.sigmoid(y_pred).cpu().numpy()
    else:
        y_pred = torch.sigmoid(torch.tensor(y_pred)).numpy()
    
    x_train = data_splits['x_train']
    y_train = data_splits['y_train'] 
    y_ips = data_splits['y_ips']
    num_users = data_splits['num_users']
    num_items = data_splits['num_items']
    
    hat_p_test = get_propensity_scores(model, model_name, x_test, y_test,
                                       x_train, y_train, y_ips, 
                                       num_users, num_items, device)
    
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_pred_torch = torch.tensor(y_pred, dtype=torch.float32).to(device)
    p_test_torch = torch.tensor(p_test, dtype=torch.float32).to(device)
    hat_p_test_torch = torch.tensor(hat_p_test, dtype=torch.float32).to(device)
    
    # Clip propensity scores for numerical stability
    p_test_torch = torch.clamp(p_test_torch, 1e-6, 1-1e-6)
    hat_p_test_torch = torch.clamp(hat_p_test_torch, 1e-6, 1-1e-6)
    
    ece = compute_ece_torch(hat_p_test_torch, y_test_torch, M=10, mode='equi_width')
    
    phi = get_phi_normalized(model, x_test, device)
    if phi is not None:
        bmse = compute_bmse_torch(phi, hat_p_test_torch, y_test_torch)
    else:
        bmse = torch.tensor(float('nan'))
    
    p_true_torch = torch.tensor(p_all_test, dtype=torch.float32).to(device)
    p_true_torch = torch.clamp(p_true_torch, 1e-6, 1-1e-6)
    y_true_torch = torch.tensor(y_all_test, dtype=torch.float32).to(device)
    
    # e_true: actual ground truth values
    # e_hat: model predictions
    # p_true: true propensity, hat_p: estimated propensity
    dr_bias = compute_dr_bias_torch(p_true_torch, hat_p_test_torch, y_true_torch, y_pred_torch)
    dr_variance = compute_dr_variance_torch(p_true_torch, hat_p_test_torch, y_true_torch, y_pred_torch)

    results = {
        'model': model_name,
        'ECE': ece.item(),
        'BMSE': bmse.item() if not torch.isnan(bmse) else float('nan'),
        'DR_Bias': dr_bias.item(),
        'DR_Variance': dr_variance.item()
    }
    
    print(f"\nResults:")
    print(f"ECE: {results['ECE']:.4f}")
    print(f"BMSE: {results['BMSE']:.4f}" if not np.isnan(results['BMSE']) else "BMSE: N/A")
    print(f"DR Bias: {results['DR_Bias']:.4f}")
    print(f"DR Variance: {results['DR_Variance']:.9f}")
    
    return results
    




def main():
    parser = argparse.ArgumentParser(
        description='Evaluate debiasing baseline models on semi-synthetic data')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['MF_DR', 'MF_MRDR_JL', 'MF_DR_BIAS', 'MF_DR_V2', 'dr_jl_abc'],
                       help='Model to evaluate')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--embedding_k', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to results.csv file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show training progress')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Evaluating {args.model} model")
    print(f"Configuration: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}, embedding_k={args.embedding_k}")
    
    # Load and prepare data
    print("\nLoading data...")
    ground_truth, propensity, num_users, num_items = load_data(verbose=args.verbose)
    
    # Convert to pairs
    x_all, y_all, p_all = convert_to_pairs(ground_truth, propensity, num_users, num_items)
    
    # Generate biased observations
    x_obs, y_obs, p_obs, obs_idx = generate_biased_observations(
        x_all, y_all, p_all, verbose=args.verbose)
    
    # Create data splits
    data_splits = create_data_splits(
        x_all, y_all, p_all, x_obs, y_obs, p_obs, obs_idx, verbose=args.verbose)
    
    # Extract true propensity scores and ground truth labels for test samples
    # We need to map test samples back to their original indices in p_all and y_all
    x_test = data_splits['x_test']
    p_all_test = []
    y_all_test = []
    for u, i in x_test:
        idx = u * num_items + i
        p_all_test.append(p_all[idx])
        y_all_test.append(y_all[idx])
    p_all_test = np.array(p_all_test)
    y_all_test = np.array(y_all_test)
    
    # Train and evaluate
    results = train_and_evaluate(args.model, data_splits, args, 
                                p_all_test=p_all_test, y_all_test=y_all_test)
    
    # Save results if requested
    if args.save_results:
        df = pd.DataFrame([results])
        csv_filename = 'results.csv'
        
        # Check if file exists
        import os
        if os.path.exists(csv_filename):
            # Append without header
            df.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            # Create new file with header
            df.to_csv(csv_filename, index=False)
        
        print(f"\nResults saved to {csv_filename}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)