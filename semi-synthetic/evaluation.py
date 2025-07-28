#!/usr/bin/env python
"""
Evaluation script for MF_DR_JL, MF_MRDR_JL, and MF_Minimax models
using semi-synthetic dataset with calibration metrics.
"""

import argparse
import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models from real_world
from real_world.matrix_factorization_DT import MF_DR_JL, MF_MRDR_JL, MF_Minimax, MF_DR_BIAS, MF_DR_BMSE, MF_DR_DCE

# ----------- util functions for 4 error metrics ------------
# 1. ECE
# 2. BMSE
# 3. DR Bias
# 4. DR Variance

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
    
    Uses sigmoid(logits) to map to [0,1] range, consistent with MF_DR_BMSE implementation.
    """
    model.eval()
    with torch.no_grad():
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        # Convert to tensor
        x_tensor = torch.LongTensor(x).to(device)
        user_idx = x_tensor[:, 0]
        item_idx = x_tensor[:, 1]
        
        # Get raw logits (pre-sigmoid predictions) based on model type
        if hasattr(model, 'prediction_model'):
            # For MF_DR_JL, MF_MRDR_JL - they use MF_BaseModel
            U_emb = model.prediction_model.W(user_idx)
            V_emb = model.prediction_model.H(item_idx)
            logits = torch.sum(U_emb.mul(V_emb), 1)
            
        elif hasattr(model, 'model_pred'):
            # For MF_Minimax - uses MF class
            U_emb = model.model_pred.W(user_idx)
            V_emb = model.model_pred.H(item_idx)
            logits = torch.sum(U_emb.mul(V_emb), 1)
            
        else:
            # For other models, try to get embeddings directly
            if hasattr(model, 'W') and hasattr(model, 'H'):
                U_emb = model.W(user_idx)
                V_emb = model.H(item_idx)
                logits = torch.sum(U_emb.mul(V_emb), 1)
            else:
                # Fallback: use forward and compute inverse sigmoid
                output = model.forward(x)
                if not torch.is_tensor(output):
                    output = torch.tensor(output, device=device, dtype=torch.float32)
                else:
                    output = output.to(device)
                # Compute logits from sigmoid output
                # logit = log(p/(1-p))
                output = torch.clamp(output, 1e-7, 1-1e-7)  # Avoid log(0)
                logits = torch.log(output / (1 - output))
        
        # Ensure logits is on correct device
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, device=device, dtype=torch.float32)
        else:
            logits = logits.to(device)
        
        if len(logits.shape) > 1:
            logits = logits.squeeze()
        
        # Use sigmoid to map logits to [0, 1] - consistent with MF_DR_BMSE
        phi = torch.sigmoid(logits)
        
        return phi.unsqueeze(1)

# ------------ Util functions ends ------------

def load_data(data_dir: str = "data", verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load ground truth data and compute propensity scores."""
    # Load ground truth
    # Check if we're in the semi-synthetic directory or parent directory
    if os.path.exists("data/synthetic_data"):
        synthetic_path = "data/synthetic_data"
        predicted_path = "data/predicted_matrix"
    else:
        synthetic_path = "semi-synthetic/data/synthetic_data"
        predicted_path = "semi-synthetic/data/predicted_matrix"
    
    with open(synthetic_path, "rb") as f:
        ground_truth = pickle.load(f)
    
    # Load dimensions
    with open(predicted_path, "rb") as f:
        _ = pickle.load(f)  # predictions (not used)
        num_users = pickle.load(f)
        num_items = pickle.load(f)
    
    # Calculate propensity scores with correct formula
    propensity = np.copy(ground_truth)
    p = 0.5
    propensity[np.where(propensity == 0.9)] = p ** 1  # 0.5
    propensity[np.where(propensity == 0.7)] = p ** 2  # 0.25
    propensity[np.where(propensity == 0.5)] = p ** 3  # 0.125
    propensity[np.where(propensity == 0.3)] = p ** 4  # 0.0625
    propensity[np.where(propensity == 0.1)] = p ** 4  # 0.0625
    
    if verbose:
        print(f"Loaded data: {num_users} users, {num_items} items")
        print(f"Ground truth shape: {ground_truth.shape}")
        print(f"Ground truth unique values: {np.unique(ground_truth)}")
        print(f"Propensity unique values: {np.unique(propensity)}")
    
    return ground_truth, propensity, num_users, num_items


def generate_all_pairs(num_users: int, num_items: int) -> np.ndarray:
    """Generate all (user, item) pairs."""
    pairs = []
    for user in range(num_users):
        for item in range(num_items):
            pairs.append([user, item])
    return np.array(pairs)


def create_train_test_split(ground_truth: np.ndarray, propensity: np.ndarray, 
                           num_users: int, num_items: int,
                           test_ratio: float = 0.2,
                           unbiased_test_ratio: float = 0.2,
                           random_state: int = 42,
                           verbose: bool = True) -> Dict:
    """Create train/test splits with biased training and unbiased test sampling."""
    
    # Generate all pairs
    x_all = generate_all_pairs(num_users, num_items)
    y_all = ground_truth.flatten()
    p_all = propensity.flatten()
    
    # Split into train/test
    indices = np.arange(len(x_all))
    train_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=random_state)
    
    # Training data
    x_train_all = x_all[train_idx]
    y_train_all = y_all[train_idx]
    p_train_all = p_all[train_idx]
    
    # Simulate biased observations for training
    obs_mask = np.random.binomial(1, p_train_all).astype(bool)
    x_train_obs = x_train_all[obs_mask]
    y_train_obs = y_train_all[obs_mask]
    p_train_obs = p_train_all[obs_mask]
    
    # Binarize training labels
    y_train_binary = np.random.binomial(1, y_train_obs).astype(np.float32)
    
    # Test data - sample 10% uniformly for unbiased evaluation
    test_sample_size = int(len(test_idx) * unbiased_test_ratio)
    test_sample_idx = np.random.choice(test_idx, size=test_sample_size, replace=False)
    
    x_test = x_all[test_sample_idx]
    y_test = y_all[test_sample_idx]  
    p_test = p_all[test_sample_idx]
    
    y_test_binary = np.random.binomial(1, y_test).astype(np.float32)
    
    if verbose:
        print(f"\nData splits:")
        print(f"Train (observed): {len(x_train_obs)} samples ({len(x_train_obs)/len(x_train_all):.2%} of train)")
        print(f"Train positive rate: {y_train_binary.mean():.4f}")
        print(f"Test (unbiased): {len(x_test)} samples ({unbiased_test_ratio:.1%} of test)")
        print(f"Test positive rate: {y_test_binary.mean():.4f}")
    
    return {
        'x_train': x_train_obs,
        'y_train': y_train_binary,
        'x_test': x_test,
        'y_test': y_test,  # Original values (0.1-0.9)
        'y_test_binary': y_test_binary,  # Binary labels
        'p_test': p_test,  # True propensities
        'num_users': num_users,
        'num_items': num_items
    }


def get_model_propensity_scores(model, model_name: str, x_test: np.ndarray, device) -> np.ndarray:
    """Extract propensity scores from trained model."""
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.LongTensor(x_test).to(device)
        
        if model_name in ['MF_DR_JL', 'MF_MRDR_JL', 'MF_DR_BIAS', 'MF_DR_BMSE', 'MF_DR_DCE']:
            # These models use NCF_BaseModel as propensity model
            prop_scores = model.propensity_model.forward(x_test_tensor)
        elif model_name == 'MF_Minimax':
            # MF_Minimax uses logistic_regression as propensity model
            prop_scores = model.model_prop(x_test_tensor)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return prop_scores.cpu().numpy()


def train_and_evaluate_model(model_name: str, data_splits: Dict, args) -> Tuple[nn.Module, np.ndarray, Dict]:
    """Train a model and return it with predictions and metrics."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    num_users = data_splits['num_users']
    num_items = data_splits['num_items']
    
    if model_name == 'MF_DR_JL':
        model = MF_DR_JL(
            num_users=num_users, 
            num_items=num_items,
            batch_size=args.batch_size,
            batch_size_prop=args.batch_size * 2,  
            embedding_k=args.embedding_k
        )
    elif model_name == 'MF_MRDR_JL':
        model = MF_MRDR_JL(
            num_users=num_users,
            num_items=num_items,
            batch_size=args.batch_size,
            batch_size_prop=args.batch_size * 2,
            embedding_k=args.embedding_k
        )
    elif model_name == 'MF_Minimax':
        model = MF_Minimax(
            num_users=num_users,
            num_items=num_items,
            batch_size=args.batch_size,
            batch_size_prop=args.batch_size * 2,
            embedding_k=args.embedding_k,
            embedding_k1=args.embedding_k,  # For prediction/imputation models
            abc_model_name='logistic_regression'
        )
    elif model_name == 'MF_DR_BIAS':
        model = MF_DR_BIAS(
            num_users=num_users,
            num_items=num_items,
            batch_size=args.batch_size,
            batch_size_prop=args.batch_size * 2,
            embedding_k=args.embedding_k
        )
    elif model_name == 'MF_DR_BMSE':
        model = MF_DR_BMSE(
            num_users=num_users,
            num_items=num_items,
            batch_size=args.batch_size,
            batch_size_prop=args.batch_size * 2,
            embedding_k=args.embedding_k,
            bmse_weight=args.bmse_weight
        )
    elif model_name == 'MF_DR_DCE':
        model = MF_DR_DCE(
            num_users=num_users,
            num_items=num_items,
            batch_size=args.batch_size,
            batch_size_prop=args.batch_size * 2,
            embedding_k=args.embedding_k
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.verbose:
        print(f"Using device: {device}")
    
    x_train = data_splits['x_train']
    y_train = data_splits['y_train']
    
    # 1. compute propensity scores
    # For MF_DR_DCE, propensity scores are computed inside fit()
    if model_name not in ['MF_DR_DCE']:
        print("Computing propensity scores...")
        model._compute_IPS(x_train, 
                          num_epoch=args.prop_epochs, 
                          lr=args.prop_lr,
                          verbose=args.verbose)
    
    # 2. train prediction model
    print(f"\nTraining prediction model...")
    if model_name == 'MF_Minimax':
        # MF_Minimax has different fit signature
        model.fit(x_train, y_train,
                 G=args.G,  # Ratio of unobserved to observed samples
                 beta=args.beta,  # Weight for adversarial loss
                 gamma=args.gamma,  # Propensity clipping
                 num_epoch=args.epochs,
                 pred_lr=args.lr,
                 impu_lr=args.lr,
                 prop_lr=args.prop_lr,
                 dis_lr=args.lr,
                 verbose=args.verbose)
    else:
        # MF_DR_JL and MF_MRDR_JL
        if model_name == 'MF_DR_JL':
            # MF_DR_JL doesn't use prior_y
            model.fit(x_train, y_train,
                     gamma=args.gamma,
                     num_epoch=args.epochs,
                     lr=args.lr,
                     G=args.G,
                     verbose=args.verbose)
        elif model_name == 'MF_MRDR_JL':
            # MF_MRDR_JL doesn't use prior_y
            model.fit(x_train, y_train,
                     gamma=args.gamma,
                     num_epoch=args.epochs,
                     lr=args.lr,
                     G=args.G,
                     verbose=args.verbose)
        else:
            # MF_DR_BIAS, MF_DR_BMSE, and MF_DR_DCE
            if model_name == 'MF_DR_DCE':
                # MF_DR_DCE needs additional ECE parameters
                model.fit(x_train, y_train,
                         gamma=args.gamma,
                         num_epoch=args.epochs,
                         lr=args.lr,
                         G=args.G,
                         verbose=args.verbose,
                         ece_weight=args.ece_weight,
                         n_bins=args.n_bins,
                         prop_epochs=args.prop_epochs,
                         prop_lr=args.prop_lr)
            else:
                model.fit(x_train, y_train,
                         gamma=args.gamma,
                         num_epoch=args.epochs,
                         lr=args.lr,
                         G=args.G,
                         verbose=args.verbose)
    
    x_test = data_splits['x_test']
    y_test = data_splits['y_test']
    y_test_binary = data_splits['y_test_binary']
    p_test = data_splits['p_test']
    
    y_pred = model.predict(x_test)
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()
    
    # Get propensity scores from model
    hat_p_test = get_model_propensity_scores(model, model_name, x_test, device)
    
    # Convert to torch tensors for metric calculation
    y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)
    y_test_binary_torch = torch.tensor(y_test_binary, dtype=torch.float32).to(device)
    y_pred_torch = torch.tensor(y_pred, dtype=torch.float32).to(device)
    p_test_torch = torch.tensor(p_test, dtype=torch.float32).to(device)
    hat_p_test_torch = torch.tensor(hat_p_test, dtype=torch.float32).to(device)
    
    p_test_torch = torch.clamp(p_test_torch, 1e-6, 1-1e-6)
    hat_p_test_torch = torch.clamp(hat_p_test_torch, 1e-6, 1-1e-6)
    
    # 1. ECE with equal-frequency binning
    ece = compute_ece_torch(hat_p_test_torch, y_test_binary_torch, M=10, mode='equal_freq')
    
    # 2. BMSE
    phi = get_phi_normalized(model, x_test, device)
    if phi is not None:
        bmse = compute_bmse_torch(phi, hat_p_test_torch, y_test_binary_torch)
    else:
        bmse = torch.tensor(float('nan'))
    
    # 3. DR Bias
    dr_bias = compute_dr_bias_torch(p_test_torch, hat_p_test_torch, y_test_torch, y_pred_torch)

    # 4. DR Variance
    dr_variance = compute_dr_variance_torch(p_test_torch, hat_p_test_torch, y_test_torch, y_pred_torch)
    
    # Additional metrics
    mse = torch.mean((y_pred_torch - y_test_torch)**2).item()
    mae = torch.mean(torch.abs(y_pred_torch - y_test_torch)).item()
    
    metrics = {
        'model': model_name,
        'ECE': ece.item(),
        'BMSE': bmse.item() if not torch.isnan(bmse) else float('nan'),
        'DR_Bias': dr_bias.item(),
        'DR_Variance': dr_variance.item(),
        'MSE': mse,
        'MAE': mae
    }
    
    print(f"\nResults for {model_name}:")
    print(f"ECE: {metrics['ECE']:.6f}")
    print(f"BMSE: {metrics['BMSE']:.6f}" if not np.isnan(metrics['BMSE']) else "BMSE: N/A")
    print(f"DR Bias: {metrics['DR_Bias']:.6f}")
    print(f"DR Variance: {metrics['DR_Variance']:.9f}")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    
    return model, y_pred, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MF models with calibration metrics on semi-synthetic data')
    
    parser.add_argument('--models', nargs='+', 
                       default=['MF_DR_BMSE', 'MF_DR_JL', 'MF_MRDR_JL', 'MF_Minimax', 'MF_DR_BIAS', 'MF_DR_DCE'],
                       choices=['MF_DR_JL', 'MF_MRDR_JL', 'MF_Minimax', 'MF_DR_BIAS', 'MF_DR_BMSE', 'MF_DR_DCE'],
                       help='Models to evaluate')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--prop_epochs', type=int, default=100,
                       help='Number of epochs for propensity model')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for prediction model')
    parser.add_argument('--prop_lr', type=float, default=0.01,
                       help='Learning rate for propensity model')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--embedding_k', type=int, default=8,
                       help='Embedding dimension')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Propensity score clipping threshold')
    parser.add_argument('--G', type=int, default=2,
                       help='Ratio of unobserved to observed samples')
    parser.add_argument('--beta', type=float, default=1,
                       help='Weight for adversarial loss (MF_Minimax only)')
    parser.add_argument('--bmse_weight', type=float, default=1,
                       help='Weight for BMSE loss (MF_DR_BMSE only)')
    parser.add_argument('--ece_weight', type=float, default=10,
                       help='Weight for ECE loss (MF_DR_DCE only)')
    parser.add_argument('--n_bins', type=int, default=10,
                       help='Number of bins for ECE calculation (MF_DR_DCE only)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test set ratio')
    parser.add_argument('--unbiased_test_ratio', type=float, default=0.2,
                       help='Ratio of test set to use as unbiased sample')
    parser.add_argument('--save_results', type=str, default='evaluation_results.csv',
                       help='Path to save results CSV')
    parser.add_argument('--verbose', default=True,
                       help='Show training progress')
    parser.add_argument('--seed', type=int, default=2024,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("Evaluation Configuration:")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}, Prop epochs: {args.prop_epochs}")
    print(f"Learning rates: pred={args.lr}, prop={args.prop_lr}")
    print(f"Batch size: {args.batch_size}, Embedding dim: {args.embedding_k}")
    print(f"Gamma: {args.gamma}, G: {args.G}")
    
    # Load data
    print("\nLoading data...")
    ground_truth, propensity, num_users, num_items = load_data(verbose=args.verbose)
    
    # Create train/test splits
    print("\nCreating train/test splits...")
    data_splits = create_train_test_split(
        ground_truth, propensity, num_users, num_items,
        test_ratio=args.test_ratio,
        unbiased_test_ratio=args.unbiased_test_ratio,
        random_state=args.seed,
        verbose=args.verbose
    )
    
    # Train and evaluate models
    all_results = []
    for model_name in args.models:
        model, predictions, metrics = train_and_evaluate_model(model_name, data_splits, args)
        
        # Add hyperparameters to results
        metrics.update({
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'embedding_k': args.embedding_k,
            'gamma': args.gamma,
            'seed': args.seed
        })
        
        all_results.append(metrics)
        
        # Save results incrementally after each model
        current_result_df = pd.DataFrame([metrics])
        if os.path.exists(args.save_results):
            # Append without header if file exists
            current_result_df.to_csv(args.save_results, mode='a', header=False, index=False)
        else:
            # Create new file with header
            current_result_df.to_csv(args.save_results, index=False)
        print(f"Results for {model_name} saved to {args.save_results}")
    
    # Display summary of all results
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY OF ALL RESULTS")
        print("="*80)
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()