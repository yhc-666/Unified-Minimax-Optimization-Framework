# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
import pdb
import arguments
import time

from dataset import load_data
from matrix_factorization_DT import *
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))


def count_parameters(model):
    """Count trainable parameters in the model and its components"""
    total_params = 0
    param_details = {}
    
    # Count parameters for each component
    if hasattr(model, 'prediction_model'):
        pred_params = sum(p.numel() for p in model.prediction_model.parameters() if p.requires_grad)
        param_details['Prediction Model'] = pred_params
        total_params += pred_params
    
    if hasattr(model, 'imputation'):
        impu_params = sum(p.numel() for p in model.imputation.parameters() if p.requires_grad)
        param_details['Imputation Model'] = impu_params
        total_params += impu_params
    
    if hasattr(model, 'propensity_model'):
        prop_params = sum(p.numel() for p in model.propensity_model.parameters() if p.requires_grad)
        param_details['Propensity Model'] = prop_params
        total_params += prop_params
    
    return total_params, param_details


def train_and_eval(dataset_name, train_args, model_args):
    
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

    np.random.seed(2020)
    torch.manual_seed(2020)

    print("# user: {}, # item: {}".format(num_user, num_item))
    # binarize
    if dataset_name == "kuai":
        y_train = binarize(y_train, 1)
        y_test = binarize(y_test, 1)
    else:
        y_train = binarize(y_train)
        y_test = binarize(y_test)

    "DR-BIAS"
    # Start timing for model initialization
    init_start_time = time.time()
    
    mf = MF_DR_BIAS(num_user, num_item,
                    batch_size=train_args['batch_size'],
                    batch_size_prop=train_args.get('batch_size_prop', 1024),
                    embedding_k=model_args.get('embedding_k', 4))
    if torch.cuda.is_available():
        mf.cuda()
    
    init_time = time.time() - init_start_time
    
    # Count parameters
    total_params, param_details = count_parameters(mf)
    
    # Train the model (DR-BIAS does propensity training inside fit)
    train_start_time = time.time()
    mf.fit(x_train, y_train, 
        lr=model_args['lr'], 
        lamb=model_args['lamb'],
        gamma=train_args.get('gamma', 0.1),
        G=train_args.get('G', 1))
    train_time = time.time() - train_start_time
    
    total_time = init_time + train_time

    test_pred = mf.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)
    f1 = 2 / (1 / np.mean(precisions[top_k_names[0]]) + 1 / np.mean(recalls[top_k_names[1]]))

    print("***"*5 + "[DR-BIAS]" + "***"*5)
    print("[DR-BIAS] test mse:", mse_mf)
    print("[DR-BIAS] test mse:", mae_mf)
    print("[DR-BIAS] test auc:", auc)
    print("[DR-BIAS] {}:{:.6f}".format(
            top_k_names[2].replace("_", "@"), np.mean(ndcgs[top_k_names[2]])))
    print("[DR-BIAS] {}:{:.6f}".format(top_k_names[3].replace("_", "@"), f1))
    print("[DR-BIAS] {}:{:.6f}".format(
            top_k_names[0].replace("_", "@"), np.mean(precisions[top_k_names[0]])))
    print("[DR-BIAS] {}:{:.6f}".format(
            top_k_names[1].replace("_", "@"), np.mean(recalls[top_k_names[1]])))
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + "[DR-BIAS]" + "***"*5)
    
    # Print complexity analysis
    print("\n" + "="*50)
    print("[DR-BIAS] Complexity Analysis:")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    for component, params in param_details.items():
        print(f"  - {component}: {params:,}")
    
    print(f"\nTraining Time:")
    print(f"  - Model Initialization: {init_time:.2f} seconds")
    print(f"  - Training (includes propensity): {train_time:.2f} seconds")
    print(f"  - Total Time: {total_time:.2f} seconds")
    print("="*50 + "\n")

def para(args):
    if args.dataset=="coat":
        args.train_args = {
            "batch_size": 128, 
            "batch_size_prop": 1024,
            "gamma": 0.0174859545582588,  # Propensity score clipping threshold
            "G": 1                         # Ratio of unobserved to observed samples
        }
        args.model_args = {
            "embedding_k": 32,             # Embedding dimension
            "lr": 0.05,                    # Learning rate
            "lamb": 0.005                  # Weight decay
        }
    elif args.dataset=="yahoo":
        args.train_args = {
            "batch_size": 4096,            # Unified with Minimax
            "batch_size_prop": 4096,       # Unified with Minimax
            "gamma": 0.05,                 # Standard gamma for yahoo
            "G": 1                         # Standard G for yahoo
        }
        args.model_args = {
            "embedding_k": 32,             # Embedding dimension for yahoo
            "lr": 0.005,                   # Unified with Minimax pred_lr
            "lamb": 1e-4
        }
    elif args.dataset=="kuai":
        args.train_args = {
            "batch_size": 4096, 
            "batch_size_prop": 4096,
            "gamma": 0.05,                 # Standard gamma for kuai
            "G": 12                        # Standard G for kuai
        }
        args.model_args = {
            "embedding_k": 64,             # Embedding dimension for kuai
            "lr": 0.005, 
            "lamb": 1e-4
        }
    return args

if __name__ == "__main__":
    args = arguments.parse_args()
    para(args=args)

    train_and_eval(args.dataset, args.train_args, args.model_args)

# python real_world/DR-BIAS.py --dataset kuai