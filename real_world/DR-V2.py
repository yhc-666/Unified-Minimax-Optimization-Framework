# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.sparse as sps
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import arguments
from tqdm import tqdm
import time

from dataset import load_data
from matrix_factorization_DT import generate_total_sample, MF_DRv2_BMSE_Imp, MF_DRv2_BMSE
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
    
    if hasattr(model, 'imputation_model'):
        impu_params = sum(p.numel() for p in model.imputation_model.parameters() if p.requires_grad)
        param_details['Imputation Model'] = impu_params
        total_params += impu_params
    
    if hasattr(model, 'propensity_model'):
        prop_params = sum(p.numel() for p in model.propensity_model.parameters() if p.requires_grad)
        param_details['Propensity Model'] = prop_params
        total_params += prop_params
    
    return total_params, param_details


def train_and_eval(dataset_name, train_args, model_args, use_imputation=True):
    
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

    np.random.seed(2020)
    torch.manual_seed(2020)

    print("# user: {}, # item: {}".format(num_user, num_item))
    # binarize
    if dataset_name == "kuai":
        y_train = binarize(y_train, 2)
        y_test = binarize(y_test, 2)
    else:
        y_train = binarize(y_train, 3)
        y_test = binarize(y_test, 3)

    # Create model based on imputation flag
    init_start_time = time.time()
    
    if use_imputation:
        print("Using MF-DRv2-BMSE with imputation model")
        mf = MF_DRv2_BMSE_Imp(
            num_user, num_item, 
            batch_size=train_args['batch_size'], 
            batch_size_prop=train_args['batch_size_prop'],
            embedding_k=model_args['embedding_k'], 
            embedding_k1=model_args['embedding_k1'],
            embedding_k_prop=model_args.get('embedding_k_prop', model_args['embedding_k']))
    else:
        print("Using MF-DRv2-BMSE without imputation model")
        mf = MF_DRv2_BMSE(
            num_user, num_item,
            batch_size=train_args['batch_size'],
            batch_size_prop=train_args['batch_size_prop'],
            embedding_k=model_args['embedding_k'],
            embedding_k1=model_args['embedding_k1'])
    
    init_time = time.time() - init_start_time
    
    # Count parameters
    total_params, param_details = count_parameters(mf)
    
    # No separate propensity pre-training for DRv2 models
    
    # Train the model
    train_start_time = time.time()
    
    if use_imputation:
        mf.fit(x_train, y_train,
               lr=model_args['pred_lr'],
               impu_lr=model_args['impu_lr'],
               prop_lr=model_args['prop_lr'],
               lamb=model_args['lamb_pred'],
               lamb_imp=model_args['lamb_imp'],
               lamb_prop=model_args['lamb_prop'],
               alpha=train_args['alpha'],
               beta=train_args['beta'],
               gamma=train_args['gamma'],
               imputation=train_args.get('imputation', 1e-3),
               num_epoch=train_args.get('num_epoch', 500))
    else:
        mf.fit(x_train, y_train,
               lr=model_args['pred_lr'],
               prop_lr=model_args['prop_lr'],
               lamb=model_args['lamb_pred'],
               lamb_prop=model_args['lamb_prop'],
               alpha=train_args['alpha'],
               beta=train_args['beta'],
               gamma=train_args['gamma'],
               num_epoch=train_args.get('num_epoch', 500))
    
    train_time = time.time() - train_start_time
    total_time = init_time + train_time

    test_pred = mf.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)

    model_name = "[MF-DRv2-BMSE-Imp]" if use_imputation else "[MF-DRv2-BMSE]"
    print("***"*5 + model_name + "***"*5)
    print(f"{model_name} test mse:", mse_mf)
    print(f"{model_name} test mae:", mae_mf)
    print(f"{model_name} test auc:", auc)
    
    # Print results for each k value
    for k in top_k_list:
        precision_key = f"precision_{k}"
        recall_key = f"recall_{k}"
        ndcg_key = f"ndcg_{k}"
        
        f1_k = 2 / (1 / np.mean(precisions[precision_key]) + 1 / np.mean(recalls[recall_key]))
        
        print("{} {}:{:.6f}".format(
                model_name, ndcg_key.replace("_", "@"), np.mean(ndcgs[ndcg_key])))
        print("{} f1@{}:{:.6f}".format(model_name, k, f1_k))
        print("{} {}:{:.6f}".format(
                model_name, precision_key.replace("_", "@"), np.mean(precisions[precision_key])))
        print("{} {}:{:.6f}".format(
                model_name, recall_key.replace("_", "@"), np.mean(recalls[recall_key])))
    
    user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
    gi,gu = gini_index(user_wise_ctr)
    print("***"*5 + model_name + "***"*5)
    
    # Print complexity analysis
    print("\n" + "="*50)
    print(f"{model_name} Complexity Analysis:")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    for component, params in param_details.items():
        print(f"  - {component}: {params:,}")
    
    print(f"\nTraining Time:")
    print(f"  - Model Initialization: {init_time:.2f} seconds")
    print(f"  - Training: {train_time:.2f} seconds")
    print(f"  - Total Time: {total_time:.2f} seconds")
    print("="*50 + "\n")


def para(args, use_imputation=True):
    """Set hyperparameters for different datasets"""
    if args.dataset=="coat":
        args.train_args = {
            "batch_size": 128,              # Mini-batch size for training (aligned with parameter table)
            "batch_size_prop": 128,         # Mini-batch size for propensity
            "alpha": 1,                     # Weight for ctcvr_loss
            "beta": 0.5,                    # Weight for cvr_loss_mnar (from parameter table)
            "gamma": 0.0174859545582588,    # Weight for bmse_loss (exact value from parameter table)
            "imputation": 1e-3,             # Weight for imputation loss (if using imputation)
            "num_epoch": 500                # Number of training epochs
        }
        args.model_args = {
            "embedding_k": 32,              # Embedding dimension for propensity model (from parameter table)
            "embedding_k1": 64,             # Embedding dimension for prediction/imputation models (from parameter table)
            "embedding_k_prop": 32,         # Propensity model embedding
            "pred_lr": 0.05,                # Learning rate for prediction model (from parameter table)
            "impu_lr": 0.01,                # Learning rate for imputation model (from parameter table)
            "prop_lr": 0.05,                # Learning rate for propensity model (from parameter table)
            "lamb_pred": 0.005,             # Weight decay for prediction model (from parameter table)
            "lamb_imp": 0.0001,             # Weight decay for imputation model (from parameter table)
            "lamb_prop": 0.001              # Weight decay for propensity model (from parameter table)
        }
    elif args.dataset=="yahoo":
        args.train_args = {
            "batch_size": 4096,             # Larger batch size for larger dataset
            "batch_size_prop": 4096,        
            "alpha": 1,                     # Weight for ctcvr_loss
            "beta": 1,                      # Weight for cvr_loss_mnar
            "gamma": 0.05,                  # Weight for bmse_loss (standard value)
            "imputation": 5,                # Larger imputation weight for yahoo
            "num_epoch": 500                # Number of training epochs
        }
        args.model_args = {
            "embedding_k": 32,              # Embedding dimension for propensity model
            "embedding_k1": 64,             # Unified with Minimax
            "embedding_k_prop": 32,         # Propensity model embedding
            "pred_lr": 0.005,               # Unified with Minimax
            "impu_lr": 0.01,                # Unified with Minimax
            "prop_lr": 0.005,               # Unified with Minimax
            "lamb_pred": 1e-3,              # Weight decay for prediction model
            "lamb_imp": 1e-4,               # Weight decay for imputation model
            "lamb_prop": 1e-4               # Weight decay for propensity model
        }
    elif args.dataset=="kuai":
        args.train_args = {
            "batch_size": 4096,             # Large batch size for kuai
            "batch_size_prop": 4096,        
            "alpha": 1,                     # Weight for ctcvr_loss
            "beta": 5,                      # Higher beta for kuai
            "gamma": 0.05,                  # Weight for bmse_loss (standard value)
            "imputation": 1e-3,             # Standard imputation weight (if using imputation)
            "num_epoch": 500                # Number of training epochs
        }
        args.model_args = {
            "embedding_k": 64,              # Embedding dimension for propensity model
            "embedding_k1": 64,             # Embedding dimension for prediction/imputation models
            "embedding_k_prop": 64,         # Propensity model embedding
            "pred_lr": 0.01,                # Learning rate for prediction model
            "impu_lr": 0.01,                # Learning rate for imputation model (if using imputation)
            "prop_lr": 0.01,                # Learning rate for propensity model
            "lamb_pred": 1e-3,              # Weight decay for prediction model
            "lamb_imp": 1e-4,               # Weight decay for imputation model (if using imputation)
            "lamb_prop": 5e-4               # Weight decay for propensity model
        }
    
    # Remove imputation-related parameters if not using imputation
    if not use_imputation:
        # Remove imputation weight from train_args
        if 'imputation' in args.train_args:
            del args.train_args['imputation']
        # Remove imputation learning rate and weight decay from model_args
        if 'impu_lr' in args.model_args:
            del args.model_args['impu_lr']
        if 'lamb_imp' in args.model_args:
            del args.model_args['lamb_imp']
    
    return args


if __name__ == "__main__":
    args = arguments.parse_args()
    
    # Check if we should use imputation model based on command line argument
    # You can add a command line argument for this, or just set it here
    use_imputation = True  # Set to False to use the version without imputation
    
    para(args=args, use_imputation=use_imputation)
    
    train_and_eval(args.dataset, args.train_args, args.model_args, use_imputation)


# python real_world/DR-V2.py --dataset coat
# python real_world/DR-V2.py --dataset yahoo  
# python real_world/DR-V2.py --dataset kuai