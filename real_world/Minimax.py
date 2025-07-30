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
from matrix_factorization_DT import generate_total_sample, MF_Minimax
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))


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
        top_k_list = [20]
        top_k_names = ("precision_20", "recall_20", "ndcg_20", "f1_20")

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

    "Minimax"
    # Start timing for model initialization
    init_start_time = time.time()
    
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
                    verbose=True)
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

    test_pred = mf.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)

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


def para(args):
    """Set hyperparameters for different datasets"""
    if args.dataset=="coat":
        args.train_args = {
            "batch_size": 128,              # Mini-batch size for training prediction/imputation models
            "batch_size_prop": 128,         # Mini-batch size for training propensity model
            "gamma": 0.0174859545582588,                  # Propensity score clipping threshold (clips to [gamma, 1.0] to avoid extreme weights)
            "G": 1,                         # Ratio of unobserved to observed samples (controls exploration in DR estimator)
            "alpha": 0.5,                   # Unused in current implementation (kept for compatibility)
            "beta": 0.1,                    # Weight for adversarial loss in propensity model training
            "theta": 1,                     # Unused in current implementation (kept for compatibility)
            "num_bins": 20                 # Number of bins for propensity score stratification
        }
        args.model_args = {
            "embedding_k": 32,               # Embedding dimension for propensity and discriminator models
            "embedding_k1": 64,              # Embedding dimension for prediction and imputation models
            "pred_lr": 0.05,               # Learning rate for prediction model
            "impu_lr": 0.01,               # Learning rate for imputation model
            "prop_lr": 0.05,               # Learning rate for propensity model during main training
            "dis_lr": 0.01,                # Learning rate for discriminator model
            "lamb_prop": 1e-3,              # Weight decay for propensity model during main training
            "prop_lamb": 1e-3,              # Weight decay for propensity model during pre-training (_compute_IPS)
            "lamb_pred": 0.005,              # Weight decay for prediction model
            "lamb_imp": 0.0001,               # Weight decay for imputation model
            "dis_lamb": 0.005,                # Weight decay for discriminator model
            "abc_model_name": "logistic_regression",  # Architecture for adversarial discriminator ("logistic_regression" or "mlp")
            "copy_model_pred": 1            # Whether to initialize imputation model with prediction model weights (1=yes, 0=no)
        }
    elif args.dataset=="yahoo":
        args.train_args = {
            "batch_size": 4096,             # Larger batch size for larger dataset
            "batch_size_prop": 4096,        # Batch size for propensity model
            "gamma": 0.025320297702893,                  # Same propensity clipping as coat
            "G": 4,                         # Same exploration ratio as coat
            "alpha": 0.5,                   # Unused parameter
            "beta": 1,                   # Much smaller adversarial weight (yahoo needs less regularization)
            "theta": 1,                     # Unused parameter
            "num_bins": 3                  # Same binning strategy
        }
        args.model_args = {
            "embedding_k": 32,              # Larger embeddings for larger dataset
            "embedding_k1": 64,             # Same size for prediction/imputation embeddings
            "pred_lr": 0.005,                # Learning rate for prediction model
            "impu_lr": 0.01,                # Learning rate for imputation model
            "prop_lr": 0.005,                # Learning rate for propensity model
            "dis_lr": 0.01,                 # Learning rate for discriminator model
            "lamb_prop": 0.00990492184668211,              # Weight decay for propensity model
            "prop_lamb": 0.00990492184668211,              # Weight decay for pre-training
            "lamb_pred": 0.00011624950138819,              # Weight decay for prediction model
            "lamb_imp": 0.039023385901065,               # Weight decay for imputation model (prevents overfitting)
            "dis_lamb": 0.0437005524910195,                # Weight decay for discriminator model
            "abc_model_name": "logistic_regression",  # Same discriminator architecture
            "copy_model_pred": 1            # Initialize imputation from prediction
        }
    elif args.dataset=="kuai":
        args.train_args = {
            "batch_size": 4096,             # Large batch for efficient training
            "batch_size_prop": 32764,        # Same as main batch size
            "gamma": 0.05,                  # Standard propensity clipping
            "G": 4,                         # Standard exploration ratio
            "alpha": 0.5,                   # Unused parameter
            "beta": 1e-5,                   # Small adversarial weight like yahoo
            "theta": 1,                     # Unused parameter
            "num_bins": 10                  # Standard binning
        }
        args.model_args = {
            "embedding_k": 16,              # Larger embeddings for complex interactions
            "embedding_k1": 16,             # Same for all embedding models
            "pred_lr": 0.01,                # Learning rate for prediction model
            "impu_lr": 0.01,                # Learning rate for imputation model
            "prop_lr": 0.01,                # Learning rate for propensity model
            "dis_lr": 0.01,                 # Learning rate for discriminator model
            "lamb_prop": 1e-2,              # Weight decay for propensity model (kuai needs more)
            "prop_lamb": 1e-2,              # Weight decay for pre-training
            "lamb_pred": 1e-5,              # Weight decay for prediction model
            "lamb_imp": 1e-2,               # Weight decay for imputation model
            "dis_lamb": 0.0,                # Weight decay for discriminator model
            "abc_model_name": "logistic_regression",  # Standard discriminator
            "copy_model_pred": 1            # Initialize imputation from prediction
        }
    return args


if __name__ == "__main__":
    args = arguments.parse_args()
    para(args=args)
    
    train_and_eval(args.dataset, args.train_args, args.model_args)


# python real_world/Minimax.py --dataset yahoo