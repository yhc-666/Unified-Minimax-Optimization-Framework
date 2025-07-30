# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.sparse as sps
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import arguments
from tqdm import tqdm

from dataset import load_data
from matrix_factorization_DT import generate_total_sample, MF_DRv2_BMSE_Imp, MF_DRv2_BMSE
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))


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
    
    # No separate propensity pre-training for DRv2 models
    
    # Train the model
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


def para(args, use_imputation=True):
    """Set hyperparameters for different datasets"""
    if args.dataset=="coat":
        args.train_args = {
            "batch_size": 512,              # Mini-batch size for training
            "batch_size_prop": 512,         # Mini-batch size for propensity
            "alpha": 1,                     # Weight for ctcvr_loss
            "beta": 2,                      # Weight for cvr_loss_mnar  
            "gamma": 0.1,                   # Weight for bmse_loss
            "imputation": 1e-3,             # Weight for imputation loss (if using imputation)
            "num_epoch": 500                # Number of training epochs
        }
        args.model_args = {
            "embedding_k": 256,             # Embedding dimension for propensity model
            "embedding_k1": 256,            # Embedding dimension for prediction/imputation models
            "embedding_k_prop": 256,        # Propensity model embedding (if different)
            "pred_lr": 5e-4,                # Learning rate for prediction model
            "impu_lr": 5e-4,                # Learning rate for imputation model
            "prop_lr": 5e-4,                # Learning rate for propensity model
            "lamb_pred": 1e-5,              # Weight decay for prediction model
            "lamb_imp": 1e-6,               # Weight decay for imputation model
            "lamb_prop": 1                  # Weight decay for propensity model
        }
    elif args.dataset=="yahoo":
        args.train_args = {
            "batch_size": 4096,             # Larger batch size for larger dataset
            "batch_size_prop": 4096,        
            "alpha": 1,                     
            "beta": 1,                      # Different weight for yahoo
            "gamma": 5e-2,                  # Smaller bmse weight
            "imputation": 5,                # Larger imputation weight
            "num_epoch": 500
        }
        args.model_args = {
            "embedding_k": 256,
            "embedding_k1": 256,
            "embedding_k_prop": 256,
            "pred_lr": 1e-3,                # Higher learning rates for yahoo
            "impu_lr": 1e-3,
            "prop_lr": 1e-3,
            "lamb_pred": 1e-6,              
            "lamb_imp": 1e-6,
            "lamb_prop": 1e-4               
        }
    elif args.dataset=="kuai":
        args.train_args = {
            "batch_size": 4096,
            "batch_size_prop": 4096,
            "alpha": 1,
            "beta": 5,                      # Higher beta for kuai
            "gamma": 5e-2,
            "num_epoch": 500                # No imputation weight for non-imputation version
        }
        args.model_args = {
            "embedding_k": 256,
            "embedding_k1": 256,
            "pred_lr": 5e-3,                # Learning rates between coat and yahoo
            "prop_lr": 1e-2,                
            "lamb_pred": 1e-6,
            "lamb_prop": 5e-4
        }
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