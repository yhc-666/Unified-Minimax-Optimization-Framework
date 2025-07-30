# -*- coding: utf-8 -*-
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import arguments

from dataset import load_data
from matrix_factorization_DT import MF_DR_JL_CE
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))


def train_and_eval(dataset_name, train_args, model_args):
    
    top_k_list = [5, 10]
    top_k_names = ("precision_5", "recall_5", "ndcg_5", "f1_5", "precision_10", "recall_10", "ndcg_10", "f1_10")
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
        top_k_names = ("precision_50", "recall_50", "ndcg_50", "f1_50", "precision_100", "recall_100", "ndcg_100", "f1_100")

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

    # Split training data to create validation set
    n_train = int(len(x_train) * 0.8)
    x_val = x_train[n_train:]
    y_val = y_train[n_train:]
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]

    "MF-DR-JL-CE"
    mf = MF_DR_JL_CE(num_user, num_item, 
                     batch_size=train_args['batch_size'], 
                     batch_size_prop=train_args['batch_size_prop'],
                     num_experts=model_args['num_experts'],
                     embedding_k=model_args['embedding_k'])
    
    # First compute propensity scores
    print("Stage 1: Computing propensity scores...")
    mf._compute_IPS(x_train, 
                    num_epoch=200,
                    lr=model_args['lr_prop'], 
                    lamb=model_args['lamb_prop'],
                    verbose=True)
    
    # Then calibrate propensity scores
    if train_args.get('calibrate_prop', True):
        print("Stage 2: Calibrating propensity scores...")
        mf._calibrate_IPS_G(x_val, x_test,
                            num_epoch=model_args.get('cal_epochs', 100),
                            lr=model_args.get('lr_cal', 0.01),
                            lamb=model_args.get('lamb_cal', 0),
                            end_T=train_args.get('end_T', 1e-3),
                            verbose=True,
                            G=train_args.get('G_cal', 10))
    
    # Finally train the prediction and imputation models
    print("Stage 3: Training prediction and imputation models...")
    mf.fit(x_train, y_train, x_val, y_val,
           stop=train_args.get('stop', 5),
           num_epoch=train_args.get('num_epoch', 1000),
           lr=model_args['lr_pred'],
           lamb=model_args['lamb_pred'],
           gamma=train_args['gamma'],
           tol=train_args.get('tol', 1e-4),
           G=train_args['G'],
           end_T=train_args.get('end_T', 1e-3),
           lr_imp=model_args['lr_imp'],
           lamb_imp=model_args['lamb_imp'],
           lr_impcal=model_args.get('lr_impcal', 0.05),
           lamb_impcal=model_args.get('lamb_impcal', 0),
           iter_impcal=train_args.get('iter_impcal', 10),
           verbose=True,
           cal=train_args.get('cal_imp', True))

    test_pred = mf.predict(x_test)
    mse_mf = mse_func(y_test, test_pred)
    auc = roc_auc_score(y_test, test_pred)
    ndcgs = ndcg_func(mf, x_test, y_test, top_k_list)
    mae_mf = mae_func(y_test, test_pred)
    precisions = precision_func(mf, x_test, y_test, top_k_list)
    recalls = recall_func(mf, x_test, y_test, top_k_list)

    print("***"*5 + "[MF-DR-JL-CE]" + "***"*5)
    print("[MF-DR-JL-CE] test mse:", mse_mf)
    print("[MF-DR-JL-CE] test mae:", mae_mf)
    print("[MF-DR-JL-CE] test auc:", auc)
    
    # Print results for each k value
    for k in top_k_list:
        precision_key = f"precision_{k}"
        recall_key = f"recall_{k}"
        ndcg_key = f"ndcg_{k}"
        
        f1_k = 2 / (1 / np.mean(precisions[precision_key]) + 1 / np.mean(recalls[recall_key]))
        
        print("[MF-DR-JL-CE] {}:{:.6f}".format(
                ndcg_key.replace("_", "@"), np.mean(ndcgs[ndcg_key])))
        print("[MF-DR-JL-CE] {}:{:.6f}".format(f"f1@{k}", f1_k))
        print("[MF-DR-JL-CE] {}:{:.6f}".format(
                precision_key.replace("_", "@"), np.mean(precisions[precision_key])))
        print("[MF-DR-JL-CE] {}:{:.6f}".format(
                recall_key.replace("_", "@"), np.mean(recalls[recall_key])))
    
    user_wise_ctr = get_user_wise_ctr(x_test, y_test, test_pred)
    gi, gu = gini_index(user_wise_ctr)
    print("***"*5 + "[MF-DR-JL-CE]" + "***"*5)


def para(args):
    """Set hyperparameters for different datasets"""
    if args.dataset == "coat":
        args.train_args = {
            "batch_size": 128,              # Mini-batch size for training
            "batch_size_prop": 1024,        # Mini-batch size for propensity model
            "gamma": 0.05,                  # Propensity score clipping threshold
            "G": 1,                         # Ratio of unobserved to observed samples
            "G_cal": 10,                    # G for calibration stage
            "end_T": 1e-3,                  # Temperature annealing end value
            "num_epoch": 1000,              # Max training epochs
            "stop": 5,                      # Early stopping patience
            "tol": 1e-4,                    # Early stopping tolerance
            "calibrate_prop": True,         # Whether to calibrate propensity scores
            "cal_imp": True,                # Whether to calibrate imputation model
            "iter_impcal": 10               # Iterations for imputation calibration per epoch
        }
        args.model_args = {
            "embedding_k": 8,               # Embedding dimension
            "num_experts": 5,               # Number of calibration experts
            "lr_pred": 0.01,                # Learning rate for prediction model
            "lr_imp": 0.01,                 # Learning rate for imputation model
            "lr_prop": 0.01,                # Learning rate for propensity model
            "lr_cal": 0.01,                 # Learning rate for calibration
            "lr_impcal": 0.05,              # Learning rate for imputation calibration
            "lamb_prop": 1e-1,              # Weight decay for propensity model
            "lamb_pred": 1e-3,              # Weight decay for prediction model
            "lamb_imp": 1e-3,               # Weight decay for imputation model
            "lamb_cal": 0,                  # Weight decay for calibration
            "lamb_impcal": 0,               # Weight decay for imputation calibration
            "cal_epochs": 100               # Epochs for calibration
        }
    elif args.dataset == "yahoo":
        args.train_args = {
            "batch_size": 4096,             # Larger batch size for larger dataset
            "batch_size_prop": 32764,       # Batch size for propensity model
            "gamma": 0.05,                  # Same propensity clipping
            "G": 1,                         # Same exploration ratio
            "G_cal": 10,                    # G for calibration
            "end_T": 1e-3,                  # Same temperature annealing
            "num_epoch": 1000,              # Max epochs
            "stop": 5,                      # Early stopping patience
            "tol": 1e-4,                    # Early stopping tolerance
            "calibrate_prop": True,         # Calibrate propensity
            "cal_imp": True,                # Calibrate imputation
            "iter_impcal": 10               # Imputation calibration iterations
        }
        args.model_args = {
            "embedding_k": 32,              # Larger embeddings for larger dataset
            "num_experts": 10,              # More experts for larger dataset
            "lr_pred": 0.01,                # Learning rate for prediction
            "lr_imp": 0.01,                 # Learning rate for imputation
            "lr_prop": 0.01,                # Learning rate for propensity
            "lr_cal": 0.01,                 # Learning rate for calibration
            "lr_impcal": 0.05,              # Learning rate for imputation calibration
            "lamb_prop": 1e-4,              # Weight decay for propensity
            "lamb_pred": 1e-3,              # Weight decay for prediction
            "lamb_imp": 1e-3,               # Weight decay for imputation
            "lamb_cal": 0,                  # Weight decay for calibration
            "lamb_impcal": 0,               # Weight decay for imputation calibration
            "cal_epochs": 100               # Calibration epochs
        }
    elif args.dataset == "kuai":
        args.train_args = {
            "batch_size": 4096,             # Large batch for efficient training
            "batch_size_prop": 32764,       # Same as main batch size
            "gamma": 0.05,                  # Standard propensity clipping
            "G": 1,                         # Standard exploration ratio
            "G_cal": 10,                    # G for calibration
            "end_T": 1e-3,                  # Temperature annealing
            "num_epoch": 1000,              # Max epochs
            "stop": 5,                      # Early stopping patience
            "tol": 1e-4,                    # Early stopping tolerance
            "calibrate_prop": True,         # Calibrate propensity
            "cal_imp": True,                # Calibrate imputation
            "iter_impcal": 10               # Imputation calibration iterations
        }
        args.model_args = {
            "embedding_k": 64,              # Larger embeddings for complex interactions
            "num_experts": 10,              # More experts for complex data
            "lr_pred": 0.05,                # Higher learning rate
            "lr_imp": 0.05,                 # Higher learning rate
            "lr_prop": 0.01,                # Standard learning rate for propensity
            "lr_cal": 0.01,                 # Learning rate for calibration
            "lr_impcal": 0.05,              # Learning rate for imputation calibration
            "lamb_prop": 1e-4,              # Weight decay for propensity
            "lamb_pred": 1e-3,              # Weight decay for prediction
            "lamb_imp": 1e-3,               # Weight decay for imputation
            "lamb_cal": 0,                  # Weight decay for calibration
            "lamb_impcal": 0,               # Weight decay for imputation calibration
            "cal_epochs": 100               # Calibration epochs
        }
    return args


if __name__ == "__main__":
    args = arguments.parse_args()
    para(args=args)
    
    train_and_eval(args.dataset, args.train_args, args.model_args)


# python real_world/DCE-DR.py --dataset coat