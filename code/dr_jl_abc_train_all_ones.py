from tensorboardX import SummaryWriter
import time
from dataset import load_data
from utils import set_seed, check_dir, generate_total_sample, ndcg_func, recall_func, precision_func
import torch
import numpy as np
import csv
from parse import parse_args
from model import dr_jl_abc_all_ones, dr_jl_abc_eqb_all_ones
import pandas as pd
from sklearn.metrics import roc_auc_score
import optuna
import os
import itertools
mse_func = lambda x,y: np.mean((x-y)**2)


def objective(trial):  
    if args.ex_idx == 1:
        args.gamma = trial.suggest_uniform('gamma', 1e-12, 0.1)
        args.G = trial.suggest_categorical('G', [1, 2, 3, 4, 5, 6, 8, 10])

        args.pred_lr = trial.suggest_loguniform('pred_lr', 0.0005, 0.1)
        args.pred_lamb = trial.suggest_loguniform('pred_lamb', 1e-7, 10.0)
        args.impu_lr = trial.suggest_loguniform('impu_lr', 0.0005, 0.1)
        args.impu_lamb = trial.suggest_loguniform('impu_lamb', 1e-7, 10.0)
        args.prop_lr = trial.suggest_loguniform('prop_lr', 0.0005, 0.1)
        args.prop_lamb = trial.suggest_loguniform('prop_lamb', 1e-7, 10.0)

        args.prop_lr2 = trial.suggest_loguniform('prop_lr2', 0.0005, 0.1)
        args.prop_lamb2 = trial.suggest_loguniform('prop_lamb2', 1e-7, 10.0)
        

        
    elif args.ex_idx == 21:        
        args.gamma = trial.suggest_uniform('gamma', 1e-12, 0.1)
        args.G = trial.suggest_categorical('G', [1, 2, 3, 4, 5, 6, 8, 10])

        args.pred_lr = trial.suggest_loguniform('pred_lr', 0.0005, 0.1)
        args.pred_lamb = trial.suggest_loguniform('pred_lamb', 1e-6, 1.0)
        args.impu_lr = trial.suggest_loguniform('impu_lr', 0.0005, 0.1)
        args.impu_lamb = trial.suggest_loguniform('impu_lamb', 1e-6, 1.0)
        args.prop_lr = trial.suggest_loguniform('prop_lr', 0.0005, 0.1)
        args.prop_lamb = trial.suggest_loguniform('prop_lamb', 1e-6, 1.0)

        args.prop_lr2 = trial.suggest_loguniform('prop_lr2', 0.0005, 0.1)
        args.prop_lamb2 = trial.suggest_loguniform('prop_lamb2', 1e-6, 1.0)

        

    if args.ex_idx == 41:        
        args.gamma = trial.suggest_uniform('gamma', 1e-12, 0.1)
        args.G = trial.suggest_categorical('G', [1, 2, 3, 4, 5, 6, 8, 10])

        args.pred_lr = trial.suggest_loguniform('pred_lr', 0.0005, 0.1)
        args.pred_lamb = trial.suggest_loguniform('pred_lamb', 1e-7, 1.0)
        args.impu_lr = trial.suggest_loguniform('impu_lr', 0.0005, 0.1)
        args.impu_lamb = trial.suggest_loguniform('impu_lamb', 1e-7, 1.0)
        args.prop_lr = trial.suggest_loguniform('prop_lr', 0.0005, 0.1)
        args.prop_lamb = trial.suggest_loguniform('prop_lamb', 1e-7, 1.0)

        args.prop_lr2 = trial.suggest_loguniform('prop_lr2', 0.0005, 0.1)
        args.prop_lamb2 = trial.suggest_loguniform('prop_lamb2', 1e-7, 1.0)

        


    
    
    hyper_param = (
        f"{args.batch_size:.6f}_"
        f"{args.batch_size_prop:.6f}_"
        f"{args.beta:.6f}_"
        f"{args.gamma:.6f}_"
        f"{args.G:.6f}_"
        f"{args.pred_lr:.6f}_"
        f"{args.pred_lamb:.6f}_"
        f"{args.impu_lr:.6f}_"
        f"{args.impu_lamb:.6f}_"
        f"{args.prop_lr:.6f}_"
        f"{args.prop_lamb:.6f}_"
        f"{args.prop_lr2:.6f}_"
        f"{args.prop_lamb2:.6f}_"
        f"{args.dis_lr:.6f}_"
        f"{args.dis_lamb:.6f}"
    )
    
    print("hyper_param", hyper_param)

    if args.is_tensorboard:
        tb_log = SummaryWriter(f"{args.tensorborad_path}{args.num_bins}{args.ex_idx}/{hyper_param}")

    else:
        tb_log = None
    
    set_seed(args.seed)
    res_list = []
    
    my_model = eval(f"{args.debias_name}(num_users, num_items, args.pred_model_name, args.prop_model_name, args.abc_model_name, aug_load_param_type, args.copy_model_pred, args.embedding_k, args.batch_size_prop, args.batch_size, args.device, args.is_tensorboard)")
    start = time.time()
    
    log_prop_epoch = my_model._compute_IPS(tb_log, x_all, obs, num_epochs=200, prop_lr=args.prop_lr, prop_lamb=args.prop_lamb)

    log_epoch = my_model.fit(tb_log, x_all, obs, x_train_tensor, y_train_tensor, grad_type=args.grad_type, num_epochs=args.num_epochs, num_bins=args.num_bins, beta=args.beta, gamma=args.gamma, G=args.G, pred_lr=args.pred_lr, impu_lr=args.impu_lr, prop_lr=args.prop_lr2, dis_lr=args.dis_lr, pred_lamb=args.pred_lamb, impu_lamb=args.impu_lamb, prop_lamb=args.prop_lamb2, dis_lamb=args.dis_lamb, tol=args.tol)

    print('time cost', time.time() - start)
    
    
    val_pred = my_model.predict(x_val_tensor).detach().cpu().numpy()
    
    log_val_auc = roc_auc_score(y_val, val_pred)
    log_val_mse= mse_func(y_val, val_pred)

    
    test_pred = my_model.predict(x_test_tensor).detach().cpu().numpy()
    log_test_auc = roc_auc_score(y_test, test_pred)
    log_test_mse = mse_func(y_test, test_pred)
    
    
    if args.data_name == 'ml100k':
        res_list = [file_name, hyper_param, log_prop_epoch, log_epoch, log_val_auc, log_test_auc]
    else:
        if 'kuai' not in args.data_name:
            log_test_ndcg = ndcg_func(my_model, x_test, y_test, device=args.device)
            log_test_recall = recall_func(my_model, x_test, y_test, device=args.device)
            res_list = [file_name, hyper_param, log_prop_epoch, log_epoch, log_val_auc, log_val_mse, log_test_auc, log_test_mse, np.mean(log_test_ndcg['ndcg_5']),np.mean(log_test_recall['recall_5'])]

    
        else:
            top_k_list = [20, 50]
            log_test_ndcg = ndcg_func(my_model, x_test, y_test, device=args.device, top_k_list=top_k_list)
            log_test_recall = recall_func(my_model, x_test, y_test, device=args.device, top_k_list=top_k_list)
            res_list = [file_name, hyper_param, log_prop_epoch, log_epoch, log_val_auc, log_val_mse, log_test_auc, log_test_mse, np.mean(log_test_ndcg['ndcg_20']), np.mean(log_test_recall['recall_20']), np.mean(log_test_ndcg['ndcg_50']), np.mean(log_test_recall['recall_50'])]



    print('res_list', res_list)
    
    with open(metric_file_path, mode='a', newline='') as tar_file:
        writer = csv.writer(tar_file)
        writer.writerow(res_list)
    

    if args.is_tensorboard:
        tb_log.close()

    if args.tune_type == 'val':
        print(f'tune on {args.tune_type}')
        return log_val_auc
    else:
        print(f'tune on {args.tune_type}')
        return log_test_auc


if __name__ == '__main__':
    args = parse_args()
    print('dr_jl_abc_train')
    print("args", args)
    if 'dr_jl_abc' in args.debias_name:
        print(f'{args.debias_name}')
    else:
        print('wrong debias method')
        exit()

    set_seed(args.seed)

    file_name = f"propensity_framework_{args.data_name}_{args.thres}{args.train_rate}{args.val_rate}_{args.debias_name}_{args.pred_model_name}_{args.impu_model_name}_{args.prop_model_name}_{args.abc_model_name}_ce_{args.load_param_type}_impu_copy{args.copy_model_pred}_{args.embedding_k}_grad_type{args.grad_type}_{args.num_epochs}_{args.tol}_{args.num_bins}_{args.beta}_{args.batch_size_prop}_{args.batch_size}_{args.sever}_{args.device}_{args.seed}_optuna_tune_on_{args.tune_type}_{args.auc_type}_auc_ex{args.ex_idx}_revision"
    args.file_name = file_name
    print("file_name", file_name)

    aug_load_param_type = file_name

    num_users, num_items, x_train, x_val, x_test, y_train, y_val, y_test = load_data(args.data_name, args.data_path, args.thres, args.train_rate, args.val_rate)
    print('positive rate', y_train.sum() / float(len(y_train)), y_test.sum() / float(len(y_test)))

    x_train_tensor = torch.from_numpy(x_train).long().to(args.device)
    y_train_tensor = torch.from_numpy(y_train).float().to(args.device)
    x_val_tensor = torch.from_numpy(x_val).long().to(args.device)
    y_val_tensor = torch.from_numpy(y_val).float().to(args.device)
    x_test_tensor = torch.from_numpy(x_test).long().to(args.device)
    y_test_tensor = torch.from_numpy(y_test).float().to(args.device)
    
    
    x_all = generate_total_sample(num_users, num_items).to(args.device)

    obs = torch.sparse.FloatTensor(torch.cat([x_train_tensor[:, 0].unsqueeze(dim=0), x_train_tensor[:, 1].unsqueeze(dim=0)], dim=0), torch.ones_like(y_train_tensor), torch.Size([num_users, num_items])).to_dense().reshape(-1)
    
    
        
    check_dir(f'../metric/{args.debias_name}/', '_')
    check_dir(f'../model_param/{args.debias_name}/', '_')
    check_dir(f'../optuna_storage/{args.debias_name}/', '_')
    print("data info")
    print("num_users", num_users)
    print("num_items", num_items)
    print('x_train_tensor', x_train_tensor, x_train_tensor.shape)
    print('y_train_tensor', y_train_tensor, y_train_tensor.shape)
    print('x_val_tensor', x_val_tensor, x_val_tensor.shape)
    print('y_val_tensor', y_val_tensor, y_val_tensor.shape)
    print('x_test_tensor', x_test_tensor, x_test_tensor.shape)
    print('y_test_tensor', y_test_tensor, y_test_tensor.shape)
    print('x_all', x_all, x_all.shape)
    print('obs', obs, obs.shape)
    print('check_x_all', obs[x_train_tensor[:, 0]*num_items+x_train_tensor[:, 1]], obs[x_train_tensor[:, 0]*num_items+x_train_tensor[:, 1]].sum())
    
    
    metric_file_path = f'../metric/{args.debias_name}/{file_name}.csv'
    if 'kuai' not in args.data_name:
        column_names = ["file_name", "hyper_param", "log_prop_epoch", "log_epoch", "val_auc", "val_mse", "auc", "mse", "ndcg_5", "recall_5"]
    else:
        column_names = ["file_name", "hyper_param", "log_prop_epoch", "log_epoch", "val_auc", "val_mse", "auc", "mse", "ndcg_20", "recall_20", "ndcg_50", "recall_50"]

    if args.data_name == 'ml100k':
        column_names = ["file_name", "hyper_param", "log_prop_epoch", "log_epoch", "val_auc", "auc"]

    with open(metric_file_path, mode='a', newline='') as tar_file:
        writer = csv.writer(tar_file)
        writer.writerow(column_names)

    if args.debias_name == 'erm':
        n_trials = 50
    else:
        n_trials=1000
 
    
    optuna_storage_path = os.path.join(f'../optuna_storage/{args.debias_name}', f'{file_name}.sqlite3')
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed), storage=f'sqlite:///{optuna_storage_path}')
    study.optimize(objective, n_trials=n_trials)

    
    print('study.trials_dataframe(), study.best_params', study.trials_dataframe(), study.best_params)
    

            



