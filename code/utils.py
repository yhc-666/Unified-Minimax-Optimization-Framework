# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import random
# environmental setting

import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    


def generate_total_sample(num_users, num_items):
    user_indices = torch.arange(num_users)
    item_indices = torch.arange(num_items)

    grid_x, grid_y = torch.meshgrid(user_indices, item_indices, indexing='ij')

    combinations = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    
    return combinations


def check_dir(dir_path, file_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    if not os.path.exists(file_path):
        return False
    else:
        return True

def ndcg_func(model, x_te, y_te, device='cuda', top_k_list = [5, 10]):
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(torch.from_numpy(x_u).long().to(device)).detach().cpu().numpy()
        # print('x_u', x_u, x_u.shape)
        # print('pred_u', pred_u, pred_u.shape)
        
        for top_k in top_k_list:
            ori_topk = top_k
            if top_k > len(pred_u):
                top_k = len(pred_u)
            pred_top_k = np.argsort(-pred_u)[:top_k]

            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(ori_topk)].append(ndcg_k)

    return result_map

def ndcg_func_impu(model, x_te, y_te, device='cuda', top_k_list = [5, 10]):
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.model_impu.predict(torch.from_numpy(x_u).long().to(device)).detach().cpu().numpy()

        for top_k in top_k_list:
            ori_topk = top_k
            if top_k > len(pred_u):
                top_k = len(pred_u)
            pred_top_k = np.argsort(-pred_u)[:top_k]

            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

            dcg_k = y_u[pred_top_k] / log2_iplus1

            best_dcg_k = y_u[np.argsort(-y_u)][:top_k] / log2_iplus1

            if np.sum(best_dcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(best_dcg_k)

            result_map["ndcg_{}".format(ori_topk)].append(ndcg_k)

    return result_map



def recall_func(model, x_te, y_te, device='cuda', top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(torch.from_numpy(x_u).long().to(device)).detach().cpu().numpy()
        for top_k in top_k_list:
            ori_topk = top_k
            if top_k > len(pred_u):
                top_k = len(pred_u)
                
            pred_top_k = np.argsort(-pred_u)[:top_k]

            recall = np.sum(y_u[pred_top_k]) / max(1, sum(y_u))

            result_map["recall_{}".format(ori_topk)].append(recall)

    return result_map




def precision_func(model, x_te, y_te, device='cuda', top_k_list = [5, 10]):
    """Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_te[:,0])
    all_tr_idx = np.arange(len(x_te))
    result_map = defaultdict(list)

    for uid in all_user_idx:
        u_idx = all_tr_idx[x_te[:,0] == uid]
        x_u = x_te[u_idx]
        y_u = y_te[u_idx]
        pred_u = model.predict(torch.from_numpy(x_u).long().to(device)).detach().cpu().numpy()
        for top_k in top_k_list:
            ori_topk = top_k
            if top_k > len(pred_u):
                top_k = len(pred_u)
            pred_top_k = np.argsort(-pred_u)[:top_k]

            recall = np.sum(y_u[pred_top_k]) / top_k

            result_map["precision_{}".format(ori_topk)].append(recall)

    return result_map
