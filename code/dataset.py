# -*- coding: utf-8 -*-

import numpy as np
import os
import math
from os.path import join

def load_data(data_name, data_path, thres, train_rate, val_rate):
    x_train = np.loadtxt(f'{data_path}/{data_name}/x_train_{thres}_{int(val_rate*100)}.txt', dtype=int)
    y_train = np.loadtxt(f'{data_path}/{data_name}/y_train_{thres}_{int(val_rate*100)}.txt', dtype=int)
    x_val = np.loadtxt(f'{data_path}/{data_name}/x_val_{thres}_{int(val_rate*100)}.txt', dtype=int)
    y_val = np.loadtxt(f'{data_path}/{data_name}/y_val_{thres}_{int(val_rate*100)}.txt', dtype=int)
    x_test = np.loadtxt(f'{data_path}/{data_name}/x_test_{thres}_{int(val_rate*100)}.txt', dtype=int)
    y_test = np.loadtxt(f'{data_path}/{data_name}/y_test_{thres}_{int(val_rate*100)}.txt', dtype=int)

    num_users = x_train[:, 0].max() + 1
    num_items = x_train[:, 1].max() + 1
    

    return num_users, num_items, x_train, x_val, x_test, y_train, y_val, y_test



def load_data_cal(data_name, data_path, thres, val_rate):
    x_train = np.loadtxt(f'{data_path}/{data_name}/x_train_{thres}_{int(val_rate*100)}.txt', dtype=int)
    y_train = np.loadtxt(f'{data_path}/{data_name}/y_train_{thres}_{int(val_rate*100)}.txt', dtype=int)
    x_val = np.loadtxt(f'{data_path}/{data_name}/x_val_{thres}_{int(val_rate*100)}.txt', dtype=int)
    y_val = np.loadtxt(f'{data_path}/{data_name}/y_val_{thres}_{int(val_rate*100)}.txt', dtype=int)
    x_test = np.loadtxt(f'{data_path}/{data_name}/x_test_{thres}_{int(val_rate*100)}.txt', dtype=int)
    y_test = np.loadtxt(f'{data_path}/{data_name}/y_test_{thres}_{int(val_rate*100)}.txt', dtype=int)
    
    num_users = x_train[:, 0].max() + 1
    num_items = x_train[:, 1].max() + 1

    
    _, x_train_cal, y_train_cal, _, x_val_cal, y_val_cal = trainval_split(x_train, y_train, 0)
    
    
    return num_users, num_items, x_train_cal, x_val_cal, x_val, x_test, y_train_cal, y_val_cal, y_val, y_test


def trainval_split(x_train, y_train, start_user):
    u = start_user
    val_idx = []
    val_dic = {}
    train_dic = {}
    idxs = []
    for idx, row in enumerate(x_train):    
        if row[0] == u:
            idxs.append(idx)
        else:
            num_tv = len(idxs)
            num_v = math.ceil(num_tv * 0.1)
            num_t = num_tv - num_v
            idx_t = idxs[:num_t]
            idx_v = idxs[num_t:]

            val_idx += idx_v
            val_dic[row[0]-1] = idx_v
            train_dic[row[0]-1] = idx_t  
            
            u = row[0]
            idxs = [row[1]]
            
    num_tv = len(idxs)
    num_v = math.ceil(num_tv * 0.1)
    num_t = num_tv - num_v
    idx_t = idxs[:num_t]
    idx_v = idxs[num_t:]

    val_idx += idx_v
    val_dic[row[0]-1] = idx_v
    train_dic[row[0]-1] = idx_t  
    
    u = row[0]
    idxs = [row[1]]
    
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    train_idx = list(set(np.arange(len(x_train))) - set(val_idx))

    x_train = x_train[train_idx]
    y_train = y_train[train_idx]


    return train_dic, x_train, y_train, val_dic, x_val, y_val