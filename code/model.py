# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.sparse as sps
from copy import deepcopy
# torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)


def equal_frequency_binning(data, quantiles, n_bins=4):
    """
    对输入数据执行等频分桶
    
    参数:
        data (torch.Tensor): 输入的一维数据
        n_bins (int): 桶的数量
    
    返回:
        bin_indices (torch.Tensor): 每个元素所属的桶索引（0到n_bins-1）
        bin_boundaries (torch.Tensor): 分桶的边界值（长度为n_bins+1）
    """
    
    # 计算分位点对应的边界值
    boundaries = torch.quantile(data, quantiles).to(data.device)
    
    # 构造完整边界（包括最小值和最大值）
    full_boundaries = torch.cat([
        torch.tensor([float('-inf')]).to(data.device), 
        boundaries, 
        torch.tensor([float('inf')]).to(data.device)
    ])
    
    # 分配桶索引（0 ~ n_bins-1）
    bin_indices = torch.bucketize(data, full_boundaries, right=True) - 1
    
    # 返回桶索引和边界值
    return bin_indices, full_boundaries[1:-1]  # 去除无穷大边界



def manual_binary_cross_entropy(pred, target, weight=None, reduction='mean'):
    epsilon = 1e-12
    # pred = torch.clamp(pred, epsilon, 1. - epsilon)
    loss = - (target * torch.log(pred + epsilon) + (1. - target) * torch.log(1. - pred + epsilon))
    
    if weight is not None:
        loss = loss * weight

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class mf(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        print('mf initialized')


    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        out = torch.sigmoid((user_emb * item_emb).sum(dim=1))

        return out  
    
    
    def forward_logit(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        out = (user_emb * item_emb).sum(dim=1)

        return out           


    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 
    
    
    def predict_logit(self, x):
        with torch.no_grad():
            pred = self.forward_logit(x)
            
            return pred  




class logistic_regression(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias=True)

        print('logistic_regression initialized')


    def forward(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        z_emb = torch.cat([user_emb, item_emb], axis=1)

        out = torch.sigmoid(self.linear_1(z_emb))

        return torch.squeeze(out)        
    
        
    def forward_logit(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])

        z_emb = torch.cat([user_emb, item_emb], axis=1)

        out = self.linear_1(z_emb)

        return torch.squeeze(out)      
    
    
    def get_emb(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])
        
        return user_emb, item_emb
        
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            
            return pred 
    
    
    def predict_logit(self, x):
        with torch.no_grad():
            pred = self.forward_logit(x)
            
            return pred  


class mlp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k, bias=True)
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=True)

        self.non_linear = torch.nn.ReLU()
        
        print('mlp initialized')
        

    def forward(self, user_emb, item_emb): 
        z_emb = torch.cat([user_emb, item_emb], axis=1)
 
        h1 = self.linear_1(z_emb)
        h1 = self.non_linear(h1)            
        
        out = self.linear_2(h1)
        
        return torch.squeeze(out)      
        
        
    def predict(self, user_emb, item_emb):
        with torch.no_grad():
            pred = self.forward(user_emb, item_emb)
            
            return pred 


        
        

class dr_jl_abc(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard

        # 预测打分模型
        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   


        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)

        print('dr_jl_abc initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")
        for name, param in self.model_abc.named_parameters():
            print(f"model_abc name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled)
                
                
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1  # [N*V]
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                
                prop_error_dis = self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()) * (obs_sampled - prop_sampled.detach())
                bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                

                optimizer_dis.zero_grad()
                prop_abc_loss_dis.backward()
                optimizer_dis.step()
                
                
                prop_error_prop = self.model_abc.predict(prop_user_emb.detach(), prop_item_emb.detach()).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        


class dr_jl_abc_eqb(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   


        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)

        print('dr_jl_abc_eqb initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")
        for name, param in self.model_abc.named_parameters():
            print(f"model_abc name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled)
                
                
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  
                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                
                prop_error_dis = self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()) * (obs_sampled - prop_sampled.detach())
                bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                

                optimizer_dis.zero_grad()
                prop_abc_loss_dis.backward()
                optimizer_dis.step()
                
                
                prop_error_prop = self.model_abc.predict(prop_user_emb.detach(), prop_item_emb.detach()).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        




class dr_jl_abc_ver2(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   


        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)

        print('dr_jl_abc_ver2 initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")
        for name, param in self.model_abc.named_parameters():
            print(f"model_abc name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled)
                print('prop_sampled', prop_sampled, prop_sampled.shape)
                
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1  # [N*V]
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  
                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                bin_unique_values, bin_counts = torch.unique(bin_indices, return_counts=True)
                print('bin_indices', bin_indices, bin_indices.shape)
                print('bin_unique_values', bin_unique_values, bin_unique_values.shape)
                print('bin_counts', bin_counts, bin_counts.shape)
                for i in range(5):
                    prop_error_dis = self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()) * (obs_sampled - prop_sampled.detach())
                    bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                    print('bin_prop_error_dis', bin_prop_error_dis, bin_prop_error_dis.shape)
                    prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                    print('self.model_abc(prop_user_emb.detach(), prop_item_emb.detach())', self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()), self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()).shape)
                    print('(obs_sampled - prop_sampled.detach())', (obs_sampled - prop_sampled.detach()), (obs_sampled - prop_sampled.detach()).shape)
                    print('epoch idx i prop_abc_loss_dis', epoch, idx, i, prop_abc_loss_dis, - bin_prop_error_dis.abs().sum())

                    optimizer_dis.zero_grad()
                    prop_abc_loss_dis.backward()
                    optimizer_dis.step()
                
                
                prop_error_prop = self.model_abc.predict(prop_user_emb.detach(), prop_item_emb.detach()).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                print('prop_abc_loss_prop', prop_abc_loss_prop, bin_prop_error_prop.shape)
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                # prop_loss = prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                print('self.model_prop(x_sampled)', self.model_prop(x_sampled), self.model_prop(x_sampled).shape)
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        
        
        


class dr_jl_abc_all_ones(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   
            
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)

        print('dr_jl_abc_all_ones initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1  # [N*V]
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                

                prop_error_prop = (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        
        



    


class dr_jl_abc_eqb_all_ones(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   
            
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)

        print('dr_jl_abc_eqb_all_ones initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                

                prop_error_prop = (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        


      

class dr_jl_abc_free(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   


        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)

        print('dr_jl_abc_free initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")
        for name, param in self.model_abc.named_parameters():
            print(f"model_abc name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1  # [N*V]
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                
                prop_error_dis = self.model_abc.forward_logit(x_sampled) * (obs_sampled - prop_sampled.detach())
                bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                

                optimizer_dis.zero_grad()
                prop_abc_loss_dis.backward()
                optimizer_dis.step()
                
                
                prop_error_prop = self.model_abc.predict_logit(x_sampled).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        
        



class dr_jl_abc_eqb_free(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   


        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)

        print('dr_jl_abc_eqb_free initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")
        for name, param in self.model_abc.named_parameters():
            print(f"model_abc name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, dis_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, dis_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                
                prop_error_dis = self.model_abc.forward_logit(x_sampled) * (obs_sampled - prop_sampled.detach())
                bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                

                optimizer_dis.zero_grad()
                prop_abc_loss_dis.backward()
                optimizer_dis.step()
                
                
                prop_error_prop = self.model_abc.predict_logit(x_sampled).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach() 






class dr_jl_abc_l2_norm(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   
            
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)

        print('dr_jl_abc_l2_norm initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, l2_norm=1.0, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, l2_norm, beta, gamma, G, pred_lr, impu_lr, prop_lr, pred_lamb, impu_lamb, prop_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1  # [N*V]
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                

                prop_error_prop = (obs_sampled - prop_sampled)
                bin_prop_error_prop_norm = torch.matmul(prop_error_prop.detach().pow(2).unsqueeze(0), bin_sum_index).squeeze(0).sqrt()
                prop_weight = l2_norm * prop_error_prop.detach() / (bin_prop_error_prop_norm[bin_indices] + 1e-12)
                weighted_prop_error_prop = prop_weight.detach() * prop_error_prop
                
                bin_prop_error_prop = torch.matmul(weighted_prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  





class dr_jl_abc_efb_l2_norm(nn.Module):
    def __init__(self, num_users, num_items, pred_model_name, prop_model_name, abc_model_name, aug_load_param_type, copy_model_pred, embedding_k, batch_size_prop, batch_size, device, is_tensorboard):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.aug_load_param_type = aug_load_param_type
        self.embedding_k = embedding_k

        self.batch_size_prop = batch_size_prop
        self.batch_size = batch_size
        
        self.device = device
        self.is_tensorboard = is_tensorboard


        if pred_model_name == 'mf':
            self.model_pred = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)
            self.model_impu = mf(self.num_users, self.num_items, embedding_k=self.embedding_k)

        
        if prop_model_name == 'logistic_regression':
            self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)   
            
        
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        else:
            pass
            
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)

        print('dr_jl_abc_efb_l2_norm initialized')
        print('num_users', self.num_users)
        print('num_items', self.num_items)
        
        print('aug_load_param_type', aug_load_param_type)
        print('copy_model_pred', copy_model_pred)
        print('embedding_k', self.embedding_k)

        print('batch_size_prop', self.batch_size_prop)
        print('batch_size', self.batch_size)
        
        print('device', self.device)
        print('is_tensorboard', self.is_tensorboard)
        
        for name, param in self.model_pred.named_parameters():
            print(f"model_pred name: {name}, value: {param}")
        for name, param in self.model_impu.named_parameters():
            print(f"model_impu name: {name}, value: {param}")
        for name, param in self.model_prop.named_parameters():
            print(f"model_prop name: {name}, value: {param}")


    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('_compute_IPS', prop_lr, prop_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0

        for epoch in range(num_epochs):
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(x_sampled)

                sub_obs = obs[x_all_idx]
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach()


            if self.is_tensorboard:
                tb_log.add_scalar(f'propensity train/epoch_loss', epoch_loss, epoch)
                
                
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        return epoch


    def fit(self, tb_log, x_all, obs, x, y, grad_type=0, num_epochs=100, num_bins=10, l2_norm=1.0, beta=1.0, gamma=0.05, G=4, pred_lr=0.01, impu_lr=0.01, prop_lr=0.01, pred_lamb=0.0, impu_lamb=0.0, prop_lamb=0.0, stop=5, tol=1e-4): 
        print('fit', grad_type, num_epochs, num_bins, l2_norm, beta, gamma, G, pred_lr, impu_lr, prop_lr, pred_lamb, impu_lamb, prop_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)

        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        for epoch in range(num_epochs):
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                obs_sampled = obs[x_all_idx]
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled)
                
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                

                prop_error_prop = (obs_sampled - prop_sampled)
                bin_prop_error_prop_norm = torch.matmul(prop_error_prop.detach().pow(2).unsqueeze(0), bin_sum_index).squeeze(0).sqrt()
                prop_weight = l2_norm * prop_error_prop.detach() / (bin_prop_error_prop_norm[bin_indices] + 1e-12)
                weighted_prop_error_prop = prop_weight.detach() * prop_error_prop
                
                bin_prop_error_prop = torch.matmul(weighted_prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                             

                # prediction model
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x), gamma, 1.0)

                pred = self.model_pred(sub_x)
                imputation_y = self.model_impu.predict(sub_x)              

                pred_u = self.model_pred(x_sampled) 
                imputation_y1 = self.model_impu.predict(x_sampled)             

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size


                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                
                # imputation model
                pred = self.model_pred.predict(sub_x)
                imputation_y = self.model_impu(sub_x)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
           
                epoch_loss += xent_loss.detach()
            
            
            if self.is_tensorboard:
                tb_log.add_scalar(f'train/epoch_loss', epoch_loss, epoch)
                            
                                        
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
                
        return epoch
    
        
    def predict(self, x):
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            
            return pred.detach()  
        