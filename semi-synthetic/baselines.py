# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time
from tqdm import tqdm

from utils import ndcg_func,  recall_func, precision_func
acc_func = lambda x,y: np.sum(x == y) / len(x)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class MF(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        # Get device from model parameters
        device = next(self.parameters()).device
        # Handle both numpy arrays and tensors
        if isinstance(x, torch.Tensor):
            user_idx = x[:,0].long().to(device)
            item_idx = x[:,1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:,0]).to(device)
            item_idx = torch.LongTensor(x[:,1]).to(device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
           
    def fit(self, x, y, 
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        epoch_pbar = tqdm(range(num_epoch), desc="[MF] Training", disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            batch_pbar = tqdm(range(total_batch), desc=f"[MF] Epoch {epoch}", disable=not verbose, leave=False)
            for idx in batch_pbar:
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()

class MF_BaseModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        # Get device from model parameters
        device = next(self.parameters()).device
        # Handle both numpy arrays and tensors
        if isinstance(x, torch.Tensor):
            user_idx = x[:, 0].long().to(device)
            item_idx = x[:, 1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:, 0]).to(device)
            item_idx = torch.LongTensor(x[:, 1]).to(device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()

class NCF_BaseModel(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(NCF_BaseModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        # Get device from model parameters
        device = next(self.parameters()).device
        # Handle both numpy arrays and tensors
        if isinstance(x, torch.Tensor):
            user_idx = x[:,0].long().to(device)
            item_idx = x[:,1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:,0]).to(device)
            item_idx = torch.LongTensor(x[:,1]).to(device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)
        out = self.sigmoid(self.linear_1(z_emb))


        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    """The neural collaborative filtering method.
    """
    
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)


        if is_training:
            return torch.squeeze(z_emb), U_emb, V_emb
        else:
            return torch.squeeze(z_emb)        
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear_1 = torch.nn.Linear(input_size, input_size / 2, bias = False)
        self.linear_2 = torch.nn.Linear(input_size / 2, 1, bias = True)
        self.xent_func = torch.nn.BCELoss()        
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        
        return torch.squeeze(x)    
    

    
# Keep
class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose = False): 

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        prior_y = y_ips.mean()
        early_stop = 0
        epoch_pbar = tqdm(range(num_epoch), desc="[MF-DR] Training", disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            batch_pbar = tqdm(range(total_batch), desc=f"[MF-DR] Epoch {epoch}", disable=not verbose, leave=False)
            for idx in batch_pbar:
                # mini-batch training
                
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()
                sub_y = torch.Tensor(sub_y).cuda()
                pred, u_emb, v_emb = self.forward(sub_x, True)  
                pred = self.sigmoid(pred)

                x_sampled = x_all[ul_idxs[G * idx* batch_size: G * (idx+1)*batch_size]] 

                pred_ul,_,_ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_y = torch.Tensor([prior_y] * G * batch_size).cuda()
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:batch_size], reduction="sum") # e^ui

                ips_loss = (xent_loss - imputation_loss)/batch_size 

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y, reduction = "sum") 

                loss = (ips_loss + direct_loss)/(x_sampled.shape[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl      
    
    



class MF_CVIB(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).cuda()
        item_idx = torch.LongTensor(x[:,1]).cuda()
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        alpha=0.1, gamma=0.01,
        tol=1e-4, verbose=True):

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size
        early_stop = 0

        epoch_pbar = tqdm(range(num_epoch), desc="[MF-CVIB] Training", disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            batch_pbar = tqdm(range(total_batch), desc=f"[MF-CVIB] Epoch {epoch}", disable=not verbose, leave=False)
            for idx in batch_pbar:
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).cuda()

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred,sub_y)

                # pair wise loss
                x_sampled = x_all[ul_idxs[idx* batch_size:(idx+1)*batch_size]]

                pred_ul,_,_ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                logp_hat = pred.log()

                pred_avg = pred.mean()
                pred_ul_avg = pred_ul.mean()

                info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + self.gamma* torch.mean(pred * logp_hat)

                loss = xent_loss + info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-CVIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-CVIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

class MF_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epoch), desc='[MF-DR-JL] Training', disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            # Add progress bar for batches
            batch_pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}', leave=False, disable=not verbose)
            for idx in batch_pbar:

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl             
    
# Keep
class MF_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epoch), desc='[MF-MRDR-JL] Training', disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            # Add progress bar for batches
            batch_pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}', leave=False, disable=not verbose)
            for idx in batch_pbar:

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2 ) * (1 - 1 / inv_prop.detach())).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl    
        
    
def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True) 

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
    
    # 使用searchsorted将数据分配到相应的桶中
    bin_indices = torch.searchsorted(full_boundaries[1:], data, right=False)
    
    return bin_indices, full_boundaries 

    

    
# Keep
class MF_DR_BIAS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epoch), desc='[MF-DR-BIAS] Training', disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            # Add progress bar for batches
            batch_pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}', leave=False, disable=not verbose)
            for idx in batch_pbar:

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()
                     
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-BIAS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-DR-BIAS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-BIAS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  
    
    
class MF_DR_MSE(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, y_ips,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, gamma = 1,
        tol=1e-4, G=1, verbose = False): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epoch), desc='[MF-DR-MSE] Training', disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            # Add progress bar for batches
            batch_pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}', leave=False, disable=not verbose)
            for idx in batch_pbar:

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].cuda()                

                sub_y = torch.Tensor(sub_y).cuda()

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).cuda()
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).cuda()
                
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                           

                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                # direct loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss               
                                
                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")

                imp_bias_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 3 ) * ((1 - 1 / inv_prop.detach()) ** 2)).sum()
                imp_mrdr_loss = (((e_loss - e_hat_loss) ** 2) * (inv_prop.detach() ** 2 ) * (1 - 1 / inv_prop.detach())).sum()
                imp_loss = gamma * imp_bias_loss + (1-gamma) * imp_mrdr_loss
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-MSE] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-DR-MSE] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-MSE] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        # pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl      



class MF_DR_V2(nn.Module):
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)        
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def fit(self, x, y, stop = 5, alpha = 1, beta = 1, theta = 1, eta = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((y, (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        #observation = prediction.type(torch.LongTensor)

        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epoch), desc='[MF-DR-V2] Training', disable=not verbose)
        for epoch in epoch_pbar:
            # sampling counterfactuals
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            # Add progress bar for batches
            batch_pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}', leave=False, disable=not verbose)
            for idx in batch_pbar:
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).cuda()
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x).cuda()                
                
                x_all_idx = ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]
                x_sampled = x_all[x_all_idx]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled).cuda()             
                
                xent_loss = -torch.sum((sub_y * torch.log(pred + 1e-6) + (1-sub_y) * torch.log(1 - pred + 1e-6)) * inv_prop)
                imputation_loss = -torch.sum(imputation_y * torch.log(pred + 1e-6) + (1-imputation_y) * torch.log(1 - pred + 1e-6))
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                # direct loss
                                
                direct_loss = -torch.sum(imputation_y1 * torch.log(pred_u + 1e-6) + (1-imputation_y1) * torch.log(1 - pred_u + 1e-6))
                
                dr_loss = (ips_loss + direct_loss)/x_sampled.shape[0]
                                                  
                pred = self.prediction_model.predict(sub_x).cuda()
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = -sub_y * torch.log(pred + 1e-6) - (1-sub_y) * torch.log(1 - pred + 1e-6)
                e_hat_loss = -imputation_y * torch.log(pred + 1e-6) - (1-imputation_y) * torch.log(1 - pred + 1e-6)
                
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                # ctr loss
                
                sub_obs = torch.Tensor(obs[x_all_idx]).cuda()
                sub_entire_y = torch.Tensor(y_entire[x_all_idx]).cuda()

                inv_prop_all = 1/torch.clip(self.propensity_model.forward(x_sampled), gamma, 1)
                prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs)                                    
                pred = self.prediction_model.forward(x_sampled)
                
                pred_loss = F.binary_cross_entropy(1/inv_prop_all * pred, sub_entire_y)

                ones_all = torch.ones(len(inv_prop_all)).cuda()
                w_all = torch.divide(sub_obs,1/inv_prop_all)-torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
                bmse_loss = (torch.mean(w_all * pred))**2
                
                loss = alpha * prop_loss + beta * pred_loss + theta * imp_loss + dr_loss + eta * bmse_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                     
                epoch_loss += xent_loss.detach().cpu().numpy()                         
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-V2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            epoch_pbar.set_postfix({'loss': epoch_loss})
            if epoch % 10 == 0 and verbose:
                print("[MF-DR-V2] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-V2] Reach preset epochs, it seems does not converge.")
        
        torch.save(self.propensity_model.state_dict(), 'weight_model0.pth')
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    

# Keep
# Helper function for dr_jl_abc
def equal_frequency_binning(data, quantiles, n_bins=4):
    """
    对输入数据执行等频分桶
    
    参数:
        data (torch.Tensor): 输入的一维数据
        quantiles (torch.Tensor): 分位点
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
    bin_indices = torch.bucketize(data, full_boundaries, right=False) - 1
    
    # 返回桶索引和边界值
    return bin_indices, full_boundaries[1:-1]  # 去除无穷大边界


# Base models from propensity framework for dr_jl_abc
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
        # Handle both numpy arrays and tensors
        device = next(self.parameters()).device
        if isinstance(x, torch.Tensor):
            user_idx = x[:, 0].long().to(device)
            item_idx = x[:, 1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:, 0]).to(device)
            item_idx = torch.LongTensor(x[:, 1]).to(device)
        user_emb = self.user_emb_table(user_idx)
        item_emb = self.item_emb_table(item_idx)
        out = torch.sigmoid((user_emb * item_emb).sum(dim=1))
        return out  
    
    
    def forward_logit(self, x):
        # Handle both numpy arrays and tensors
        device = next(self.parameters()).device
        if isinstance(x, torch.Tensor):
            user_idx = x[:, 0].long().to(device)
            item_idx = x[:, 1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:, 0]).to(device)
            item_idx = torch.LongTensor(x[:, 1]).to(device)
        user_emb = self.user_emb_table(user_idx)
        item_emb = self.item_emb_table(item_idx)
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
    
    def forward(self, x, user_emb=None, item_emb=None):
        # Support both modes: with indices or with embeddings
        if user_emb is not None and item_emb is not None:
            # Called with embeddings directly (for model_abc)
            z_emb = torch.cat([user_emb, item_emb], axis=1)
        else:
            # Called with indices (original behavior)
            user_emb = self.user_emb_table(x[:, 0])
            item_emb = self.item_emb_table(x[:, 1])
            z_emb = torch.cat([user_emb, item_emb], axis=1)
        out = torch.sigmoid(self.linear_1(z_emb))
        return torch.squeeze(out)        
    
        
    def forward_logit(self, x):
        # Handle both numpy arrays and tensors
        device = next(self.parameters()).device
        if isinstance(x, torch.Tensor):
            user_idx = x[:, 0].long().to(device)
            item_idx = x[:, 1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:, 0]).to(device)
            item_idx = torch.LongTensor(x[:, 1]).to(device)
        user_emb = self.user_emb_table(user_idx)
        item_emb = self.item_emb_table(item_idx)
        z_emb = torch.cat([user_emb, item_emb], axis=1)
        out = self.linear_1(z_emb)
        return torch.squeeze(out)      
    
    
    def get_emb(self, x):
        # Handle both numpy arrays and tensors
        device = next(self.parameters()).device
        if isinstance(x, torch.Tensor):
            user_idx = x[:, 0].long().to(device)
            item_idx = x[:, 1].long().to(device)
        else:
            user_idx = torch.LongTensor(x[:, 0]).to(device)
            item_idx = torch.LongTensor(x[:, 1]).to(device)
        user_emb = self.user_emb_table(user_idx)
        item_emb = self.item_emb_table(item_idx)
        
        return user_emb, item_emb
        
        
    def predict(self, x, user_emb=None, item_emb=None):
        with torch.no_grad():
            pred = self.forward(x, user_emb, item_emb)
            
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
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        # Default parameters for compatibility
        self.batch_size_prop = kwargs.get('batch_size_prop', 32768)
        self.batch_size = kwargs.get('batch_size', 4096)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_tensorboard = False  # Disable tensorboard by default
        
        # Model configuration
        pred_model_name = kwargs.get('pred_model_name', 'mf')
        prop_model_name = kwargs.get('prop_model_name', 'logistic_regression')
        abc_model_name = kwargs.get('abc_model_name', 'mlp')
        self.aug_load_param_type = kwargs.get('aug_load_param_type', '')
        copy_model_pred = kwargs.get('copy_model_pred', 0)
        
        # Initialize models
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
        
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)
        print('dr_jl_abc initialized')
        
    def _compute_IPS(self, tb_log, x_all, obs, num_epochs=200, prop_lr=0.01, prop_lamb=0.0, stop=5, tol=1e-4):
        print('\n' + '='*60)
        print('[dr_jl_abc] Stage 1: Computing Initial Propensity Scores')
        print(f'Parameters: lr={prop_lr}, lambda={prop_lamb}, epochs={num_epochs}')
        print('='*60)
        
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        
        last_loss = 1e9
        early_stop = 0
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc='[dr_jl_abc] IPS Stage', disable=False)  # Enable progress bar
        for epoch in epoch_pbar:
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
                
            epoch_pbar.set_postfix({'loss': epoch_loss.item() if torch.is_tensor(epoch_loss) else epoch_loss})
            
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print(f'\n[dr_jl_abc] IPS Stage completed early at epoch {epoch} (converged)')
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
        print(f'\n[dr_jl_abc] IPS Stage completed after {num_epochs} epochs')
        
        # Save Stage-1 propensity model state for potential reuse
        self.stage_1_prop_state = self.model_prop.state_dict()
        
        return epoch
        
    def fit(self, x, y, y_ips=None,
            num_epoch=100, batch_size=128, lr=0.05, lamb=0, 
            tol=1e-4, verbose=False, **kwargs):
        
        # Convert numpy arrays to tensors
        if not torch.is_tensor(x):
            x = torch.LongTensor(x).to(self.device)
        if not torch.is_tensor(y):
            y = torch.FloatTensor(y).to(self.device)
            
        # Default parameters
        grad_type = kwargs.get('grad_type', 0)
        num_bins = kwargs.get('num_bins', 10)
        beta = kwargs.get('beta', 1.0)
        gamma = kwargs.get('gamma', 1e-12)  # Original default value from propensity framework
        G = kwargs.get('G', 4)
        pred_lr = kwargs.get('pred_lr', 0.01)
        impu_lr = kwargs.get('impu_lr', 0.01)
        prop_lr = kwargs.get('prop_lr', 0.01)
        dis_lr = kwargs.get('dis_lr', 0.01)
        pred_lamb = kwargs.get('pred_lamb', 0.0)
        impu_lamb = kwargs.get('impu_lamb', 0.0)
        prop_lamb = kwargs.get('prop_lamb', 0.0)
        dis_lamb = kwargs.get('dis_lamb', 0.0)
        stop = kwargs.get('stop', 5)
        
        # Use the batch_size parameter passed to fit() instead of initialization value
        self.batch_size = batch_size
        
        # Generate all samples and observation matrix
        x_all = generate_total_sample(self.num_users, self.num_items)
        x_all = torch.LongTensor(x_all).to(self.device)
        obs = torch.sparse.FloatTensor(
            torch.cat([x[:, 0].unsqueeze(dim=0), x[:, 1].unsqueeze(dim=0)], dim=0), 
            torch.ones_like(y), 
            torch.Size([self.num_users, self.num_items])
        ).to_dense().reshape(-1)
        
        # Compute initial propensity scores
        tb_log = None  # No tensorboard
        prop_epoch = self._compute_IPS(tb_log, x_all, obs, num_epochs=200, prop_lr=prop_lr, prop_lamb=prop_lamb)
        
        # Main training
        # Note: Stage-1 propensity model state is saved in self.stage_1_prop_state
        # and could be restored if needed to preserve Stage-1 learning
        print('\n' + '='*60)
        print('[dr_jl_abc] Stage 2: Main Training')
        print(f'Parameters: epochs={num_epoch}, batch_size={batch_size}, G={G}')
        print(f'Learning rates: pred={pred_lr}, impu={impu_lr}, prop={prop_lr}, dis={dis_lr}')
        print(f'Regularization: pred={pred_lamb}, impu={impu_lamb}, prop={prop_lamb}, dis={dis_lamb}')
        print(f'Other params: num_bins={num_bins}, beta={beta}, gamma={gamma}, tol={tol}, stop={stop}')
        print('='*60)
        print('fit', grad_type, num_epoch, num_bins, beta, gamma, G, pred_lr, impu_lr, prop_lr, dis_lr, pred_lamb, impu_lamb, prop_lamb, dis_lamb, stop)    
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=pred_lamb)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=impu_lamb)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=prop_lamb)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)
        num_samples = len(x)
        total_batch = num_samples // self.batch_size
                
        last_loss = 1e9
        early_stop = 0
        
        # Fix bin_edges to match original model - use [1:-1] slice to remove 0 and 1
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        
        # Add progress bar for epochs
        epoch_pbar = tqdm(range(num_epoch), desc='[dr_jl_abc] Training', disable=not verbose)
        for epoch in epoch_pbar:
            all_idx = torch.randperm(num_samples, device=self.device)
            ul_idxs = torch.randperm(x_all.shape[0], device=self.device)
            epoch_loss = 0
            
            # Add progress bar for batches
            batch_pbar = tqdm(range(total_batch), desc=f'Epoch {epoch}', leave=False, disable=not verbose)
            for idx in batch_pbar:  
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
                
                # Use bucketize for dr_jl_abc (not eqb variant)
                # Note: dr_jl_abc_eqb uses equal_frequency_binning instead
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1
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
            
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print(f"[dr_jl_abc] epoch:{epoch}, xent:{epoch_loss}")
                    break
                else:
                    early_stop += 1
            last_loss = epoch_loss
            
            epoch_pbar.set_postfix({'loss': epoch_loss.item() if torch.is_tensor(epoch_loss) else epoch_loss})
            if epoch % 10 == 0 and verbose:
                print(f"[dr_jl_abc] epoch:{epoch}, xent:{epoch_loss}")
                
            if epoch == num_epoch - 1:
                print("[dr_jl_abc] Reach preset epochs, it seems does not converge.")
        
        # Training completed
        print(f'\n[dr_jl_abc] Stage 2 Completed: Trained for {epoch + 1} epochs')
        print(f'[dr_jl_abc] Final loss: {epoch_loss.item() if torch.is_tensor(epoch_loss) else epoch_loss:.6f}')
        print('='*60 + '\n')
    
    def predict(self, x):
        # Convert input to tensor if needed
        if not torch.is_tensor(x):
            x = torch.LongTensor(x).to(self.device)
        
        with torch.no_grad():
            pred = self.model_pred.predict(x)
            return pred.detach()
    

