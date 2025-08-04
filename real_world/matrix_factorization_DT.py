# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time
from tqdm import tqdm
import copy

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(abs(x-y))

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, is_training=False):
        if isinstance(x, np.ndarray):
            x = torch.LongTensor(x).to(self.device)
        user_idx = x[:,0]
        item_idx = x[:,1]
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
    
    def predict(self, x):
        with torch.no_grad():
            pred = self.forward(x)
            return pred

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = self.sigmoid(torch.sum(U_emb.mul(V_emb), 1))

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def forward_logit(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out
    
    def get_emb(self, x):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        return U_emb, V_emb

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.sigmoid(self.linear_1(z_emb))

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)        
    
    def forward_logit(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        out = self.linear_1(z_emb)

        if is_training:
            return torch.squeeze(out), U_emb, V_emb
        else:
            return torch.squeeze(out)
    
    def get_emb(self, x):
        user_idx = torch.LongTensor(x[:, 0]).to(self.device)
        item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        return U_emb, V_emb
        
    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu()


class Embedding_Sharing(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(Embedding_Sharing, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(self.device)
        item_idx = torch.LongTensor(x[:,1]).to(self.device)
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
    


class MF_DR(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        pbar = tqdm(range(num_epoch), desc="[MF-DR-PS] Computing IPS", disable=not verbose)
        for epoch in pbar:

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            pbar.set_postfix({'loss': epoch_loss})
            
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("\n[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch == num_epoch - 1:
                print("[MF-DR-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, prior_y, gamma,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G = 1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size
        
        prior_y = prior_y.mean()
        early_stop = 0
        pbar = tqdm(range(num_epoch), desc="[MF-DR] Training", disable=not verbose)
        for epoch in pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)

                sub_y = torch.Tensor(sub_y).to(self.device)

                pred, u_emb, v_emb = self.prediction_model.forward(sub_x, True)  

                x_sampled = x_all[ul_idxs[G * idx* self.batch_size: G * (idx+1)*self.batch_size]] # batch size

                pred_ul,_,_ = self.prediction_model.forward(x_sampled, True)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                
                imputation_y = torch.Tensor([prior_y]* G *selected_idx.shape[0]).to(self.device)
                imputation_loss = F.binary_cross_entropy(pred, imputation_y[0:self.batch_size], reduction="sum") # e^ui

                ips_loss = (xent_loss - imputation_loss) # batch size

                # direct loss
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_y,reduction="sum")

                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            pbar.set_postfix({'loss': epoch_loss})
            
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("\n[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()


# timing expriment
class MF_DR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
        
        pbar = tqdm(range(num_epoch), desc="[MF-DRJL-PS] Computing IPS", disable=not verbose)
        for epoch in pbar:

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            pbar.set_postfix({'loss': epoch_loss})
            
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("\n[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch == num_epoch - 1:
                print("[MF-DRJL-PS] Reach preset epochs, it seems does not converge.")        

    def fit(self, x, y, stop = 5,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        pbar = tqdm(range(num_epoch), desc="[MF-DR-JL] Training", disable=not verbose)
        for epoch in pbar: 
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).to(self.device)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).to(self.device)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                 
                ips_loss = (xent_loss - imputation_loss) # batch size
                                
                # direct loss                
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                                           
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).to(self.device)
                imputation_y = self.imputation_model.forward(sub_x)

                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()
    


class MF_DR_DCE(nn.Module):
    """MF-DR with Doubly Robust estimation and Calibrated Expected (DCE) loss for propensity scores"""
    
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        
        # Initialize models same as MF_DR_JL
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, 
            batch_size=self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, 
            batch_size=self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, 
            batch_size=self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _compute_IPS(self, x, num_epoch=1000, lr=0.05, lamb=0, 
                     tol=1e-3, verbose=False, ece_weight=0.1, n_bins=10):
        """
        Compute propensity scores with ECE loss for better calibration
        
        Args:
            x: Training data
            num_epoch: Maximum epochs
            lr: Learning rate
            lamb: L2 regularization
            tol: Early stopping tolerance
            verbose: Print progress
            ece_weight: Weight for ECE loss term
            n_bins: Number of bins for ECE calculation
        """
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
                           shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_total_loss = 1e9  # Track combined loss for early stopping
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
        
        pbar = tqdm(range(num_epoch), desc="[MF-DR-DCE-PS] Computing IPS with ECE", disable=not verbose)
        for epoch in pbar:
            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_mse = 0
            epoch_ece = 0
            epoch_total_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)

                # MSE loss
                mse_loss = nn.MSELoss()(prop, sub_obs)
                
                # ECE loss with equal-frequency binning
                ece_loss = compute_ece_loss_equal_freq(prop, sub_obs, n_bins)
                
                # Combined loss
                prop_loss = mse_loss + ece_weight * ece_loss
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_mse += mse_loss.detach().cpu().numpy()
                epoch_ece += ece_loss.detach().cpu().numpy()
                epoch_total_loss += prop_loss.detach().cpu().numpy()

            pbar.set_postfix({'mse': epoch_mse, 'ece': epoch_ece, 'total': epoch_total_loss})
            
            # Early stopping based on combined loss
            relative_loss_div = (last_total_loss - epoch_total_loss)/(last_total_loss + 1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("\n[MF-DR-DCE-PS] Early stopping based on combined loss convergence")
                        print("[MF-DR-DCE-PS] epoch:{}, mse_loss:{}, ece_loss:{}, total_loss:{}".format(
                            epoch, epoch_mse, epoch_ece, epoch_total_loss))
                    break
                early_stop += 1
            else:
                early_stop = 0  # Reset early stop counter if improving
                
            last_total_loss = epoch_total_loss

            if epoch == num_epoch - 1:
                print("[MF-DR-DCE-PS] Reach preset epochs, combined loss may not have converged.")
    
    def fit(self, x, y, stop=5, num_epoch=1000, lr=0.05, lamb=0, gamma=0.1,
            tol=1e-4, G=1, verbose=True, ece_weight=0.1, n_bins=10, 
            prop_epochs=100, prop_lr=0.005):
        """
        Fit the model with ECE-regularized propensity scores
        
        Args:
            x: Observed user-item pairs
            y: Observed ratings
            stop: Early stopping patience
            num_epoch: Maximum training epochs
            lr: Learning rate
            lamb: L2 regularization
            gamma: Propensity clipping threshold
            tol: Early stopping tolerance
            G: Ratio of unobserved to observed samples
            verbose: Print progress
            ece_weight: Weight for ECE loss in propensity training
            n_bins: Number of bins for ECE calculation
            prop_epochs: Number of epochs for propensity model training
            prop_lr: Learning rate for propensity model
        """
        # First compute propensity scores with ECE loss
        self._compute_IPS(x, num_epoch=prop_epochs, lr=prop_lr, lamb=lamb, 
                         tol=tol, verbose=verbose, ece_weight=ece_weight, n_bins=n_bins)
        
        # Then run doubly robust training (same as MF_DR_JL)
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x) 
        total_batch = num_sample // self.batch_size
        early_stop = 0
        
        pbar = tqdm(range(num_epoch), desc="[MF-DR-DCE] Training", disable=not verbose)
        for epoch in pbar: 
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                sub_y = torch.Tensor(sub_y).to(self.device)
                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).to(self.device)

                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                 
                ips_loss = (xent_loss - imputation_loss)
                                
                # direct loss                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                                           
                epoch_loss += xent_loss.detach().cpu().numpy()                

                # Update imputation model
                pred = self.prediction_model.predict(sub_x).to(self.device)
                imputation_y = self.imputation_model.forward(sub_x)

                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-DCE] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-DCE] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-DCE] Reach preset epochs, it seems does not converge.")
    
    def predict(self, x):
        """Make predictions for user-item pairs"""
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()


class MF_DR_BMSE(MF_DR_JL):
    """
    Matrix Factorization Doubly Robust with BMSE regularization.
    Inherits from MF_DR_JL and adds BMSE loss to the prediction model.
    """
    def __init__(
        self,
        num_users,
        num_items,
        batch_size,
        batch_size_prop,
        embedding_k=4,
        bmse_weight=1.0,
        *args, **kwargs
    ):
        super().__init__(num_users, num_items, batch_size,
                         batch_size_prop, embedding_k, *args, **kwargs)
        self.bmse_weight = bmse_weight

    def get_phi(self, x):
        """
        φ(x) = σ(u^T v) mapped to [0,1] range for BMSE calculation
        """
        if isinstance(x, np.ndarray):
            x = torch.LongTensor(x).to(self.device)

        user_idx, item_idx = x[:, 0], x[:, 1]
        U_emb = self.prediction_model.W(user_idx)
        V_emb = self.prediction_model.H(item_idx)
        logits = torch.sum(U_emb.mul(V_emb), dim=1, keepdim=True)
        return torch.sigmoid(logits)  # shape: (B,1)

    def fit(self, x, y, stop=5,
            num_epoch=1000, lr=0.05, lamb=0, gamma=0.1,
            tol=1e-4, G=1, verbose=True):
        """
        Fit the model using MF_DR_JL's logic with additional BMSE regularization.
        Propensity scores should be pre-computed using _compute_IPS().
        """
        
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        x_all = generate_total_sample(self.num_users, self.num_items)
        num_sample = len(x)
        total_batch = num_sample // self.batch_size
        early_stop = 0
        
        pbar = tqdm(range(num_epoch), desc="[MF-DR-BMSE] Training", disable=not verbose)
        for epoch in pbar:
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # Get pre-trained propensity scores
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                
                sub_y = torch.Tensor(sub_y).to(self.device)
                
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)
                
                x_sampled = x_all[ul_idxs[G*idx*self.batch_size : G*(idx+1)*self.batch_size]]
                
                pred_u = self.prediction_model.forward(x_sampled)
                imputation_y1 = self.imputation_model.predict(x_sampled).to(self.device)

                # Standard DR loss components
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum")
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                ips_loss = xent_loss - imputation_loss
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                
                # Calculate BMSE loss
                # Get phi for observed samples
                phi_obs = self.get_phi(sub_x)
                # Clip propensity scores for numerical stability
                p_hat_obs = torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)
                
                # Get phi for unobserved samples  
                phi_unobs = self.get_phi(x_sampled)
                p_hat_unobs = torch.clip(self.propensity_model.forward(x_sampled).detach(), gamma, 1)
                
                # Ensure propensity scores have correct shape
                if len(p_hat_obs.shape) == 1:
                    p_hat_obs = p_hat_obs.unsqueeze(1)
                if len(p_hat_unobs.shape) == 1:
                    p_hat_unobs = p_hat_unobs.unsqueeze(1)
                
                # For observed samples (o=1): term = (1/p̂) * φ
                term_obs = (1.0 / p_hat_obs) * phi_obs
                
                # For unobserved samples (o=0): term = -(1/(1-p̂)) * φ  
                term_unobs = -(1.0 / (1 - p_hat_unobs)) * phi_unobs
                
                # Compute the average over all samples in D
                # D = observed samples ∪ unobserved samples
                total_samples = sub_x.shape[0] + x_sampled.shape[0]
                avg_term = (term_obs.sum(dim=0) + term_unobs.sum(dim=0)) / total_samples
                
                # BMSE is the squared norm of the average
                bmse_loss = avg_term.pow(2).sum()
                
                # Total loss for prediction model
                loss = (ips_loss + direct_loss) / x_sampled.shape[0] + self.bmse_weight * bmse_loss

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

                # Update imputation model (same as parent)
                pred = self.prediction_model.predict(sub_x).to(self.device)
                imputation_y = self.imputation_model.forward(sub_x)

                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum()

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-DR-BMSE] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-BMSE] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-BMSE] Reach preset epochs, it seems does not converge.")
    
    # predict() method is inherited from parent class




# timing expriment
class MF_MRDR_JL(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users = self.num_users, num_items = self.num_items, batch_size = self.batch_size, embedding_k=self.embedding_k, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        pbar = tqdm(range(num_epoch), desc="[MF-MRDR-JL] Computing IPS", disable=not verbose)
        for epoch in pbar:

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                # propensity score
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)
                
                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-MRDRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                pbar.set_postfix({'loss': epoch_loss})

            if epoch == num_epoch - 1:
                print("[MF-MRDRJL-PS] Reach preset epochs, it seems does not converge.")        


    def fit(self, x, y, stop = 1,
        num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1,
        tol=1e-4, G=1, verbose=True): 

        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9

        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0

        pbar = tqdm(range(num_epoch), desc="[MF-MRDR-JL] Training", disable=not verbose)
        for epoch in pbar: 
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]

                # propensity score

                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)             
                
                sub_y = torch.Tensor(sub_y).to(self.device)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x)
                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.forward(x_sampled)
          
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")
                        
                ips_loss = (xent_loss - imputation_loss) # batch size
                
                
                # direct loss
                
                
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                 
                # propensity loss
                
                loss = (ips_loss + direct_loss)/x_sampled.shape[0]

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                     
                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation_model.forward(sub_x)
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2
                            ) * (inv_prop.detach())**2 *(1-1/inv_prop.detach())).sum()   

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                pbar.set_postfix({'loss': epoch_loss})

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")
                
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()            
        

# timing expriment
# DR-BIAS 
class MF_DR_BIAS(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, x, y, gamma=0.1,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, G=1, verbose = False): 

        # First train the propensity model
        self._compute_IPS(x, num_epoch=num_epoch, lr=lr, lamb=lamb, tol=tol, verbose=verbose)
        
        optimizer_prediction = torch.optim.Adam(
            self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)

        last_loss = 1e9

            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // self.batch_size

        early_stop = 0
        pbar = tqdm(range(num_epoch), desc="[MF-DR-BIAS] Training", disable=not verbose)
        for epoch in pbar:
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score using trained model
                inv_prop = 1/torch.clip(self.propensity_model.forward(sub_x).detach(), gamma, 1)                

                sub_y = torch.Tensor(sub_y).to(self.device)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).to(self.device)
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).to(self.device)
                
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

                pred = self.prediction_model.predict(sub_x).to(self.device)
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

            if epoch % 10 == 0 and verbose:
                pbar.set_postfix({'loss': epoch_loss})

            if epoch == num_epoch - 1:
                print("[MF-DR-BIAS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self, x,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):
        
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs)
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0

        pbar = tqdm(range(num_epoch), desc="[MF-DR-BIAS-PS] Computing IPS", disable=not verbose)
        for epoch in pbar:

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            pbar.set_postfix({'loss': epoch_loss})
            
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("\n[MF-DR-BIAS-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch == num_epoch - 1:
                print("[MF-DR-BIAS-PS] Reach preset epochs, it seems does not converge.")  
    
    

        
    
def compute_ece_loss(props, obs, n_bins=10):
    """
    Compute Expected Calibration Error loss for propensity scores
    
    Args:
        props: Predicted propensity scores
        obs: Observed binary outcomes
        n_bins: Number of bins for calibration
    
    Returns:
        ece_loss: Expected calibration error
    """
    # Create equal-width bins (no sorting, to match evaluation)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=props.device)
    # Use right=False and subtract 1 to match evaluation
    bin_indices = torch.bucketize(props, bin_boundaries, right=False) - 1
    # Clamp to valid range
    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
    
    ece_loss = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_props = props[mask]
            bin_obs = obs[mask]
            # ECE: |avg(predicted) - avg(observed)|
            bin_ece = torch.abs(bin_props.mean() - bin_obs.mean())
            ece_loss += bin_ece * mask.sum() / props.shape[0]
    
    return ece_loss


def compute_ece_loss_equal_freq(props, obs, n_bins=10):
    """
    Compute Expected Calibration Error loss using equal-frequency binning
    
    Args:
        props: Predicted propensity scores
        obs: Observed binary outcomes
        n_bins: Number of bins for calibration
    
    Returns:
        ece_loss: Expected calibration error
    """
    # Create equal-frequency bins using quantiles
    # Generate quantile points (excluding 0 and 1)
    quantiles = torch.linspace(0, 1, n_bins + 1, device=props.device)[1:-1]
    
    # Use equal_frequency_binning logic
    boundaries = torch.quantile(props, quantiles)
    
    # Construct full boundaries (including min and max)
    full_boundaries = torch.cat([
        torch.tensor([float('-inf')], device=props.device), 
        boundaries, 
        torch.tensor([float('inf')], device=props.device)
    ])
    
    # Assign bin indices using right=True to match equal_frequency_binning
    bin_indices = torch.bucketize(props, full_boundaries, right=True) - 1
    bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
    
    ece_loss = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_props = props[mask]
            bin_obs = obs[mask]
            # ECE: |avg(predicted) - avg(observed)|
            bin_ece = torch.abs(bin_props.mean() - bin_obs.mean())
            ece_loss += bin_ece * mask.sum() / props.shape[0]
    
    return ece_loss


def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)

def equal_frequency_binning(data, quantiles, n_bins=4):
    """
    Perform equal frequency binning on input data
    
    Args:
        data (torch.Tensor): Input 1D data (propensity scores)
        quantiles (torch.Tensor): Quantile values (e.g., [0.1, 0.2, ..., 0.9] for 10 bins)
        n_bins (int): Number of bins
    
    Returns:
        bin_indices (torch.Tensor): Bin index for each element (0 to n_bins-1)
        full_boundaries (torch.Tensor): Bin boundaries (length n_bins+1)
    """
    # Calculate boundaries at the quantile points
    boundaries = torch.quantile(data, quantiles).to(data.device)
    
    # Construct full boundaries (including min and max)
    full_boundaries = torch.cat([
        torch.tensor([float('-inf')]).to(data.device), 
        boundaries, 
        torch.tensor([float('inf')]).to(data.device)
    ])
    
    # Assign bin indices (0 ~ n_bins-1)
    bin_indices = torch.bucketize(data, full_boundaries, right=True) - 1
    
    # Return bin indices and boundaries
    return bin_indices, full_boundaries[1:-1]  # Remove infinity boundaries


class logistic_regression(nn.Module):    
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.user_emb_table = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.item_emb_table = torch.nn.Embedding(self.num_items, self.embedding_k)
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, 1, bias=True)
        
    def forward(self, x_or_user_emb, item_emb=None):
        if item_emb is None:
            # x is indices - handle both numpy arrays and tensors
            if isinstance(x_or_user_emb, np.ndarray):
                user_idx = torch.LongTensor(x_or_user_emb[:, 0])
                item_idx = torch.LongTensor(x_or_user_emb[:, 1])
            else:  # already a tensor
                user_idx = x_or_user_emb[:, 0].long()
                item_idx = x_or_user_emb[:, 1].long()
            
            # Move to device if needed
            device = next(self.parameters()).device
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            
            user_emb = self.user_emb_table(user_idx)
            item_emb = self.item_emb_table(item_idx)
        else:
            # x_or_user_emb is user_emb, item_emb is provided
            user_emb = x_or_user_emb
            
        z_emb = torch.cat([user_emb, item_emb], axis=1)
        out = self.linear_1(z_emb)
        return torch.sigmoid(torch.squeeze(out))
    
    def forward_logit(self, x):
        # Handle both numpy arrays and tensors
        if isinstance(x, np.ndarray):
            user_idx = torch.LongTensor(x[:, 0])
            item_idx = torch.LongTensor(x[:, 1])
        else:  # already a tensor
            user_idx = x[:, 0].long()
            item_idx = x[:, 1].long()
        
        # Move to device if needed
        device = next(self.parameters()).device
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        
        user_emb = self.user_emb_table(user_idx)
        item_emb = self.item_emb_table(item_idx)
        z_emb = torch.cat([user_emb, item_emb], axis=1)
        out = self.linear_1(z_emb)
        return torch.squeeze(out)      
    
    def predict(self, x_or_user_emb, item_emb=None):
        with torch.no_grad():
            pred = self.forward(x_or_user_emb, item_emb)
            return pred 
    
    def predict_logit(self, x):
        with torch.no_grad():
            pred = self.forward_logit(x)
            return pred
    
    def get_emb(self, x):
        # Handle both numpy arrays and tensors
        if isinstance(x, np.ndarray):
            user_idx = torch.LongTensor(x[:, 0])
            item_idx = torch.LongTensor(x[:, 1])
        else:  # already a tensor
            user_idx = x[:, 0].long()
            item_idx = x[:, 1].long()
        
        device = next(self.parameters()).device
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        
        user_emb = self.user_emb_table(user_idx)
        item_emb = self.item_emb_table(item_idx)
        return user_emb, item_emb


class mlp(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k, bias=True)
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=True)
        self.non_linear = torch.nn.ReLU()
        
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



class MF_Minimax(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, embedding_k1=8, 
                 abc_model_name='logistic_regression', copy_model_pred=1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.embedding_k1 = embedding_k1
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop  # Same as batch_size for simplicity
        
        # Use MF from matrix_factorization_DT.py for prediction and imputation
        self.model_pred = MF(self.num_users, self.num_items, self.batch_size, embedding_k=self.embedding_k1)
        self.model_impu = MF(self.num_users, self.num_items, self.batch_size, embedding_k=self.embedding_k1)
        
        # Use logistic_regression for propensity
        self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Use logistic_regression or mlp for abc
        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Copy model weights if specified
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        
        # Move to cuda if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    # 原来lamb=0
    def _compute_IPS(self, x, num_epoch=1000, lr=0.05, lamb=1e-5, tol=1e-4, verbose=False):
        print('Stage1: computing_IPS', lr, lamb)
        
        # Generate obs from x
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=lr, weight_decay=lamb)
        
        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        last_loss = 1e9
        early_stop = 0
        stop = 5  # Default stop value

        for epoch in tqdm(range(num_epoch), desc="Computing IPS", disable=not verbose):
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(torch.LongTensor(x_sampled).to(self.device))

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)
                
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print("[Minimax-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print("[Minimax-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
            
        return epoch

    def fit(self, x, y, x_test=None, y_test=None, G=4, alpha=1, beta=1, theta=1, num_epoch=1000, 
            pred_lr=0.05, impu_lr=0.05, prop_lr=0.05, dis_lr=0.05,
            lamb_prop=0, lamb_pred=0, lamb_imp=0, dis_lamb=0, gamma=0.05, num_bins=10,
            tol=1e-4, verbose=True, early_stop_patience=20, early_stop_min_delta=1e-4, 
            eval_freq=10, progress_callback=None):
        """
        Train the MF_Minimax model using adversarial propensity learning and doubly robust estimation
        
        Args:
            x: Training user-item pairs
            y: Training ratings
            x_test: Test user-item pairs (optional, for early stopping)
            y_test: Test ratings (optional, for early stopping)
            G: Ratio of unobserved to observed samples for DR estimation
            alpha: Unused parameter (kept for compatibility)
            beta: Weight for adversarial loss in propensity training (higher = more adversarial regularization)
            theta: Unused parameter (kept for compatibility)
            num_epoch: Maximum training epochs
            pred_lr: Learning rate for prediction model
            impu_lr: Learning rate for imputation model
            prop_lr: Learning rate for propensity model
            dis_lr: Learning rate for discriminator model
            lamb_prop: Weight decay for propensity model
            lamb_pred: Weight decay for prediction model
            lamb_imp: Weight decay for imputation model
            dis_lamb: Weight decay for discriminator model
            gamma: Propensity score clipping threshold (avoids extreme importance weights)
            num_bins: Number of bins for propensity score stratification
            tol: Early stopping tolerance for training loss
            verbose: Whether to print training progress
            early_stop_patience: Patience for early stopping based on test AUC (if test data provided)
            early_stop_min_delta: Minimum improvement required to reset patience counter
            eval_freq: Frequency of evaluation on test set (every N epochs)
            progress_callback: Optional callback function for progress updates (epoch, total_epochs, loss, train_auc, test_auc)
        """ 
        
        print('Stage2: fitting', G, alpha, beta, theta, gamma, num_bins, pred_lr, impu_lr, prop_lr, dis_lr, lamb_prop, lamb_pred, lamb_imp, dis_lamb)
        
        # Create optimizers with separate learning rates and weight decays
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=lamb_pred)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=lamb_imp)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=lamb_prop)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)
        
        
        # Generate all samples and obs
        x_all = generate_total_sample(self.num_users, self.num_items)
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        num_samples = len(x)
        total_batch = num_samples // self.batch_size
        
        last_loss = 1e9
        early_stop = 0
        stop = 5  # Default stop value
        
        bin_edges = torch.linspace(0, 1, steps=int(num_bins) + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        
        # Initialize early stopping variables
        use_early_stopping = x_test is not None and y_test is not None
        if use_early_stopping:
            from sklearn.metrics import roc_auc_score
            best_test_auc = -float('inf')
            patience_counter = 0
            best_epoch = 0
            # Save initial model states
            best_model_states = {
                'pred': self.model_pred.state_dict(),
                'impu': self.model_impu.state_dict(),
                'prop': self.model_prop.state_dict(),
                'abc': self.model_abc.state_dict()
            }
            if verbose:
                print(f"Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}, eval_freq={eval_freq}")
        
        for epoch in tqdm(range(num_epoch), desc="Training Minimax", disable=not verbose):
            all_idx = np.arange(num_samples)
            np.random.shuffle(all_idx)
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                sub_x_tensor = torch.LongTensor(sub_x).to(self.device)
                sub_y_tensor = torch.Tensor(sub_y).to(self.device)
                 
                # Add bounds checking to prevent index out of range
                start_idx = G * idx * self.batch_size
                end_idx = min(G * (idx + 1) * self.batch_size, len(ul_idxs))
                
                if start_idx >= len(ul_idxs):
                    break  # No more samples available
                    
                x_all_idx = ul_idxs[start_idx:end_idx]
                
                if len(x_all_idx) == 0:
                    continue  # Skip empty batches
                    
                x_sampled = x_all[x_all_idx]
                x_sampled_tensor = torch.LongTensor(x_sampled).to(self.device)
                obs_sampled = torch.Tensor(obs[x_all_idx]).to(self.device)
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled_tensor)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled_tensor)
                
                # Use equal frequency binning like original
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                bin_sum_index = torch.nn.functional.one_hot(bin_indices.long(), num_classes=int(num_bins)).float()
                
                # Pass embeddings to ABC model, not indices
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
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x_tensor), gamma, 1.0)

                pred = self.model_pred(sub_x_tensor)
                imputation_y = self.model_impu.predict(sub_x_tensor)              

                pred_u = self.model_pred(x_sampled_tensor) 
                imputation_y1 = self.model_impu.predict(x_sampled_tensor)             

                xent_loss = F.binary_cross_entropy(pred, sub_y_tensor, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size

                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                # imputation model
                pred = self.model_pred.predict(sub_x_tensor)
                imputation_y = self.model_impu(sub_x_tensor)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y_tensor, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()
            
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print("[Minimax] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
            
            # Variables to track for callback
            current_train_auc = None
            current_test_auc = None
            
            # Calculate train AUC periodically for callback
            if progress_callback and (epoch + 1) % eval_freq == 0:
                with torch.no_grad():
                    train_pred = self.predict(x)
                    current_train_auc = roc_auc_score(y, train_pred)
            
            # Early stopping based on test AUC
            if use_early_stopping and (epoch + 1) % eval_freq == 0:
                # Evaluate on test set
                with torch.no_grad():
                    test_pred = self.predict(x_test)
                    test_auc = roc_auc_score(y_test, test_pred)
                    current_test_auc = test_auc  # Store for callback
                
                # Check if improvement
                if test_auc > best_test_auc + early_stop_min_delta:
                    best_test_auc = test_auc
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model states
                    best_model_states = {
                        'pred': self.model_pred.state_dict(),
                        'impu': self.model_impu.state_dict(),
                        'prop': self.model_prop.state_dict(),
                        'abc': self.model_abc.state_dict()
                    }
                    if verbose:
                        print(f"[Minimax] epoch:{epoch}, xent:{epoch_loss:.4f}, test_auc:{test_auc:.4f} (best)")
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"[Minimax] epoch:{epoch}, xent:{epoch_loss:.4f}, test_auc:{test_auc:.4f} (patience: {patience_counter}/{early_stop_patience})")
                
                # Check if should stop
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"[Minimax] Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} with test AUC: {best_test_auc:.4f}")
                    # Restore best model states
                    self.model_pred.load_state_dict(best_model_states['pred'])
                    self.model_impu.load_state_dict(best_model_states['impu'])
                    self.model_prop.load_state_dict(best_model_states['prop'])
                    self.model_abc.load_state_dict(best_model_states['abc'])
                    return best_epoch
            
            elif epoch % 10 == 0 and verbose:
                print("[Minimax] epoch:{}, xent:{}".format(epoch, epoch_loss))
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    epoch=epoch,
                    total_epochs=num_epoch,
                    loss=epoch_loss,
                    train_auc=current_train_auc,
                    test_auc=current_test_auc
                )
                
        # If we complete all epochs and early stopping was used, restore best model
        if use_early_stopping and best_epoch < epoch:
            if verbose:
                print(f"[Minimax] Training completed. Restoring best model from epoch {best_epoch} with test AUC: {best_test_auc:.4f}")
            self.model_pred.load_state_dict(best_model_states['pred'])
            self.model_impu.load_state_dict(best_model_states['impu'])
            self.model_prop.load_state_dict(best_model_states['prop'])
            self.model_abc.load_state_dict(best_model_states['abc'])
            
        return epoch
    
    def predict(self, x):
        x_tensor = torch.LongTensor(x).to(self.device)
        pred = self.model_pred.predict(x_tensor)
        return pred.detach().cpu().numpy()


class MF_MinimaxV2(nn.Module):
    """
    MF_MinimaxV2: Uses fixed uniform binning (dr_jl_abc style) instead of equal frequency binning
    """
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, embedding_k=4, embedding_k1=8, 
                 abc_model_name='logistic_regression', copy_model_pred=1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.embedding_k1 = embedding_k1
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop  # Same as batch_size for simplicity
        
        # Use MF from matrix_factorization_DT.py for prediction and imputation
        self.model_pred = MF(self.num_users, self.num_items, self.batch_size, embedding_k=self.embedding_k1)
        self.model_impu = MF(self.num_users, self.num_items, self.batch_size, embedding_k=self.embedding_k1)
        
        # Use logistic_regression for propensity
        self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Use logistic_regression or mlp for abc
        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Copy model weights if specified
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        
        # Move to cuda if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    # 原来lamb=0
    def _compute_IPS(self, x, num_epoch=1000, lr=0.05, lamb=1e-5, tol=1e-4, verbose=False):
        print('Stage1: computing_IPS', lr, lamb)
        
        # Generate obs from x
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=lr, weight_decay=lamb)
        
        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        last_loss = 1e9
        early_stop = 0
        stop = 5  # Default stop value

        for epoch in tqdm(range(num_epoch), desc="Computing IPS", disable=not verbose):
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                x_all_idx = ul_idxs[idx*self.batch_size_prop: (idx+1)*self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.model_prop(torch.LongTensor(x_sampled).to(self.device))

                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)
                
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print("[MinimaxV2-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    return epoch
                else:
                    early_stop += 1
                    
            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print("[MinimaxV2-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
            
        return epoch

    def fit(self, x, y, x_test=None, y_test=None, G=4, alpha=1, beta=1, theta=1, num_epoch=1000, 
            pred_lr=0.05, impu_lr=0.05, prop_lr=0.05, dis_lr=0.05,
            lamb_prop=0, lamb_pred=0, lamb_imp=0, dis_lamb=0, gamma=0.05, num_bins=10,
            tol=1e-4, verbose=True, early_stop_patience=20, early_stop_min_delta=1e-4, 
            eval_freq=10, progress_callback=None):
        """
        Train the MF_MinimaxV2 model using fixed uniform binning (dr_jl_abc style)
        
        Args: Same as MF_Minimax
        """ 
        
        print('Stage2: fitting (V2 with fixed binning)', G, alpha, beta, theta, gamma, num_bins, pred_lr, impu_lr, prop_lr, dis_lr, lamb_prop, lamb_pred, lamb_imp, dis_lamb)
        
        # Create optimizers with separate learning rates and weight decays
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=lamb_pred)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=lamb_imp)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=lamb_prop)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)
        
        
        # Generate all samples and obs
        x_all = generate_total_sample(self.num_users, self.num_items)
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        num_samples = len(x)
        total_batch = num_samples // self.batch_size
        
        last_loss = 1e9
        early_stop = 0
        stop = 5  # Default stop value
        
        # Use fixed binning like dr_jl_abc (full range including 0 and 1)
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)
        print('bin_edges (V2 fixed)', bin_edges)
        
        # Initialize early stopping variables
        use_early_stopping = x_test is not None and y_test is not None
        if use_early_stopping:
            from sklearn.metrics import roc_auc_score
            best_test_auc = -float('inf')
            patience_counter = 0
            best_epoch = 0
            # Save initial model states
            best_model_states = {
                'pred': self.model_pred.state_dict(),
                'impu': self.model_impu.state_dict(),
                'prop': self.model_prop.state_dict(),
                'abc': self.model_abc.state_dict()
            }
            if verbose:
                print(f"Early stopping enabled: patience={early_stop_patience}, min_delta={early_stop_min_delta}, eval_freq={eval_freq}")
        
        for epoch in tqdm(range(num_epoch), desc="Training MinimaxV2", disable=not verbose):
            all_idx = np.arange(num_samples)
            np.random.shuffle(all_idx)
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):  
                # data prepare
                selected_idx = all_idx[idx*self.batch_size:(idx+1)*self.batch_size]
                sub_x = x[selected_idx] 
                sub_y = y[selected_idx]
                sub_x_tensor = torch.LongTensor(sub_x).to(self.device)
                sub_y_tensor = torch.Tensor(sub_y).to(self.device)
                 
                # Add bounds checking to prevent index out of range
                start_idx = G * idx * self.batch_size
                end_idx = min(G * (idx + 1) * self.batch_size, len(ul_idxs))
                
                if start_idx >= len(ul_idxs):
                    break  # No more samples available
                    
                x_all_idx = ul_idxs[start_idx:end_idx]
                
                if len(x_all_idx) == 0:
                    continue  # Skip empty batches
                    
                x_sampled = x_all[x_all_idx]
                x_sampled_tensor = torch.LongTensor(x_sampled).to(self.device)
                obs_sampled = torch.Tensor(obs[x_all_idx]).to(self.device)
                
                # propensity model
                prop_sampled = self.model_prop(x_sampled_tensor)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled_tensor)
                
                # Use fixed binning like dr_jl_abc
                bin_indices = torch.bucketize(prop_sampled.detach(), bin_edges, right=False) - 1
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)  

                # Use dynamic num_classes like dr_jl_abc
                bin_sum_index = torch.nn.functional.one_hot(bin_indices, num_classes=bin_indices.max() + 1).float()
                
                # Pass embeddings to ABC model, not indices
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
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x_tensor), gamma, 1.0)

                pred = self.model_pred(sub_x_tensor)
                imputation_y = self.model_impu.predict(sub_x_tensor)              

                pred_u = self.model_pred(x_sampled_tensor) 
                imputation_y1 = self.model_impu.predict(x_sampled_tensor)             

                xent_loss = F.binary_cross_entropy(pred, sub_y_tensor, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')

                ips_loss = (xent_loss - imputation_loss) # batch size

                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                optimizer_prediction.step()
                
                # imputation model
                pred = self.model_pred.predict(sub_x_tensor)
                imputation_y = self.model_impu(sub_x_tensor)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y_tensor, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')

                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()
            
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-12)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print("[MinimaxV2] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
            
            # Variables to track for callback
            current_train_auc = None
            current_test_auc = None
            
            # Calculate train AUC periodically for callback
            if progress_callback and (epoch + 1) % eval_freq == 0:
                with torch.no_grad():
                    train_pred = self.predict(x)
                    current_train_auc = roc_auc_score(y, train_pred)
            
            # Early stopping based on test AUC
            if use_early_stopping and (epoch + 1) % eval_freq == 0:
                # Evaluate on test set
                with torch.no_grad():
                    test_pred = self.predict(x_test)
                    test_auc = roc_auc_score(y_test, test_pred)
                    current_test_auc = test_auc  # Store for callback
                
                # Check if improvement
                if test_auc > best_test_auc + early_stop_min_delta:
                    best_test_auc = test_auc
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model states
                    best_model_states = {
                        'pred': self.model_pred.state_dict(),
                        'impu': self.model_impu.state_dict(),
                        'prop': self.model_prop.state_dict(),
                        'abc': self.model_abc.state_dict()
                    }
                    if verbose:
                        print(f"[MinimaxV2] epoch:{epoch}, xent:{epoch_loss:.4f}, test_auc:{test_auc:.4f} (best)")
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"[MinimaxV2] epoch:{epoch}, xent:{epoch_loss:.4f}, test_auc:{test_auc:.4f} (patience: {patience_counter}/{early_stop_patience})")
                
                # Check if should stop
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"[MinimaxV2] Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch} with test AUC: {best_test_auc:.4f}")
                    # Restore best model states
                    self.model_pred.load_state_dict(best_model_states['pred'])
                    self.model_impu.load_state_dict(best_model_states['impu'])
                    self.model_prop.load_state_dict(best_model_states['prop'])
                    self.model_abc.load_state_dict(best_model_states['abc'])
                    return best_epoch
            
            elif epoch % 10 == 0 and verbose:
                print("[MinimaxV2] epoch:{}, xent:{}".format(epoch, epoch_loss))
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    epoch=epoch,
                    total_epochs=num_epoch,
                    loss=epoch_loss,
                    train_auc=current_train_auc,
                    test_auc=current_test_auc
                )
                
        # If we complete all epochs and early stopping was used, restore best model
        if use_early_stopping and best_epoch < epoch:
            if verbose:
                print(f"[MinimaxV2] Training completed. Restoring best model from epoch {best_epoch} with test AUC: {best_test_auc:.4f}")
            self.model_pred.load_state_dict(best_model_states['pred'])
            self.model_impu.load_state_dict(best_model_states['impu'])
            self.model_prop.load_state_dict(best_model_states['prop'])
            self.model_abc.load_state_dict(best_model_states['abc'])
            
        return epoch
    
    def predict(self, x):
        x_tensor = torch.LongTensor(x).to(self.device)
        pred = self.model_pred.predict(x_tensor)
        return pred.detach().cpu().numpy()


def ECELoss(scores, labels, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    predictions = scores.ge(torch.mul(torch.ones_like(scores), 0.5)).type(torch.IntTensor)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = scores.ge(bin_lower.item()) * scores.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = scores[in_bin].mean()
            if bin_upper < 0.501: # for binary classification
                avg_confidence_in_bin = 1 - avg_confidence_in_bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.cpu().data



# time
class MF_DR_JL_CE(nn.Module):
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, num_experts, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.prediction_model = MF_BaseModel(num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, *args, **kwargs)
        self.imputation_model = MF_BaseModel(num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.propensity_model = NCF_BaseModel(num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, *args, **kwargs)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        
        ## for calibration
        self.num_experts = num_experts
        self.prop_selection_net = nn.Sequential(nn.Linear(self.embedding_k, num_experts), nn.Softmax(dim=1))
        self.imp_selection_net = nn.Sequential(nn.Linear(self.embedding_k, num_experts), nn.Softmax(dim=1))

        self.a_prop = nn.Parameter(torch.FloatTensor([1 for i in range(num_experts)]))
        self.b_prop = nn.Parameter(torch.FloatTensor([-1 for i in range(num_experts)]))
        self.a_imp = nn.Parameter(torch.FloatTensor([1 for i in range(num_experts)]))
        self.b_imp = nn.Parameter(torch.FloatTensor([-1 for i in range(num_experts)]))
        
        self.sm = nn.Softmax(dim = 1)
        
        # Move all models to device
        self.to(self.device)
     
    def calibration_experts(self, x, T, mode='prop'):
        # get emb
        if mode == 'prop':
            u_emb, _ = self.propensity_model.get_emb(x)
            logit = self.propensity_model.forward_logit(x)
        else:
            u_emb, _ = self.imputation_model.get_emb(x)
            logit = self.imputation_model.forward_logit(x)
        
        # get selection dist (Gumbel softmax)
        if mode == 'prop':
            selection_dist = self.prop_selection_net(u_emb) # (batch_size, num_experts)
        else:
            selection_dist = self.imp_selection_net(u_emb)
        
        g = torch.distributions.Gumbel(torch.tensor(0.0).to(self.device), torch.tensor(1.0).to(self.device)).sample(selection_dist.size())
        eps = torch.tensor(1e-10).to(self.device) # for numerical stability
        selection_dist = selection_dist + eps
        selection_dist = self.sm((selection_dist.log() + g) / T) # (batch_size, num_experts) (row sum to 1)
        
        # calibration experts
        logits = torch.unsqueeze(logit, 1) # (batch_size, 1)
        logits = logits.repeat(1, self.num_experts) # (batch_size, num_experts)

        if mode == 'prop':
            expert_outputs = self.sigmoid(logits * self.a_prop + self.b_prop) # (batch_size, num_experts)
        else:
            expert_outputs = self.sigmoid(logits * self.a_imp + self.b_imp)
        
        expert_outputs = expert_outputs * selection_dist # (batch_size, num_experts)
        expert_outputs = expert_outputs.sum(1) # (batch_size, )
        
        # [0, 1]
        expert_outputs = expert_outputs - torch.lt(expert_outputs, 0) * expert_outputs
        expert_outputs = expert_outputs - torch.gt(expert_outputs, 1) * (expert_outputs - torch.ones_like(expert_outputs).to(self.device))
        
        return expert_outputs
        
    def _compute_IPS(self, x, num_epoch=200, lr=0.05, lamb=0, tol=1e-4, verbose=False):
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        optimizer_propensity = torch.optim.Adam(self.propensity_model.parameters(), lr=lr, weight_decay=lamb)
        
        last_loss = 1e9
        
        num_sample = len(obs) ## 전체 pair 개수 |D|
        total_batch = num_sample // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        early_stop = 0
        
        pbar = tqdm(range(num_epoch), desc="[MF-DRJL-CE-PS] Computing IPS", disable=not verbose)
        for epoch in pbar:
            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                
                x_sampled = x_all[x_all_idx]
                prop = self.propensity_model.forward(x_sampled)
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)

                prop_loss = nn.MSELoss()(prop, sub_obs)
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            pbar.set_postfix({'loss': epoch_loss})
            
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    if verbose:
                        print("\n[MF-DRJL-CE-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if epoch == num_epoch - 1:
                print("[MF-DRJL-CE-PS] Reach preset epochs, it seems does not converge.") 

    def _calibrate_IPS_G(self, x_val, x_test, num_epoch=100, lr=0.01, lamb=0, end_T=1e-3, verbose=False, G=10):
        x_all = generate_total_sample(self.num_users, self.num_items) # all (u,i) pairs = D
        obs = sps.csr_matrix((np.ones(x_val.shape[0]), (x_val[:, 0], x_val[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)

        t0 = time.time()
        self.propensity_model.eval()
        
        ## data prep
        ul_idxs = np.arange(x_all.shape[0]) # idxs
        np.random.shuffle(ul_idxs)
        neg_idxs = ul_idxs[:len(x_val) * G]
        
        ui_idxs = np.concatenate((x_all[neg_idxs], x_val), axis=0) # (u,i) pairs
        sub_obs = torch.FloatTensor(np.concatenate((obs[neg_idxs], np.ones(len(x_val))), axis=0)).to(self.device) ## y (label)
        
        ## fit CE - Adam
        optimizer = torch.optim.Adam([self.a_prop, self.b_prop] + list(self.prop_selection_net.parameters()), lr=lr, weight_decay=lamb)

        total_batch = len(ui_idxs) // self.batch_size_prop
        current_T = torch.tensor(1.).to(self.device)
        for epoch in range(num_epoch):
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                batch_idx = ui_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                batch_y = sub_obs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]      
                
                prop = self.calibration_experts(batch_idx, current_T, mode='prop')
                prop_loss = F.binary_cross_entropy(prop, batch_y)

                optimizer.zero_grad()
                prop_loss.backward()
                optimizer.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()

            current_T = torch.tensor(1. * ((end_T / 1.) ** (epoch / num_epoch))).to(self.device)
            
            if verbose:
                if epoch % 10 == 0:    
                    print("epoch:", epoch, "loss:", epoch_loss)
        
        if verbose:
            print("calibraton done in", time.time() - t0)
            
            ## test ECE
            obs = sps.csr_matrix((np.ones(x_test.shape[0]), (x_test[:, 0], x_test[:, 1])), shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)

            ul_idxs = np.arange(x_all.shape[0]) # idxs
            np.random.shuffle(ul_idxs)
            neg_idxs = ul_idxs[:len(x_test) * G]
            
            ui_idxs = np.concatenate((x_all[neg_idxs], x_test), axis=0) # (u,i) pairs            
            sub_obs = np.concatenate((obs[neg_idxs], np.ones(len(x_test))), axis=0)

            self.a_prop.requires_grad = False
            self.b_prop.requires_grad = False
            for param in self.prop_selection_net.parameters():
                param.requires_grad = False
            scores_uncal = self.propensity_model.forward(ui_idxs).detach().cpu()
            scores_cal = self.calibration_experts(ui_idxs, current_T, mode='prop').detach().cpu()
            
            print(scores_uncal.mean(), scores_uncal.std())
            print(scores_uncal)
            print(scores_cal.mean(), scores_cal.std())
            print(scores_cal)
            
            ECE_uncal = ECELoss(scores_uncal, torch.LongTensor(sub_obs))
            ECE_cal = ECELoss(scores_cal, torch.LongTensor(sub_obs))
            print("test ECE:", ECE_uncal, ECE_cal)

    def fit(self, x, y, x_val, y_val, stop = 5, num_epoch=1000, lr=0.05, lamb=0, gamma = 0.1, tol=1e-4, G=1, end_T=1e-3, lr_imp=0.05, lamb_imp=0, lr_impcal=0.05, lamb_impcal=0, iter_impcal=10, verbose=False, cal=True): 
        optimizer_prediction = torch.optim.Adam(self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(self.imputation_model.parameters(), lr=lr_imp, weight_decay=lamb_imp)
        optimizer_impcal = torch.optim.Adam([self.a_imp, self.b_imp] + list(self.imp_selection_net.parameters()), lr=lr_impcal, weight_decay=lamb_impcal)
        self.propensity_model.eval()
        
        x_all = generate_total_sample(self.num_users, self.num_items) ## D
        num_sample = len(x) # O
        total_batch = num_sample // self.batch_size

        early_stop = 0
        last_loss = 1e9
        current_T = torch.tensor(1.).to(self.device)
        
        pbar = tqdm(range(num_epoch), desc="[MF-DR-JL-CE] Training", disable=not verbose)
        for epoch in pbar: 
            # O
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # D
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0
            for idx in range(total_batch):
                ## prediction model update
                self.prediction_model.train()
                self.imputation_model.eval()
                # O part (if o_ui=1)
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(self.device)
                
                inv_prop = 1/torch.clip(self.calibration_experts(sub_x, 1e-3, mode='prop').detach(), gamma, 1)
                
                pred = self.prediction_model.forward(sub_x)

                imputation_y = self.calibration_experts(sub_x, current_T, mode='imp')
                              
                xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # e/p
                imputation_loss = F.binary_cross_entropy(pred, torch.clip(imputation_y,0,1), reduction="sum") # e^
                ips_loss = (xent_loss - imputation_loss) # batch size, e/p - e^ (current) <<<<===>>>> e/p - e^/p + e^ (paper)

                # D part (if o_ui=0)
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]] # negative ratio=G
                
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.calibration_experts(x_sampled, current_T, mode='imp')
                direct_loss = F.binary_cross_entropy(pred_u, torch.clip(imputation_y1,0,1), reduction="sum")
    
                # total loss
                loss = ips_loss/self.batch_size + direct_loss/self.batch_size #/G

                optimizer_prediction.zero_grad()
                loss.backward()
                optimizer_prediction.step()
                                                           
                epoch_loss += xent_loss.detach().cpu().numpy()              
                
                ## imputation model update (O)
                self.prediction_model.eval()
                self.imputation_model.train()
                pred = self.prediction_model.predict(sub_x).to(self.device) ## prediction: y_hat
                imputation_y = self.imputation_model.forward(sub_x) ## pseudo label: y_tilde
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none") ## actual loss: e
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none") ## imputed loss: e_hat
                imp_loss = (((e_loss.detach() - e_hat_loss) ** 2) * inv_prop).sum() ## error deviation: (e - e_hat)^2 / p  -> loss function for imputation model

                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()

            ## imputation model calibration (O) + Propensity (하는게 성능 더 좋긴해 in coat)
            if cal:
                self.imputation_model.eval()
                inv_prop = 1/torch.clip(self.calibration_experts(x_val, 1e-3, mode='prop').detach(), gamma, 1)
                
                for i in range(iter_impcal):
                    prop = self.calibration_experts(x_val, current_T, mode='imp')
                    prop_loss = F.binary_cross_entropy(prop, torch.FloatTensor(y_val).to(self.device), weight=inv_prop, reduction="sum")

                    optimizer_impcal.zero_grad()
                    prop_loss.backward()
                    optimizer_impcal.step()
                    
                current_T = torch.tensor(1. * ((end_T / 1.) ** (epoch / num_epoch))).to(self.device)

            pbar.set_postfix({'loss': epoch_loss})
            
            ## early stop
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print("\n[MF-DR-JL-CE] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1

            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL-CE] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL-CE] Reach preset epochs, it seems does not converge.")
                
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()


class MF_DRv2_BMSE_Imp(nn.Module):
    """
    Matrix Factorization with DR + BMSE from DRv2.py with imputation model.
    Uses same MF architecture for all three models (base, imputation, propensity).
    Joint training without separate propensity pre-training.
    """
    def __init__(self, num_users, num_items, batch_size, batch_size_prop,
                 embedding_k=256, embedding_k1=256, embedding_k_prop=256,
                 *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        
        # All models use MF architecture
        self.base_model = MF(num_users, num_items, batch_size, embedding_k=embedding_k1)
        self.imputation_model = MF(num_users, num_items, batch_size, embedding_k=embedding_k1)
        self.propensity_model = MF(num_users, num_items, batch_size, embedding_k=embedding_k_prop)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
        self.imputation_model.to(self.device)
        self.propensity_model.to(self.device)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def fit(self, x, y, num_epoch=500, lr=0.005, impu_lr=0.005, prop_lr=0.005,
            lamb=1e-5, lamb_imp=1e-6, lamb_prop=1e-1,
            alpha=1, beta=2, gamma=0.1, imputation=1e-3,
            tol=1e-4, early_stop_rounds=10, verbose=True):
        """
        Fit the model using joint training approach from DRv2.py.
        
        Parameters match DRv2.py loss_args:
        - alpha: weight for ctcvr_loss
        - beta: weight for cvr_loss_mnar
        - gamma: weight for bmse_loss
        - imputation: weight for loss_imp
        """
        
        # Create optimizers
        base_optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=lr, weight_decay=lamb)
        imputation_optimizer = torch.optim.Adam(
            self.imputation_model.parameters(), lr=impu_lr, weight_decay=lamb_imp)
        propensity_optimizer = torch.optim.Adam(
            self.propensity_model.parameters(), lr=prop_lr, weight_decay=lamb_prop)
        
        # Loss functions
        none_criterion = nn.MSELoss(reduction='none')
        mean_criterion = nn.MSELoss()
        
        # Create matrices for batch processing (following DRv2.py approach)
        x_tensor = torch.LongTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create observed matrix
        obs_matrix = torch.zeros(self.num_users, self.num_items).to(self.device)
        obs_matrix[x[:, 0], x[:, 1]] = 1
        
        # Create rating matrix
        rating_matrix = torch.zeros(self.num_users, self.num_items).to(self.device)
        rating_matrix[x[:, 0], x[:, 1]] = y_tensor
        
        num_sample = len(x)
        total_batch = num_sample // self.batch_size
        
        last_loss = 1e9
        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            epoch_loss = 0
            
            for idx in range(total_batch):
                # Get batch
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                # Get all user-item pairs in the batch block
                batch_users = np.unique(sub_x[:, 0])
                batch_items = np.unique(sub_x[:, 1])
                
                # Create all pairs
                users_all, items_all = np.meshgrid(batch_users, batch_items, indexing='ij')
                users_all = users_all.flatten()
                items_all = items_all.flatten()
                
                # Get sub-matrices
                sub_obs = obs_matrix[batch_users][:, batch_items].flatten()
                sub_ratings = rating_matrix[batch_users][:, batch_items].flatten()
                
                # Convert to tensors
                users_all = torch.LongTensor(users_all).to(self.device)
                items_all = torch.LongTensor(items_all).to(self.device)
                sub_obs = sub_obs.to(self.device)
                sub_ratings = sub_ratings.to(self.device)
                
                # Set models to train mode
                self.base_model.train()
                self.imputation_model.train()
                self.propensity_model.train()
                
                # Forward pass
                p_hat = torch.sigmoid(self.propensity_model.forward(
                    torch.stack([users_all, items_all], dim=1)))
                r_hat = torch.sigmoid(self.base_model.forward(
                    torch.stack([users_all, items_all], dim=1)))
                
                # Compute losses (following DRv2.py exactly)
                e_true = none_criterion(r_hat, sub_ratings)
                r_tilde = self.imputation_model.forward(
                    torch.stack([users_all, items_all], dim=1))
                e_hat = none_criterion(r_hat, torch.sigmoid(r_tilde))
                cost_e = none_criterion(e_hat, e_true)
                loss_imp = torch.mean(torch.multiply(sub_obs, torch.divide(cost_e, p_hat)))
                
                ctr_loss = mean_criterion(p_hat, sub_obs)
                ctcvr_loss = mean_criterion(
                    torch.multiply(p_hat, r_hat), 
                    torch.multiply(sub_obs, sub_ratings))
                cvr_loss_mnar = torch.mean(torch.add(
                    e_hat, 
                    torch.divide(torch.multiply(sub_obs, e_true - e_hat), p_hat)))
                
                # BMSE loss (exact implementation from DRv2.py)
                ones_all = torch.ones(len(p_hat)).to(self.device)
                w_all = torch.divide(sub_obs, p_hat) - torch.divide(
                    (ones_all - sub_obs), (ones_all - p_hat))
                bmse_loss = (torch.mean(w_all * r_hat)) ** 2
                
                # Total loss
                loss = (ctr_loss + 
                        alpha * ctcvr_loss + 
                        beta * cvr_loss_mnar + 
                        imputation * loss_imp + 
                        gamma * bmse_loss)
                
                # Backward and optimize
                base_optimizer.zero_grad()
                imputation_optimizer.zero_grad()
                propensity_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()
                imputation_optimizer.step()
                propensity_optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
            
            # Check convergence
            relative_loss_div = abs(last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop >= early_stop_rounds:
                    if verbose:
                        print(f"\n[MF-DRv2-BMSE-Imp] Early stop at epoch {epoch}, loss: {epoch_loss}")
                    break
                else:
                    early_stop += 1
            else:
                early_stop = 0
                
            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print(f"[MF-DRv2-BMSE-Imp] epoch: {epoch}, loss: {epoch_loss}")
                
    def predict(self, x):
        self.base_model.eval()
        with torch.no_grad():
            x = torch.LongTensor(x).to(self.device)
            pred = torch.sigmoid(self.base_model.forward(x))
        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self, x, *args, **kwargs):
        """No separate IPS computation - joint training instead"""
        pass


class MF_DRv2_BMSE(nn.Module):
    """
    Matrix Factorization with DR + BMSE from DRv2.py without imputation model.
    Uses same MF architecture for both models (base, propensity).
    Joint training without separate propensity pre-training.
    """
    def __init__(self, num_users, num_items, batch_size, batch_size_prop,
                 embedding_k=256, embedding_k1=256, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        
        # Both models use MF architecture
        self.base_model = MF(num_users, num_items, batch_size, embedding_k=embedding_k1)
        self.propensity_model = MF(num_users, num_items, batch_size, embedding_k=embedding_k)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
        self.propensity_model.to(self.device)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def fit(self, x, y, num_epoch=500, lr=0.01, prop_lr=0.01,
            lamb=1e-5, lamb_prop=1e-1,
            alpha=1, beta=1, gamma=0.01,
            tol=1e-4, early_stop_rounds=10, verbose=True):
        """
        Fit the model using joint training approach from DRv2.py (no imputation version).
        
        Parameters match DRv2.py loss_args:
        - alpha: weight for ctcvr_loss
        - beta: weight for cvr_loss_mnar
        - gamma: weight for bmse_loss
        """
        
        # Create optimizers
        base_optimizer = torch.optim.Adam(
            self.base_model.parameters(), lr=lr, weight_decay=lamb)
        propensity_optimizer = torch.optim.Adam(
            self.propensity_model.parameters(), lr=prop_lr, weight_decay=lamb_prop)
        
        # Loss functions
        none_criterion = nn.MSELoss(reduction='none')
        mean_criterion = nn.MSELoss()
        
        # Create matrices for batch processing (following DRv2.py approach)
        x_tensor = torch.LongTensor(x).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create observed matrix
        obs_matrix = torch.zeros(self.num_users, self.num_items).to(self.device)
        obs_matrix[x[:, 0], x[:, 1]] = 1
        
        # Create rating matrix
        rating_matrix = torch.zeros(self.num_users, self.num_items).to(self.device)
        rating_matrix[x[:, 0], x[:, 1]] = y_tensor
        
        num_sample = len(x)
        total_batch = num_sample // self.batch_size
        
        last_loss = 1e9
        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            
            epoch_loss = 0
            
            for idx in range(total_batch):
                # Get batch
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                
                # Get all user-item pairs in the batch block
                batch_users = np.unique(sub_x[:, 0])
                batch_items = np.unique(sub_x[:, 1])
                
                # Create all pairs
                users_all, items_all = np.meshgrid(batch_users, batch_items, indexing='ij')
                users_all = users_all.flatten()
                items_all = items_all.flatten()
                
                # Get sub-matrices
                sub_obs = obs_matrix[batch_users][:, batch_items].flatten()
                sub_ratings = rating_matrix[batch_users][:, batch_items].flatten()
                
                # Convert to tensors
                users_all = torch.LongTensor(users_all).to(self.device)
                items_all = torch.LongTensor(items_all).to(self.device)
                sub_obs = sub_obs.to(self.device)
                sub_ratings = sub_ratings.to(self.device)
                
                # Set models to train mode
                self.base_model.train()
                self.propensity_model.train()
                
                # Forward pass
                p_hat = torch.sigmoid(self.propensity_model.forward(
                    torch.stack([users_all, items_all], dim=1)))
                r_hat = torch.sigmoid(self.base_model.forward(
                    torch.stack([users_all, items_all], dim=1)))
                
                # Compute losses (following DRv2.py exactly - no imputation version)
                ctr_loss = mean_criterion(p_hat, sub_obs)
                ctcvr_loss = mean_criterion(
                    torch.multiply(p_hat, r_hat), 
                    torch.multiply(sub_obs, sub_ratings))
                cvr_loss_mnar = none_criterion(r_hat, sub_ratings)
                cvr_loss_mnar = torch.mean(torch.divide(
                    torch.multiply(sub_obs, cvr_loss_mnar), p_hat))
                
                # BMSE loss (exact implementation from DRv2.py)
                ones_all = torch.ones(len(p_hat)).to(self.device)
                w_all = torch.divide(sub_obs, p_hat) - torch.divide(
                    (ones_all - sub_obs), (ones_all - p_hat))
                bmse_loss = (torch.mean(w_all * r_hat)) ** 2
                
                # Total loss
                loss = (ctr_loss + 
                        alpha * ctcvr_loss + 
                        beta * cvr_loss_mnar + 
                        gamma * bmse_loss)
                
                # Backward and optimize
                base_optimizer.zero_grad()
                propensity_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()
                propensity_optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
            
            # Check convergence
            relative_loss_div = abs(last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop >= early_stop_rounds:
                    if verbose:
                        print(f"\n[MF-DRv2-BMSE] Early stop at epoch {epoch}, loss: {epoch_loss}")
                    break
                else:
                    early_stop += 1
            else:
                early_stop = 0
                
            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print(f"[MF-DRv2-BMSE] epoch: {epoch}, loss: {epoch_loss}")
                
    def predict(self, x):
        self.base_model.eval()
        with torch.no_grad():
            x = torch.LongTensor(x).to(self.device)
            pred = torch.sigmoid(self.base_model.forward(x))
        return pred.detach().cpu().numpy()
    
    def _compute_IPS(self, x, *args, **kwargs):
        """No separate IPS computation - joint training instead"""
        pass
    


# ============================================================================
# Enhanced Models for MF_MinimaxV3
# ============================================================================

class MF_Enhanced(nn.Module):
    """Enhanced Matrix Factorization with BatchNorm, Dropout, and Bias terms"""
    
    def __init__(self, num_users, num_items, batch_size, embedding_k=4, dropout_rate=0.2, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        
        # User and item embeddings with better initialization
        self.W = nn.Embedding(self.num_users, self.embedding_k)
        self.H = nn.Embedding(self.num_items, self.embedding_k)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.H.weight)
        
        # Bias terms
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        # Batch normalization (use 1D for embeddings)
        self.bn_user = nn.BatchNorm1d(self.embedding_k)
        self.bn_item = nn.BatchNorm1d(self.embedding_k)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x, is_training=False):
        # Handle both numpy arrays and tensors
        if isinstance(x, np.ndarray):
            user_idx = torch.LongTensor(x[:, 0]).to(self.device)
            item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        else:  # x is already a tensor
            user_idx = x[:, 0].long().to(self.device)
            item_idx = x[:, 1].long().to(self.device)
        
        # Get embeddings
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        # Apply batch normalization (only during training)
        if self.training and U_emb.size(0) > 1:  # Batch norm needs batch_size > 1
            U_emb = self.bn_user(U_emb)
            V_emb = self.bn_item(V_emb)
        
        # Apply dropout
        U_emb = self.dropout(U_emb)
        V_emb = self.dropout(V_emb)
        
        # Get biases
        u_bias = self.user_bias(user_idx)
        i_bias = self.item_bias(item_idx)
        
        # Compute prediction: dot product + biases
        out = torch.sum(torch.mul(U_emb, V_emb), 1, keepdim=True) + u_bias + i_bias + self.global_bias
        #out = torch.sum(torch.mul(U_emb, V_emb), 1, keepdim=True)
        
        if is_training:
            return torch.squeeze(self.sigmoid(out)), U_emb, V_emb
        else:
            return torch.squeeze(self.sigmoid(out))
    
    def predict(self, x):
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            pred = self.forward(x)
        self.train()  # Back to training mode
        return pred.detach().cpu()


class MLP_Enhanced(nn.Module):
    """Enhanced MLP for discriminator with deeper architecture"""
    
    def __init__(self, num_users, num_items, embedding_k=4, hidden_dims=[64, 32], dropout_rate=0.2, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.dropout_rate = dropout_rate
        
        # Embeddings with initialization
        self.W = nn.Embedding(self.num_users, self.embedding_k)
        self.H = nn.Embedding(self.num_items, self.embedding_k)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.H.weight)
        
        # Build MLP layers
        layers = []
        in_dim = embedding_k * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        # Handle both numpy arrays and tensors
        if isinstance(x, np.ndarray):
            user_idx = torch.LongTensor(x[:, 0]).to(self.device)
            item_idx = torch.LongTensor(x[:, 1]).to(self.device)
        else:  # x is already a tensor
            user_idx = x[:, 0].long().to(self.device)
            item_idx = x[:, 1].long().to(self.device)
        
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        
        # Concatenate embeddings
        z_emb = torch.cat([U_emb, V_emb], dim=1)
        
        # Pass through MLP
        out = self.mlp(z_emb)
        
        return torch.squeeze(self.sigmoid(out))
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred = self.forward(x)
        self.train()
        return pred.detach().cpu()


class MF_MinimaxV3(nn.Module):
    """
    MF_MinimaxV3: Enhanced version with architecture improvements and better training
    """
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, 
                 embedding_k=4, embedding_k1=8, dropout_rate=0.2,
                 abc_model_name='mlp_enhanced', copy_model_pred=1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.embedding_k1 = embedding_k1
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        self.dropout_rate = dropout_rate
        
        # Use enhanced MF for prediction and imputation
        self.model_pred = MF_Enhanced(self.num_users, self.num_items, self.batch_size, 
                                      embedding_k=self.embedding_k1, dropout_rate=dropout_rate)
        self.model_impu = MF_Enhanced(self.num_users, self.num_items, self.batch_size, 
                                      embedding_k=self.embedding_k1, dropout_rate=dropout_rate)
        
        # Use standard logistic regression for propensity (keep it simple)
        self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Use enhanced MLP or standard logistic regression for discriminator
        if abc_model_name == 'mlp_enhanced':
            self.model_abc = MLP_Enhanced(self.num_users, self.num_items, 
                                          embedding_k=self.embedding_k, dropout_rate=dropout_rate)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        else:
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Copy model weights if specified
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        
        # Move to cuda if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
    
    def _compute_IPS(self, x, num_epoch=500, lr=0.05, lamb=1e-5, tol=1e-4, verbose=False):
        """Pre-train propensity model with extended epochs"""
        print('Stage1: computing_IPS (V3 with extended training)', lr, lamb)
        
        # Generate obs from x
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=lr, weight_decay=lamb)
        # Add learning rate scheduler for propensity
        scheduler_prop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_propensity, mode='min', 
                                                                    factor=0.5, patience=30)
        
        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        last_loss = 1e9
        early_stop = 0
        
        for epoch in tqdm(range(num_epoch), desc='Stage1: computing_IPS'):
            # Sampling all samples
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0
            
            for idx in range(total_batch):
                # Mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]
                
                # Propensity prediction
                prop = self.model_prop.forward(x_sampled)
                
                # Get observed labels
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)
                
                # MSE loss for propensity
                prop_loss = nn.MSELoss()(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_prop.parameters(), max_norm=1.0)
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()
            
            # Learning rate scheduling
            scheduler_prop.step(epoch_loss)
            
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 10:  # Increased early stop patience
                    if verbose:
                        print('[Stage1] Early stop at epoch {}, loss: {}'.format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
            else:
                early_stop = 0
                
            last_loss = epoch_loss
            
            if epoch % 50 == 0 and verbose:
                print('[Stage1] epoch: {}, loss: {}'.format(epoch, epoch_loss))
    
    def fit(self, x, y, x_test=None, y_test=None, G=4, alpha=1, beta=1, theta=1, num_epoch=1000, 
            pred_lr=0.05, impu_lr=0.05, prop_lr=0.05, dis_lr=0.05,
            lamb_prop=0, lamb_pred=0, lamb_imp=0, dis_lamb=0, gamma=0.05, num_bins=10,
            tol=1e-4, verbose=True, early_stop_patience=30, early_stop_min_delta=1e-4, 
            eval_freq=5, progress_callback=None, grad_clip_norm=1.0):
        """
        Enhanced training with learning rate scheduling and gradient clipping
        """
        
        print('Stage2: fitting (V3 with enhanced training)', G, alpha, beta, theta, gamma, num_bins, 
              pred_lr, impu_lr, prop_lr, dis_lr, lamb_prop, lamb_pred, lamb_imp, dis_lamb)
        
        # Create optimizers
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=lamb_pred)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=lamb_imp)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=lamb_prop)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)
        
        # Learning rate schedulers
        scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_prediction, T_max=num_epoch, eta_min=pred_lr*0.1)
        scheduler_impu = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_imputation, T_max=num_epoch, eta_min=impu_lr*0.1)
        scheduler_prop = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_propensity, T_max=num_epoch, eta_min=prop_lr*0.1)
        scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=num_epoch, eta_min=dis_lr*0.1)
        
        # Generate all samples and obs
        x_all = generate_total_sample(self.num_users, self.num_items)
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        num_samples = len(x)
        total_batch = num_samples // self.batch_size
        
        last_loss = 1e9
        early_stop = 0
        stop = 5  # Default stop value
        
        # Use adaptive binning like original MF_Minimax
        bin_edges = torch.linspace(0, 1, steps=int(num_bins) + 1, device=self.device)[1:-1]
        print('bin_edges (V3)', bin_edges)
        
        # Initialize early stopping variables
        use_early_stopping = x_test is not None and y_test is not None
        if use_early_stopping:
            from sklearn.metrics import roc_auc_score
            best_test_auc = -float('inf')
            patience_counter = 0
            best_epoch = 0
            # Save initial model states
            best_model_states = {
                'pred': copy.deepcopy(self.model_pred.state_dict()),
                'impu': copy.deepcopy(self.model_impu.state_dict()),
                'prop': copy.deepcopy(self.model_prop.state_dict()),
                'abc': copy.deepcopy(self.model_abc.state_dict())
            }
            if verbose:
                print("Early stopping enabled with patience =", early_stop_patience)
        
        # Training loop
        for epoch in tqdm(range(num_epoch), desc='Stage2: fitting'):
            # Set models to training mode
            self.model_pred.train()
            self.model_impu.train()
            self.model_prop.train()
            self.model_abc.train()
            
            # Shuffle data
            all_idx = np.arange(num_samples)
            np.random.shuffle(all_idx)
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0
            
            for idx in range(total_batch):
                # Get batch data
                batch_idx = all_idx[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_x = x[batch_idx]
                batch_y = y[batch_idx]
                batch_y_tensor = torch.Tensor(batch_y).to(self.device)
                
                # Get counterfactual samples for propensity/discriminator training
                x_sampled = x_all[ul_idxs[G * idx * self.batch_size : G * (idx + 1) * self.batch_size]]
                x_sampled_tensor = torch.LongTensor(x_sampled).to(self.device)
                obs_sampled = torch.Tensor(obs[ul_idxs[G * idx * self.batch_size : G * (idx + 1) * self.batch_size]]).to(self.device)
                
                # Propensity model on counterfactual samples
                prop_sampled = self.model_prop(x_sampled_tensor)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled_tensor)
                
                # Use equal frequency binning (adaptive binning)
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
                bin_sum_index = torch.nn.functional.one_hot(bin_indices.long(), num_classes=int(num_bins)).float()
                
                # Discriminator training
                prop_error_dis = self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()) * (obs_sampled - prop_sampled.detach())
                bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                
                optimizer_dis.zero_grad()
                prop_abc_loss_dis.backward()
                torch.nn.utils.clip_grad_norm_(self.model_abc.parameters(), max_norm=grad_clip_norm)
                optimizer_dis.step()
                
                # Propensity training
                prop_error_prop = self.model_abc.predict(prop_user_emb.detach(), prop_item_emb.detach()).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_prop.parameters(), max_norm=grad_clip_norm)
                optimizer_propensity.step()
                
                # Now train prediction and imputation models on observed samples
                # Predictions
                pred, _, _ = self.model_pred.forward(batch_x, is_training=True)
                imputation, _, _ = self.model_impu.forward(batch_x, is_training=True)
                
                # Get inverse propensity weights for observed samples
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(torch.LongTensor(batch_x).to(self.device)), gamma, 1.0)
                
                # Get predictions for counterfactual samples
                pred_ul, _, _ = self.model_pred.forward(x_sampled, is_training=True)
                imputation_ul, _, _ = self.model_impu.forward(x_sampled, is_training=True)
                
                # Compute DR losses for prediction model
                xent_loss = F.binary_cross_entropy(pred, batch_y_tensor, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation.detach(), reduction='sum')
                ips_loss = (xent_loss - imputation_loss)  # batch size
                direct_loss = F.binary_cross_entropy(pred_ul, imputation_ul.detach(), reduction='sum')
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_pred.parameters(), max_norm=grad_clip_norm)
                optimizer_prediction.step()
                
                # Train imputation model
                pred_detached = self.model_pred.predict(torch.LongTensor(batch_x).to(self.device))
                imputation_y = self.model_impu(torch.LongTensor(batch_x).to(self.device))
                
                # Ensure pred_detached is on the same device
                pred_detached = pred_detached.to(self.device)
                
                e_loss = F.binary_cross_entropy(pred_detached.detach(), batch_y_tensor, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred_detached.detach(), reduction='none')
                
                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_impu.parameters(), max_norm=grad_clip_norm)
                optimizer_imputation.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()
            
            # Learning rate scheduling
            scheduler_pred.step()
            scheduler_impu.step()
            scheduler_prop.step()
            scheduler_dis.step()
            
            # Evaluation and early stopping
            if use_early_stopping and epoch % eval_freq == 0:
                self.model_pred.eval()
                self.model_impu.eval()
                with torch.no_grad():
                    # Compute test AUC
                    test_pred = self.predict(x_test)
                    test_auc = roc_auc_score(y_test, test_pred)
                    
                    # Compute train AUC on a sample
                    train_sample_size = min(10000, len(x))
                    train_idx = np.random.choice(len(x), train_sample_size, replace=False)
                    train_pred = self.predict(x[train_idx])
                    train_auc = roc_auc_score(y[train_idx], train_pred)
                
                # Progress callback
                if progress_callback:
                    progress_callback(epoch, num_epoch, epoch_loss, train_auc, test_auc)
                
                # Check for improvement
                if test_auc > best_test_auc + early_stop_min_delta:
                    best_test_auc = test_auc
                    patience_counter = 0
                    best_epoch = epoch
                    # Save best model states
                    best_model_states = {
                        'pred': copy.deepcopy(self.model_pred.state_dict()),
                        'impu': copy.deepcopy(self.model_impu.state_dict()),
                        'prop': copy.deepcopy(self.model_prop.state_dict()),
                        'abc': copy.deepcopy(self.model_abc.state_dict())
                    }
                    if verbose:
                        print(f'\n[Early Stop] New best test AUC: {test_auc:.6f} at epoch {epoch}')
                else:
                    patience_counter += 1
                
                # # Check overfitting
                # if train_auc - test_auc > 0.15:
                #     if verbose:
                #         print(f'\n[Early Stop] Severe overfitting detected: train_auc={train_auc:.4f}, test_auc={test_auc:.4f}')
                #     break
                
                # Check patience
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f'\n[Early Stop] Patience exhausted. Best test AUC: {best_test_auc:.6f} at epoch {best_epoch}')
                    # Restore best model
                    self.model_pred.load_state_dict(best_model_states['pred'])
                    self.model_impu.load_state_dict(best_model_states['impu'])
                    self.model_prop.load_state_dict(best_model_states['prop'])
                    self.model_abc.load_state_dict(best_model_states['abc'])
                    break
            
            # Regular convergence check
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print('[Training] Early stop at epoch {}, loss: {}'.format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
            else:
                early_stop = 0
            
            last_loss = epoch_loss
            
            if verbose and epoch % 50 == 0:
                print('[Training] epoch: {}, loss: {:.4f}'.format(epoch, epoch_loss))
                if use_early_stopping:
                    print(f'[Training] Current LRs - pred: {scheduler_pred.get_last_lr()[0]:.6f}, ' + 
                          f'impu: {scheduler_impu.get_last_lr()[0]:.6f}, ' +
                          f'prop: {scheduler_prop.get_last_lr()[0]:.6f}, ' +
                          f'dis: {scheduler_dis.get_last_lr()[0]:.6f}')
        
        if use_early_stopping and verbose:
            print(f'\nTraining completed. Best test AUC: {best_test_auc:.6f} achieved at epoch {best_epoch}')
    
    def predict(self, x):
        self.model_pred.eval()
        with torch.no_grad():
            x_tensor = torch.LongTensor(x).to(self.device)
            pred = self.model_pred.predict(x_tensor)
        self.model_pred.train()
        return pred.detach().cpu().numpy()


class MF_MinimaxV4(nn.Module):
    """
    MF_MinimaxV4: Uses standard models (like V1) with V3's training improvements
    - Standard MF for prediction/imputation (no BatchNorm/Dropout)
    - Standard logistic_regression or mlp for discriminator
    - Keeps V3's training enhancements: LR scheduling, gradient clipping, better early stopping
    """
    def __init__(self, num_users, num_items, batch_size, batch_size_prop, 
                 embedding_k=4, embedding_k1=8,
                 abc_model_name='logistic_regression', copy_model_pred=1, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.embedding_k1 = embedding_k1
        self.batch_size = batch_size
        self.batch_size_prop = batch_size_prop
        
        # Use standard MF for prediction and imputation (like V1)
        self.model_pred = MF(self.num_users, self.num_items, self.batch_size, embedding_k=self.embedding_k1)
        self.model_impu = MF(self.num_users, self.num_items, self.batch_size, embedding_k=self.embedding_k1)
        
        # Use standard logistic regression for propensity
        self.model_prop = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Use standard logistic_regression or mlp for discriminator (like V1)
        if abc_model_name == 'logistic_regression':
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        elif abc_model_name == 'mlp':
            self.model_abc = mlp(self.num_users, self.num_items, embedding_k=self.embedding_k)
        else:
            # Default to logistic regression if unknown
            self.model_abc = logistic_regression(self.num_users, self.num_items, embedding_k=self.embedding_k)
        
        # Copy model weights if specified
        if copy_model_pred == 1:
            self.model_impu.load_state_dict(self.model_pred.state_dict())
        
        # Move to cuda if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_pred.to(self.device)
        self.model_impu.to(self.device)
        self.model_prop.to(self.device)
        self.model_abc.to(self.device)
        
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
    
    def _compute_IPS(self, x, num_epoch=500, lr=0.05, lamb=1e-5, tol=1e-4, verbose=False):
        """Pre-train propensity model with extended epochs (from V3)"""
        print('Stage1: computing_IPS (V4 with V3 training)', lr, lamb)
        
        # Generate obs from x
        obs = sps.csr_matrix((np.ones(x.shape[0]), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=lr, weight_decay=lamb)
        # Add learning rate scheduler for propensity (from V3)
        scheduler_prop = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_propensity, mode='min', 
                                                                    factor=0.5, patience=30)
        
        num_samples = len(obs)
        total_batch = num_samples // self.batch_size_prop
        x_all = generate_total_sample(self.num_users, self.num_items)
        
        last_loss = 1e9
        early_stop = 0
        
        for epoch in tqdm(range(num_epoch), desc='Stage1: computing_IPS'):
            # Sampling all samples
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0
            
            for idx in range(total_batch):
                # Mini-batch training
                x_all_idx = ul_idxs[idx * self.batch_size_prop : (idx+1) * self.batch_size_prop]
                x_sampled = x_all[x_all_idx]
                
                # Propensity prediction - use standard forward (no is_training parameter)
                prop = self.model_prop(torch.LongTensor(x_sampled).to(self.device))
                
                # Get observed labels
                sub_obs = obs[x_all_idx]
                sub_obs = torch.Tensor(sub_obs).to(self.device)
                
                # MSE loss for propensity
                prop_loss = F.mse_loss(prop, sub_obs)
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                # Gradient clipping (from V3)
                torch.nn.utils.clip_grad_norm_(self.model_prop.parameters(), max_norm=1.0)
                optimizer_propensity.step()
                
                epoch_loss += prop_loss.detach().cpu().numpy()
            
            # Learning rate scheduling (from V3)
            scheduler_prop.step(epoch_loss)
            
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 10:  # Increased early stop patience (from V3)
                    if verbose:
                        print('[Stage1] Early stop at epoch {}, loss: {}'.format(epoch, epoch_loss))
                    break
                else:
                    early_stop += 1
            else:
                early_stop = 0
                
            last_loss = epoch_loss
            
            if epoch % 50 == 0 and verbose:
                print('[Stage1] epoch: {}, loss: {}'.format(epoch, epoch_loss))
    
    def fit(self, x, y, x_test=None, y_test=None, G=4, alpha=1, beta=1, theta=1, num_epoch=1000, 
            pred_lr=0.05, impu_lr=0.05, prop_lr=0.05, dis_lr=0.05,
            lamb_prop=0, lamb_pred=0, lamb_imp=0, dis_lamb=0, gamma=0.05, num_bins=10,
            tol=1e-4, verbose=True, early_stop_patience=30, early_stop_min_delta=1e-4, 
            eval_freq=5, progress_callback=None, grad_clip_norm=1.0):
        """
        Training with V3's enhancements but using standard models
        """
        
        print('Stage2: fitting (V4 with standard models + V3 training)', G, alpha, beta, theta, gamma, num_bins, 
              pred_lr, impu_lr, prop_lr, dis_lr, lamb_prop, lamb_pred, lamb_imp, dis_lamb)
        
        # Create optimizers
        optimizer_prediction = torch.optim.Adam(self.model_pred.parameters(), lr=pred_lr, weight_decay=lamb_pred)
        optimizer_imputation = torch.optim.Adam(self.model_impu.parameters(), lr=impu_lr, weight_decay=lamb_imp)
        optimizer_propensity = torch.optim.Adam(self.model_prop.parameters(), lr=prop_lr, weight_decay=lamb_prop)
        optimizer_dis = torch.optim.Adam(self.model_abc.parameters(), lr=dis_lr, weight_decay=dis_lamb)
        
        # Learning rate schedulers (from V3)
        scheduler_pred = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_prediction, T_max=num_epoch, eta_min=pred_lr*0.1)
        scheduler_impu = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_imputation, T_max=num_epoch, eta_min=impu_lr*0.1)
        scheduler_prop = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_propensity, T_max=num_epoch, eta_min=prop_lr*0.1)
        scheduler_dis = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=num_epoch, eta_min=dis_lr*0.1)
        
        # Generate all samples and obs
        x_all = generate_total_sample(self.num_users, self.num_items)
        obs = sps.csr_matrix((np.ones(len(y)), (x[:, 0], x[:, 1])), 
                            shape=(self.num_users, self.num_items), dtype=np.float32).toarray().reshape(-1)
        
        num_samples = len(x)
        total_batch = num_samples // self.batch_size
        
        last_loss = 1e9
        early_stop = 0
        stop = 5  # Default stop value
        
        # Use adaptive binning like original MF_Minimax
        bin_edges = torch.linspace(0, 1, steps=int(num_bins) + 1, device=self.device)[1:-1]
        print('bin_edges (V4)', bin_edges)
        
        # Initialize early stopping variables
        use_early_stopping = x_test is not None and y_test is not None
        if use_early_stopping:
            from sklearn.metrics import roc_auc_score
            best_test_auc = -float('inf')
            patience_counter = 0
            best_epoch = 0
            # Save initial model states (use copy.deepcopy from V3)
            best_model_states = {
                'pred': copy.deepcopy(self.model_pred.state_dict()),
                'impu': copy.deepcopy(self.model_impu.state_dict()),
                'prop': copy.deepcopy(self.model_prop.state_dict()),
                'abc': copy.deepcopy(self.model_abc.state_dict())
            }
            if verbose:
                print("Early stopping enabled with patience =", early_stop_patience)
        
        # Training loop
        for epoch in tqdm(range(num_epoch), desc='Stage2: fitting'):
            # Shuffle data
            all_idx = np.arange(num_samples)
            np.random.shuffle(all_idx)
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)
            
            epoch_loss = 0
            
            for idx in range(total_batch):
                # Get batch data
                selected_idx = all_idx[idx * self.batch_size : (idx + 1) * self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_x_tensor = torch.LongTensor(sub_x).to(self.device)
                sub_y_tensor = torch.Tensor(sub_y).to(self.device)
                
                # Add bounds checking (from V3)
                start_idx = G * idx * self.batch_size
                end_idx = min(G * (idx + 1) * self.batch_size, len(ul_idxs))
                
                if start_idx >= len(ul_idxs):
                    break  # No more samples available
                    
                x_all_idx = ul_idxs[start_idx:end_idx]
                
                if len(x_all_idx) == 0:
                    continue  # Skip empty batches
                
                x_sampled = x_all[x_all_idx]
                x_sampled_tensor = torch.LongTensor(x_sampled).to(self.device)
                obs_sampled = torch.Tensor(obs[x_all_idx]).to(self.device)
                
                # Propensity model on counterfactual samples
                prop_sampled = self.model_prop(x_sampled_tensor)
                with torch.no_grad():
                    prop_user_emb, prop_item_emb = self.model_prop.get_emb(x_sampled_tensor)
                
                # Use equal frequency binning (adaptive binning)
                bin_indices, full_boundaries = equal_frequency_binning(prop_sampled.detach(), bin_edges, n_bins=num_bins)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
                bin_sum_index = torch.nn.functional.one_hot(bin_indices.long(), num_classes=int(num_bins)).float()
                
                # Discriminator training
                prop_error_dis = self.model_abc(prop_user_emb.detach(), prop_item_emb.detach()) * (obs_sampled - prop_sampled.detach())
                bin_prop_error_dis = torch.matmul(prop_error_dis.unsqueeze(0), bin_sum_index).squeeze(0)
                prop_abc_loss_dis = - bin_prop_error_dis.abs().sum() / float(num_samples)
                
                optimizer_dis.zero_grad()
                prop_abc_loss_dis.backward()
                torch.nn.utils.clip_grad_norm_(self.model_abc.parameters(), max_norm=grad_clip_norm)
                optimizer_dis.step()
                
                # Propensity training
                prop_error_prop = self.model_abc.predict(prop_user_emb.detach(), prop_item_emb.detach()).detach() * (obs_sampled - prop_sampled)
                bin_prop_error_prop = torch.matmul(prop_error_prop.unsqueeze(0), bin_sum_index).squeeze(0)
                prop_abc_loss_prop = bin_prop_error_prop.abs().sum() / float(num_samples)
                prop_nll_loss = F.binary_cross_entropy(prop_sampled, obs_sampled, reduction='mean')
                prop_loss = prop_nll_loss + beta * prop_abc_loss_prop
                
                optimizer_propensity.zero_grad()
                prop_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_prop.parameters(), max_norm=grad_clip_norm)
                optimizer_propensity.step()
                
                # Now train prediction and imputation models on observed samples
                # Use standard forward calls (no is_training parameter)
                pred = self.model_pred(sub_x_tensor)
                imputation_y = self.model_impu.predict(sub_x_tensor)
                
                # Get inverse propensity weights for observed samples
                inv_prop = 1.0 / torch.clip(self.model_prop.predict(sub_x_tensor), gamma, 1.0)
                
                # Get predictions for counterfactual samples
                pred_u = self.model_pred(x_sampled_tensor)
                imputation_y1 = self.model_impu.predict(x_sampled_tensor)
                
                # Compute DR losses for prediction model (same as V1)
                xent_loss = F.binary_cross_entropy(pred, sub_y_tensor, weight=inv_prop.detach(), reduction='sum')
                imputation_loss = F.binary_cross_entropy(pred, imputation_y.detach(), reduction='sum')
                ips_loss = (xent_loss - imputation_loss)  # batch size
                direct_loss = F.binary_cross_entropy(pred_u, imputation_y1.detach(), reduction='sum')
                dr_loss = (ips_loss + direct_loss) / float(x_sampled.shape[0])
                
                optimizer_prediction.zero_grad()
                dr_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_pred.parameters(), max_norm=grad_clip_norm)
                optimizer_prediction.step()
                
                # Train imputation model (same as V1)
                pred = self.model_pred.predict(sub_x_tensor)
                imputation_y = self.model_impu(sub_x_tensor)
                
                e_loss = F.binary_cross_entropy(pred.detach(), sub_y_tensor, reduction='none')
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred.detach(), reduction='none')
                
                imp_loss = torch.sum(((e_loss - e_hat_loss) ** 2) * inv_prop.detach()) / float(x_sampled.shape[0])
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_impu.parameters(), max_norm=grad_clip_norm)
                optimizer_imputation.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()
            
            # Learning rate scheduling (from V3)
            scheduler_pred.step()
            scheduler_impu.step()
            scheduler_prop.step()
            scheduler_dis.step()
            
            # Variables to track for callback
            current_train_auc = None
            current_test_auc = None
            
            # Calculate train AUC periodically for callback
            if progress_callback and (epoch + 1) % eval_freq == 0:
                with torch.no_grad():
                    train_pred = self.predict(x)
                    current_train_auc = roc_auc_score(y, train_pred)
            
            # Evaluation and early stopping (enhanced from V3)
            if use_early_stopping and (epoch + 1) % eval_freq == 0:
                with torch.no_grad():
                    # Compute test AUC
                    test_pred = self.predict(x_test)
                    test_auc = roc_auc_score(y_test, test_pred)
                    current_test_auc = test_auc
                    
                    # Compute train AUC on a sample (from V3)
                    if not current_train_auc:
                        train_sample_size = min(10000, len(x))
                        train_idx = np.random.choice(len(x), train_sample_size, replace=False)
                        train_pred = self.predict(x[train_idx])
                        train_auc = roc_auc_score(y[train_idx], train_pred)
                    else:
                        train_auc = current_train_auc
                
                # Check for improvement
                if test_auc > best_test_auc + early_stop_min_delta:
                    best_test_auc = test_auc
                    patience_counter = 0
                    best_epoch = epoch
                    # Save best model states
                    best_model_states = {
                        'pred': copy.deepcopy(self.model_pred.state_dict()),
                        'impu': copy.deepcopy(self.model_impu.state_dict()),
                        'prop': copy.deepcopy(self.model_prop.state_dict()),
                        'abc': copy.deepcopy(self.model_abc.state_dict())
                    }
                    if verbose:
                        print(f'\n[Early Stop] New best test AUC: {test_auc:.6f} at epoch {epoch}')
                else:
                    patience_counter += 1
                    if verbose:
                        print(f"[Minimax V4] epoch:{epoch}, xent:{epoch_loss:.4f}, test_auc:{test_auc:.4f} (patience: {patience_counter}/{early_stop_patience})")
                
                # Check patience
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f'\n[Early Stop] Patience exhausted. Best test AUC: {best_test_auc:.6f} at epoch {best_epoch}')
                    # Restore best model
                    self.model_pred.load_state_dict(best_model_states['pred'])
                    self.model_impu.load_state_dict(best_model_states['impu'])
                    self.model_prop.load_state_dict(best_model_states['prop'])
                    self.model_abc.load_state_dict(best_model_states['abc'])
                    return best_epoch
            
            elif epoch % 10 == 0 and verbose:
                print("[Minimax V4] epoch:{}, xent:{}".format(epoch, epoch_loss))
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    epoch=epoch,
                    total_epochs=num_epoch,
                    loss=epoch_loss,
                    train_auc=current_train_auc,
                    test_auc=current_test_auc
                )
            
            # Regular convergence check
            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > stop:
                    if verbose:
                        print('[Training] Early stop at epoch {}, loss: {}'.format(epoch, epoch_loss))
                    return epoch
                else:
                    early_stop += 1
            else:
                early_stop = 0
            
            last_loss = epoch_loss
            
            if verbose and epoch % 50 == 0:
                print('[Training] epoch: {}, loss: {:.4f}'.format(epoch, epoch_loss))
                if use_early_stopping:
                    print(f'[Training] Current LRs - pred: {scheduler_pred.get_last_lr()[0]:.6f}, ' + 
                          f'impu: {scheduler_impu.get_last_lr()[0]:.6f}, ' +
                          f'prop: {scheduler_prop.get_last_lr()[0]:.6f}, ' +
                          f'dis: {scheduler_dis.get_last_lr()[0]:.6f}')
        
        # If we complete all epochs and early stopping was used, restore best model
        if use_early_stopping and best_epoch < epoch:
            if verbose:
                print(f'\nTraining completed. Restoring best model from epoch {best_epoch} with test AUC: {best_test_auc:.6f}')
            self.model_pred.load_state_dict(best_model_states['pred'])
            self.model_impu.load_state_dict(best_model_states['impu'])
            self.model_prop.load_state_dict(best_model_states['prop'])
            self.model_abc.load_state_dict(best_model_states['abc'])
        
        return epoch
    
    def predict(self, x):
        """Simple prediction without train/eval mode switching (standard models don't need it)"""
        x_tensor = torch.LongTensor(x).to(self.device)
        pred = self.model_pred.predict(x_tensor)
        return pred.detach().cpu().numpy()
    