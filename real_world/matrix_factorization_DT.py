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

        for epoch in range(num_epoch):

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

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

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
        for epoch in range(num_epoch):
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

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.forward(x)
        return pred.detach().cpu().numpy()



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
        
        for epoch in range(num_epoch):

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

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

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
        
        for epoch in range(num_epoch): 
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

        for epoch in range(num_epoch):

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
                print("[MF-MRDRJL-PS] epoch:{}, xent:{}".format(epoch, epoch_loss))

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

        for epoch in range(num_epoch): 
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
                imputation_y = self.imputation_model.predict(sub_x).to(self.device)
                
                
                x_sampled = x_all[ul_idxs[G*idx* self.batch_size : G*(idx+1)*self.batch_size]]
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation_model.predict(x_sampled).to(self.device)
          
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

                pred = self.prediction_model.predict(sub_x).to(self.device)
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
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")
                
    
    def predict(self, x):
        pred = self.prediction_model.predict(x)
        return pred.detach().cpu().numpy()            
        
    
    
    
    


# DR-BIAS incorrect
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].to(self.device)                

                sub_y = torch.Tensor(sub_y).to(self.device)

                        
                pred = self.prediction_model.forward(sub_x)
                imputation_y = self.imputation.predict(sub_x).to(self.device)
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]
                                       
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
                    print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-MRDR-JL] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-MRDR-JL] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
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
            # x is indices
            user_emb = self.user_emb_table(x_or_user_emb[:, 0])
            item_emb = self.item_emb_table(x_or_user_emb[:, 1])
        else:
            # x_or_user_emb is user_emb, item_emb is provided
            user_emb = x_or_user_emb
            
        z_emb = torch.cat([user_emb, item_emb], axis=1)
        out = self.linear_1(z_emb)
        return torch.sigmoid(torch.squeeze(out))
    
    def forward_logit(self, x):
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])
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
        user_emb = self.user_emb_table(x[:, 0])
        item_emb = self.item_emb_table(x[:, 1])
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

    def _compute_IPS(self, x, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, verbose=False):
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

    def fit(self, x, y, G=4, alpha=1, beta=1, theta=1, num_epoch=1000, 
            pred_lr=0.05, impu_lr=0.05, prop_lr=0.05, dis_lr=0.05,
            lamb_prop=0, lamb_pred=0, lamb_imp=0, dis_lamb=0, gamma=0.05, num_bins=10,
            tol=1e-4, verbose=True):
        """
        Train the MF_Minimax model using adversarial propensity learning and doubly robust estimation
        
        Args:
            x: Training user-item pairs
            y: Training ratings
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
            tol: Early stopping tolerance
            verbose: Whether to print training progress
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
        
        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=self.device)[1:-1]
        print('bin_edges', bin_edges)
        
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
                 
                x_all_idx = ul_idxs[G*idx*self.batch_size:G*(idx+1)*self.batch_size]
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
                        print("[Minimax] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    return epoch
                else:
                    early_stop += 1

            last_loss = epoch_loss
            
            if epoch % 10 == 0 and verbose:
                print("[Minimax] epoch:{}, xent:{}".format(epoch, epoch_loss))
                
        return epoch
    
    def predict(self, x):
        x_tensor = torch.LongTensor(x).to(self.device)
        pred = self.model_pred.predict(x_tensor)
        return pred.detach().cpu().numpy()
