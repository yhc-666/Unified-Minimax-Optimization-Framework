# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.sparse as sps
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import arguments

from dataset import load_data
from matrix_factorization_DT import generate_total_sample
from utils import gini_index, ndcg_func, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, precision_func, recall_func

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)
mae_func = lambda x,y: np.mean(np.abs(x-y))


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
        print('_compute_IPS', lr, lamb)
        
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

        for epoch in range(num_epoch):
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
        
        print('fit', G, alpha, beta, theta, gamma, num_bins, pred_lr, impu_lr, prop_lr, dis_lr, lamb_prop, lamb_pred, lamb_imp, dis_lamb)
        
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
        
        for epoch in range(num_epoch):
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

    "Minimax"
    mf = MF_Minimax(num_user, num_item, batch_size=train_args['batch_size'], batch_size_prop=train_args['batch_size_prop'],
                    embedding_k=model_args['embedding_k'], embedding_k1=model_args['embedding_k1'],
                    abc_model_name=model_args.get('abc_model_name', 'logistic_regression'),
                    copy_model_pred=model_args.get('copy_model_pred', 1))
    
    # First compute propensity scores
    mf._compute_IPS(x_train, 
                    num_epoch=200,  # Fixed for propensity pre-training
                    lr=model_args.get('prop_lr', 0.01), 
                    lamb=model_args.get('prop_lamb', 0),
                    verbose=True)
    
    # Then train the full model
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


def para(args):
    """Set hyperparameters for different datasets"""
    if args.dataset=="coat":
        args.train_args = {
            "batch_size": 128,              # Mini-batch size for training prediction/imputation models
            "batch_size_prop": 128,         # Mini-batch size for training propensity model
            "gamma": 0.05,                  # Propensity score clipping threshold (clips to [gamma, 1.0] to avoid extreme weights)
            "G": 4,                         # Ratio of unobserved to observed samples (controls exploration in DR estimator)
            "alpha": 0.5,                   # Unused in current implementation (kept for compatibility)
            "beta": 0.1,                    # Weight for adversarial loss in propensity model training
            "theta": 1,                     # Unused in current implementation (kept for compatibility)
            "num_bins": 10                  # Number of bins for propensity score stratification
        }
        args.model_args = {
            "embedding_k": 8,               # Embedding dimension for propensity and discriminator models
            "embedding_k1": 8,              # Embedding dimension for prediction and imputation models
            "pred_lr": 0.005,               # Learning rate for prediction model
            "impu_lr": 0.005,               # Learning rate for imputation model
            "prop_lr": 0.005,               # Learning rate for propensity model during main training
            "dis_lr": 0.005,                # Learning rate for discriminator model
            "lamb_prop": 1e-3,              # Weight decay for propensity model during main training
            "prop_lamb": 1e-3,              # Weight decay for propensity model during pre-training (_compute_IPS)
            "lamb_pred": 1e-4,              # Weight decay for prediction model
            "lamb_imp": 1e-4,               # Weight decay for imputation model
            "dis_lamb": 0.0,                # Weight decay for discriminator model
            "abc_model_name": "logistic_regression",  # Architecture for adversarial discriminator ("logistic_regression" or "mlp")
            "copy_model_pred": 1            # Whether to initialize imputation model with prediction model weights (1=yes, 0=no)
        }
    elif args.dataset=="yahoo":
        args.train_args = {
            "batch_size": 4096,             # Larger batch size for larger dataset
            "batch_size_prop": 4096,        # Batch size for propensity model
            "gamma": 0.05,                  # Same propensity clipping as coat
            "G": 4,                         # Same exploration ratio as coat
            "alpha": 0.5,                   # Unused parameter
            "beta": 1e-5,                   # Much smaller adversarial weight (yahoo needs less regularization)
            "theta": 1,                     # Unused parameter
            "num_bins": 10                  # Same binning strategy
        }
        args.model_args = {
            "embedding_k": 16,              # Larger embeddings for larger dataset
            "embedding_k1": 16,             # Same size for prediction/imputation embeddings
            "pred_lr": 0.01,                # Learning rate for prediction model
            "impu_lr": 0.01,                # Learning rate for imputation model
            "prop_lr": 0.01,                # Learning rate for propensity model
            "dis_lr": 0.01,                 # Learning rate for discriminator model
            "lamb_prop": 1e-3,              # Weight decay for propensity model
            "prop_lamb": 1e-3,              # Weight decay for pre-training
            "lamb_pred": 1e-5,              # Weight decay for prediction model
            "lamb_imp": 1e-1,               # Weight decay for imputation model (prevents overfitting)
            "dis_lamb": 0.0,                # Weight decay for discriminator model
            "abc_model_name": "logistic_regression",  # Same discriminator architecture
            "copy_model_pred": 1            # Initialize imputation from prediction
        }
    elif args.dataset=="kuai":
        args.train_args = {
            "batch_size": 4096,             # Large batch for efficient training
            "batch_size_prop": 4096,        # Same as main batch size
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


# python Minimax.py --dataset coat