"""
Neural Collaborative Filtering (NCF) for rating prediction.
Similar to MF.py but uses concatenation + linear layer instead of dot product.
"""
import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, user_num, item_num, embedding_size, l2_reg_lambda):
        super(NCF, self).__init__()
        self.l2_reg_lambda = l2_reg_lambda

        # Initialize embeddings and biases with normal distribution
        self.user_embedding = nn.Parameter(torch.randn(user_num, embedding_size))
        self.item_embedding = nn.Parameter(torch.randn(item_num, embedding_size))
        self.user_bias = nn.Parameter(torch.randn(user_num))
        self.item_bias = nn.Parameter(torch.randn(item_num))
        self.global_bias = nn.Parameter(torch.randn(1))

        # Linear layer for concatenated embeddings (no sigmoid!)
        self.linear = nn.Linear(embedding_size * 2, 1, bias=False)

    def forward(self, user_id, item_id):
        # Lookup embeddings and biases
        user_feature = self.user_embedding[user_id]
        item_feature = self.item_embedding[item_id]
        b_u = self.user_bias[user_id]
        b_i = self.item_bias[item_id]

        # Concatenate embeddings and pass through linear layer
        z_emb = torch.cat([user_feature, item_feature], dim=1)
        prediction = self.linear(z_emb).squeeze()

        # Add biases
        prediction = prediction + b_u + b_i + self.global_bias

        return prediction

    def loss(self, prediction, y):
        # MSE loss
        mse = torch.mean(torch.square(prediction - y))

        # L2 regularization
        l2_regularization = torch.sum(torch.square(self.user_embedding))
        l2_regularization += torch.sum(torch.square(self.item_embedding))
        l2_regularization += torch.sum(torch.square(self.user_bias))
        l2_regularization += torch.sum(torch.square(self.item_bias))
        l2_regularization += torch.sum(torch.square(self.global_bias))
        l2_regularization += torch.sum(torch.square(self.linear.weight))

        total_loss = mse + self.l2_reg_lambda * l2_regularization

        return total_loss, mse
