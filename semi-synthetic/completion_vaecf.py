"""
Using Variational Autoencoder for Collaborative Filtering (VAE-CF) to generate the complete rating matrix.
"""
import pickle
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from VAECF import VAECF
import torch
import torch.optim as optim

matrix = np.loadtxt("semi-synthetic/data/ml-100k/u.data", dtype=int)[:, :-1]
user = matrix[:, 0] - 1
item = matrix[:, 1] - 1
rating = matrix[:, 2]
user_num = np.max(user)+1
item_num = np.max(item)+1
print(f"Users: {user_num}, Items: {item_num}")
total_num = user.shape[0]
user_train, item_train, rating_train = user[:int(total_num*0.9)], item[:int(total_num*0.9)], rating[:int(total_num*0.9)]
user_test, item_test, rating_test = user[int(total_num*0.9):], item[int(total_num*0.9):], rating[int(total_num*0.9):]
train_num = user_train.shape[0]

batch_size = 1024
embedding_size = 64
l2_reg_lambda = 0.001

# VAE-CF model - proper rating prediction (no sigmoid)
model = VAECF(user_num=user_num, item_num=item_num, embedding_size=embedding_size,
              l2_reg_lambda=l2_reg_lambda, hidden_dims=(600, 200), use_kl=True, kl_beta=0.2)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# Prepare training pairs
x_train = np.column_stack([user_train, item_train])
x_test = np.column_stack([user_test, item_test])

# Build user interaction history from training data (using actual ratings)
print("\nBuilding user interaction history...")
model.set_user_hist_from_pairs(x_train, rating_train.astype(np.float32))

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

early_stop = 1
best_mse = 100
epoch = 0
print("\nTraining VAE-CF model...")
while early_stop < 5:
    epoch += 1
    n_batch = train_num // batch_size
    model.train()

    epoch_loss = 0.0
    for batch in range(n_batch):
        # Get batch data
        batch_user = user_train[batch * batch_size:(batch + 1) * batch_size]
        batch_item = item_train[batch * batch_size:(batch + 1) * batch_size]
        batch_rating = rating_train[batch * batch_size:(batch + 1) * batch_size]

        # Convert to tensors
        user_tensor = torch.LongTensor(batch_user).to(device)
        item_tensor = torch.LongTensor(batch_item).to(device)
        rating_tensor = torch.FloatTensor(batch_rating).to(device)

        # Forward pass
        prediction = model.forward(user_tensor, item_tensor)
        total_loss, mse_loss = model.loss(prediction, rating_tensor)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    # Evaluation
    model.eval()
    with torch.no_grad():
        user_test_tensor = torch.LongTensor(user_test).to(device)
        item_test_tensor = torch.LongTensor(item_test).to(device)
        rating_test_tensor = torch.FloatTensor(rating_test).to(device)
        test_prediction = model.forward(user_test_tensor, item_test_tensor)
        _, mse = model.loss(test_prediction, rating_test_tensor)
        mse = mse.item()

    if mse < best_mse:
        best_mse = mse
        early_stop = 0
    else:
        early_stop += 1
    print(f"Epoch: {epoch}, Train Loss: {epoch_loss/n_batch:.4f}, Test MSE: {mse:.4f}")

# Generate predictions for all user-item pairs
print("\nGenerating predictions for all user-item pairs...")
all_users = np.array([x0 for x0 in np.arange(user_num) for y0 in np.arange(item_num)])
all_items = np.array([y0 for x0 in np.arange(user_num) for y0 in np.arange(item_num)])

# Generate predictions in batches to avoid memory issues
model.eval()
all_predictions = []
batch_size_pred = 10000
n_batches = (len(all_users) + batch_size_pred - 1) // batch_size_pred

with torch.no_grad():
    for i in range(n_batches):
        start_idx = i * batch_size_pred
        end_idx = min((i + 1) * batch_size_pred, len(all_users))

        batch_users = all_users[start_idx:end_idx]
        batch_items = all_items[start_idx:end_idx]

        user_tensor = torch.LongTensor(batch_users).to(device)
        item_tensor = torch.LongTensor(batch_items).to(device)
        batch_pred = model.forward(user_tensor, item_tensor)
        all_predictions.append(batch_pred.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"Processed {end_idx}/{len(all_users)} pairs...")

prediction = np.concatenate(all_predictions)

# Ensure directory exists
import os
os.makedirs("semi-synthetic/data_vaecf", exist_ok=True)

file = open("semi-synthetic/data_vaecf/predicted_matrix", "wb")
pickle.dump(prediction, file)
pickle.dump(user_num, file)
pickle.dump(item_num, file)
file.close()
print(f"\nPredicted matrix saved to data_vaecf/! Shape: {prediction.shape}")
print(f"Prediction range: [{prediction.min():.2f}, {prediction.max():.2f}]")
