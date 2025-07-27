"""
Using standard matrix factorization to generate the complete rating matrix.
"""
import pickle
import numpy as np
from MF import MF
import torch
import torch.optim as optim

matrix = np.loadtxt("semi-synthetic/data/ml-100k/u.data", dtype=int)[:, :-1]
user = matrix[:, 0] - 1
item = matrix[:, 1] - 1
rating = matrix[:, 2]
user_num = np.max(user)+1
item_num = np.max(item)+1
print(user_num, item_num)
total_num = user.shape[0]
user_train, item_train, rating_train = user[:int(total_num*0.9)], item[:int(total_num*0.9)], rating[:int(total_num*0.9)]
user_test, item_test, rating_test = user[int(total_num*0.9):], item[int(total_num*0.9):], rating[int(total_num*0.9):] 
train_num = user_train.shape[0]

batch_size = 1024
l2_reg_lambda = 1e-3    # Validated by grid-search
mf = MF(user_num=user_num, item_num=item_num, embedding_size=64, l2_reg_lambda=l2_reg_lambda)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mf = mf.to(device)

# Initialize optimizer
optimizer = optim.Adam(mf.parameters())

# Convert test data to tensors
user_test_tensor = torch.LongTensor(user_test).to(device)
item_test_tensor = torch.LongTensor(item_test).to(device)
rating_test_tensor = torch.FloatTensor(rating_test).to(device)

early_stop = 1
best_mse = 100
epoch = 0
while early_stop < 5:
    epoch += 1
    n_batch = train_num // batch_size
    mf.train()
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
        prediction = mf(user_tensor, item_tensor)
        loss, _ = mf.loss(prediction, rating_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation
    mf.eval()
    with torch.no_grad():
        test_prediction = mf(user_test_tensor, item_test_tensor)
        _, mse = mf.loss(test_prediction, rating_test_tensor)
        mse = mse.item()
    
    if mse < best_mse:
        best_mse = mse
        early_stop = 0
    else:
        early_stop += 1
    print("Epoch:", epoch, "MSE:", mse)

# Generate predictions for all user-item pairs
all_matrix = np.array([[x0, y0] for x0 in np.arange(user_num) for y0 in np.arange(item_num)])
user_all = all_matrix[:, 0]
item_all = all_matrix[:, 1]

# Convert to tensors
user_all_tensor = torch.LongTensor(user_all).to(device)
item_all_tensor = torch.LongTensor(item_all).to(device)

# Generate predictions
mf.eval()
with torch.no_grad():
    prediction = mf(user_all_tensor, item_all_tensor)
    prediction = prediction.cpu().numpy()

file = open("semi-synthetic/data/predicted_matrix", "wb")
pickle.dump(prediction, file)
pickle.dump(user_num, file)
pickle.dump(item_num, file)
file.close()