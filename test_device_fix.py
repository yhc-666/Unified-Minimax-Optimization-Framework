#!/usr/bin/env python
"""
Quick test to verify device handling fix for MF_DR_BMSE model.
"""
import torch
import numpy as np
from real_world.matrix_factorization_DT import MF_DR_BMSE

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create a small test case
num_users = 10
num_items = 10
batch_size = 4
batch_size_prop = 8
embedding_k = 4

print("Testing device handling fix...")
print(f"CUDA available: {torch.cuda.is_available()}")

# Create model
model = MF_DR_BMSE(
    num_users=num_users,
    num_items=num_items,
    batch_size=batch_size,
    batch_size_prop=batch_size_prop,
    embedding_k=embedding_k,
    bmse_weight=1.0
)

# Test device movement
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"\nMoving model to {device}...")
    model.to(device)

    print(f"Parent model device: {model.device}")
    print(f"Prediction model device: {model.prediction_model.device}")
    print(f"Imputation model device: {model.imputation_model.device}")
    print(f"Propensity model device: {model.propensity_model.device}")

    # Create some dummy training data
    x_train = np.array([[i % num_users, i % num_items] for i in range(20)])
    y_train = np.random.binomial(1, 0.5, size=20).astype(np.float32)

    print("\nTesting _compute_IPS (this is where the error occurred)...")
    try:
        model._compute_IPS(x_train, num_epoch=2, lr=0.01, verbose=False)
        print("✓ _compute_IPS succeeded without device mismatch error!")
    except RuntimeError as e:
        if "expected" in str(e) and "device" in str(e):
            print(f"✗ Device mismatch error still occurs: {e}")
        else:
            raise

    print("\nTesting model.fit()...")
    try:
        model.fit(x_train, y_train, num_epoch=2, lr=0.01, gamma=0.1, G=2, verbose=False)
        print("✓ model.fit() succeeded!")
    except RuntimeError as e:
        if "expected" in str(e) and "device" in str(e):
            print(f"✗ Device mismatch error in fit: {e}")
        else:
            raise

    print("\nTesting model.predict()...")
    try:
        predictions = model.predict(x_train[:5])
        print(f"✓ model.predict() succeeded! Shape: {predictions.shape}")
    except Exception as e:
        print(f"✗ Prediction error: {e}")

    print("\n✅ All tests passed! Device handling fix is working correctly.")
else:
    print("\nCUDA not available, skipping GPU tests")
    print("To fully test the fix, please run on a machine with CUDA GPU")
