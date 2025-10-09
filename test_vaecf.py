#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify VAECF integration with Minimax framework
Tests on coat dataset with reduced epochs for quick validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'real_world'))

import numpy as np
import torch
from dataset import load_data
from matrix_factorization_DT import MF_Minimax
from utils import rating_mat_to_sample, binarize, set_all_seeds, set_deterministic
from sklearn.metrics import roc_auc_score

# Set seeds
set_all_seeds(2020)
set_deterministic()

print("="*60)
print("Testing VAECF Integration")
print("="*60)

# Load coat dataset (smallest dataset)
print("\n[1/5] Loading coat dataset...")
train_mat, test_mat = load_data("coat")
x_train, y_train = rating_mat_to_sample(train_mat)
x_test, y_test = rating_mat_to_sample(test_mat)
num_user = train_mat.shape[0]
num_item = train_mat.shape[1]

print(f"  - Users: {num_user}, Items: {num_item}")
print(f"  - Training samples: {len(x_train)}")
print(f"  - Test samples: {len(x_test)}")

# Binarize
y_train = binarize(y_train, 3)
y_test = binarize(y_test, 3)

# Test VAECF model
print("\n[2/5] Initializing VAECF model...")
mf = MF_Minimax(
    num_user, num_item,
    batch_size=128,
    batch_size_prop=128,
    embedding_k=32,      # For propensity/discriminator
    embedding_k1=64,     # For prediction/imputation (VAECF latent dim)
    abc_model_name='logistic_regression',
    copy_model_pred=0,   # No copying for VAECF (different architecture)
    pred_model_name='VAECF'
)

print("  ✓ VAECF model initialized successfully")
print(f"  - Model type: {mf.pred_model_name}")
print(f"  - Prediction model: {type(mf.model_pred).__name__}")
print(f"  - Imputation model: {type(mf.model_impu).__name__}")

# Pre-train propensity
print("\n[3/5] Pre-training propensity scores...")
mf._compute_IPS(x_train, num_epoch=50, lr=0.05, lamb=1e-3, verbose=False)
print("  ✓ Propensity scores computed")

# Train with reduced epochs for quick test
print("\n[4/5] Training VAECF (reduced epochs for testing)...")
mf.fit(
    x_train, y_train,
    x_test=x_test, y_test=y_test,
    pred_lr=0.05,
    impu_lr=0.01,
    prop_lr=0.05,
    dis_lr=0.01,
    alpha=0.5,
    beta=0.5,
    theta=1,
    lamb_prop=1e-3,
    lamb_pred=5e-3,
    lamb_imp=1e-4,
    dis_lamb=5e-3,
    G=1,
    gamma=0.017,
    num_bins=30,
    num_epoch=100,  # Reduced from 1000 for quick test
    verbose=False,
    early_stop_patience=10,
    early_stop_min_delta=1e-3,
    eval_freq=10
)
print("  ✓ Training completed")

# Evaluate
print("\n[5/5] Evaluating VAECF...")
train_pred = mf.predict(x_train)
test_pred = mf.predict(x_test)

train_auc = roc_auc_score(y_train, train_pred)
test_auc = roc_auc_score(y_test, test_pred)

print("\n" + "="*60)
print("VAECF Test Results:")
print("="*60)
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"AUC Gap:   {train_auc - test_auc:.4f}")

# Sanity checks
print("\n" + "="*60)
print("Sanity Checks:")
print("="*60)

checks_passed = True

# Check 1: Predictions are in valid range
if np.all((train_pred >= 0) & (train_pred <= 1)) and np.all((test_pred >= 0) & (test_pred <= 1)):
    print("✓ Predictions in valid range [0, 1]")
else:
    print("✗ ERROR: Predictions out of range!")
    checks_passed = False

# Check 2: AUC is reasonable
if test_auc > 0.5:
    print(f"✓ Test AUC ({test_auc:.4f}) better than random (0.5)")
else:
    print(f"✗ WARNING: Test AUC ({test_auc:.4f}) not better than random")
    checks_passed = False

# Check 3: Model can make predictions
if len(test_pred) == len(y_test):
    print(f"✓ Prediction shape matches ({len(test_pred)} samples)")
else:
    print("✗ ERROR: Prediction shape mismatch!")
    checks_passed = False

# Check 4: User history was initialized
if hasattr(mf.model_pred, '_user_hist_dense') and mf.model_pred._user_hist_dense is not None:
    print(f"✓ VAECF user history initialized (shape: {mf.model_pred._user_hist_dense.shape})")
else:
    print("✗ ERROR: VAECF user history not initialized!")
    checks_passed = False

print("\n" + "="*60)
if checks_passed:
    print("SUCCESS: All sanity checks passed! ✓")
    print("VAECF is properly integrated and working.")
else:
    print("FAILURE: Some checks failed! ✗")
    print("Please review the errors above.")
print("="*60)
