#!/bin/bash

# Example 1: Optimize for both AUC (maximize) and MSE (minimize)

# auc mse ndcg5 ndcg_10

echo "Running multi-objective optimization for AUC and MSE..."
python real_world/optuna_search.py \
    --dataset coat \
    --metrics auc mse ndcg_5 ndcg_10 \
    --directions maximize minimize maximize maximize \
    --n_trials 3 \
    --save_all_trials

# Example 2: Optimize for multiple ranking metrics
echo "Running multi-objective optimization for multiple ranking metrics..."
python real_world/optuna_search.py \
    --dataset coat \
    --metrics auc ndcg recall \
    --directions maximize maximize maximize \
    --n_trials 50 \
    --save_all_trials

# Example 3: Balance between performance and error metrics
echo "Running multi-objective optimization balancing performance and error..."
python optuna_search.py \
    --dataset yahoo \
    --metrics auc mse mae \
    --directions maximize minimize minimize \
    --n_trials 100 \
    --save_all_trials