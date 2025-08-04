#!/bin/bash


# auc mse ndcg5 ndcg_10

python real_world/optuna_search.py \
    --dataset coat \
    --metrics auc ndcg_5 f1_5 \
    --directions maximize maximize maximize \
    --n_trials 300 \
    --save_all_trials

python real_world/optuna_search.py \
    --dataset yahoo \
    --metrics auc ndcg_5 f1_5 \
    --directions maximize maximize maximize \
    --n_trials 300 \
    --save_all_trials

python real_world/optuna_search.py \
    --dataset kuai \
    --model_type minimaxv3 \
    --metrics auc ndcg_20 f1_20 \
    --directions maximize maximize maximize \
    --n_trials 600 \
    --output_dir optuna_results/minimax_v3 \
    --save_all_trials

python real_world/optuna_search.py \
    --dataset kuai \
    --model_type minimaxv4 \
    --metrics auc ndcg_20 f1_20 \
    --directions maximize maximize maximize \
    --n_trials 600 \
    --output_dir optuna_results/minimax_v4 \
    --save_all_trials