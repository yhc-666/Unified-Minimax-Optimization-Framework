#!/bin/bash


# auc mse ndcg5 ndcg_10

python real_world/optuna_search.py \
    --dataset coat \
    --metrics auc \
    --directions maximize \
    --n_trials 2 \
    --save_all_trials

python real_world/optuna_search.py \
    --dataset yahoo \
    --metrics auc mse ndcg_5 ndcg_10 \
    --directions maximize minimize maximize maximize \
    --n_trials 100 \
    --save_all_trials

python real_world/optuna_search.py \
    --dataset kuai \
    --metrics auc mse ndcg_50 ndcg_100 \
    --directions maximize minimize maximize maximize \
    --n_trials 100 \
    --save_all_trials