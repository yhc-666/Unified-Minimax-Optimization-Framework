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
    --metrics auc ndcg_20 f1_20 \
    --directions maximize maximize maximize \
    --n_trials 200 \
    --save_all_trials