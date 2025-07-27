# Unified Minimax Optimization Framework

A PyTorch implementation of minimax optimization techniques for debiased recommendation systems with missing-not-at-random (MNAR) data. This framework implements adversarial propensity learning and doubly robust estimation for unbiased recommendation.

## Overview

This framework addresses the challenge of selection bias in recommendation systems where user feedback is missing-not-at-random (MNAR). It implements a minimax optimization approach that:

- Uses adversarial learning to estimate propensity scores
- Applies doubly robust estimation for unbiased predictions
- Supports multiple real-world datasets (Coat, Yahoo, Kuai)
- Includes hyperparameter optimization via Optuna

## Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- scikit-learn
- scipy
- tqdm
- optuna (for hyperparameter search)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Unified-Minimax-Optimization-Framework.git
cd Unified-Minimax-Optimization-Framework

# Install dependencies
pip install torch numpy scikit-learn scipy tqdm optuna
```

## Dataset Setup

The framework supports three datasets:

1. **Coat Shopping Dataset**: Implicit feedback on coat shopping
2. **Yahoo Music Dataset**: User ratings for music items  
3. **Kuai Dataset**: Large-scale recommendation dataset

Place dataset files in `real_world/data/[dataset_name]/` directory.

## Usage

### Running the Main Model

```bash
# Run on Coat dataset
python real_world/Minimax.py --dataset coat

# Run on Yahoo dataset  
python real_world/Minimax.py --dataset yahoo

# Run on Kuai dataset
python real_world/Minimax.py --dataset kuai
```

### Hyperparameter Optimization with Optuna

For single-objective optimization:
```bash
# Optimize AUC on Coat dataset
python real_world/optuna_search.py --dataset coat --n_trials 100 --metrics auc --save_all_trials

# Optimize NDCG@10 on Yahoo dataset
python real_world/optuna_search.py --dataset yahoo --n_trials 200 --metrics ndcg_10
```

For multi-objective optimization:
```bash
# Optimize both AUC and NDCG@10
python real_world/optuna_search.py --dataset coat --n_trials 200 \
    --metrics auc ndcg_10 --directions maximize maximize

# Optimize AUC (maximize) and MSE (minimize)
python real_world/optuna_search.py --dataset yahoo --n_trials 300 \
    --metrics auc mse --directions maximize minimize
```

### Batch Hyperparameter Search

```bash
# Run optuna search in background
nohup ./real_world/run_optuna.sh > optuna_experiment.log 2>&1 &
```

## Model Architecture

The framework implements a MF_Minimax model with four key components:

1. **Prediction Model**: Matrix factorization for rating prediction
2. **Imputation Model**: Estimates missing ratings
3. **Propensity Model**: Estimates observation probabilities with adversarial training
4. **Discriminator Model**: Provides adversarial signal for propensity learning

### Key Features

- **Adversarial Propensity Learning**: Uses minimax optimization to learn robust propensity scores
- **Doubly Robust Estimation**: Combines IPW and direct estimation for reduced variance
- **Propensity Clipping**: Prevents extreme importance weights
- **Equal Frequency Binning**: Stratifies propensity scores for stable training

## Hyperparameters

### Dataset-Specific Safe Batch Sizes

Based on dataset dimensions and memory constraints:

- **Coat**: 290 users × 300 items
  - Safe batch size: ≤ 20,000
  - Recommended: 128-512
  
- **Yahoo**: 15,401 users × 1,000 items  
  - Safe batch size: ≤ 3,000,000
  - Recommended: 2048-8192
  
- **Kuai**: 7,163 users × 10,596 items
  - Safe batch size: ≤ 15,000,000  
  - Recommended: 2048-8192

### Key Hyperparameters

- `embedding_k`: Embedding dimension for propensity/discriminator models
- `embedding_k1`: Embedding dimension for prediction/imputation models
- `beta`: Weight for adversarial loss in propensity training
- `gamma`: Propensity score clipping threshold
- `G`: Ratio of unobserved to observed samples
- `num_bins`: Number of bins for propensity stratification
- `lamb_*`: L2 regularization weights for each model component

## Output Files

### Optuna Results

Results are saved in `optuna_results/[dataset]/`:
- `[dataset]_all_trials.csv`: All trial results with hyperparameters and metrics
- `[dataset]_pareto_optimal_params.csv`: Pareto optimal solutions for multi-objective
- `[dataset]_summary.txt`: Human-readable summary of best results

### Evaluation Metrics

The framework evaluates:
- **AUC**: Area under ROC curve
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Precision@K**: Precision at top K recommendations
- **Recall@K**: Recall at top K recommendations  
- **F1@K**: Harmonic mean of Precision and Recall
- **MSE/MAE**: Mean Squared/Absolute Error

## Implementation Details

### Minimax Optimization

The framework solves:
```
min_θ max_φ L(θ, φ) = L_pred(θ) + β * L_adv(θ, φ)
```

Where:
- θ: Parameters of propensity model
- φ: Parameters of discriminator
- L_pred: Prediction loss with doubly robust estimation
- L_adv: Adversarial loss for propensity learning

### Training Process

1. **Stage 1**: Pre-train propensity model with standard MLE
2. **Stage 2**: Alternating optimization:
   - Update discriminator to maximize propensity error
   - Update propensity model to minimize prediction + adversarial loss
   - Update prediction/imputation models with doubly robust loss

## Troubleshooting

### Common Issues

1. **ValueError with shape mismatch**: Fixed in latest version - handles users with fewer items than top_k
2. **Memory issues**: Reduce batch_size or embedding dimensions
3. **Slow training**: Enable GPU with CUDA, reduce num_epoch

### Debug Mode

For verbose training output:
```python
# In Minimax.py, set verbose=True in fit() and _compute_IPS() calls
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{minimax2024,
  title={Unified Minimax Optimization Framework for Debiased Recommendations},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Coat dataset from [1]
- Yahoo Music dataset from Yahoo Research
- Kuai dataset from [2]

[1] Schnabel et al. "Recommendations as Treatments: Debiasing Learning and Evaluation" ICML 2016
[2] Gao et al. "Kuai dataset" reference
