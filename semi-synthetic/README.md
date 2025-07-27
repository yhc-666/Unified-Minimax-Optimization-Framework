Codes for Semi-synthetic Experiments in Section 4.

1. Download MovieLens 100K dataset from https://grouplens.org/datasets/movielens/, and put the file "u.data" into the directory of "data/".
2. Run "completion.py" to complete the entire rating matrix.
3. Run "convert.py" to generate five predicted CVR matrices.


For evaluation of baselines
```bash
python semi-synthetic/evaluate_baseline.py --model MF_DR --save_results --verbose
python semi-synthetic/evaluate_baseline.py --model MF_MRDR_JL --save_results --verbose
python semi-synthetic/evaluate_baseline.py --model MF_DR_BIAS --save_results --verbose
python semi-synthetic/evaluate_baseline.py --model MF_DR_V2 --save_results --verbose
python semi-synthetic/evaluate_baseline.py --model dr_jl_abc --save_results --verbose
```
