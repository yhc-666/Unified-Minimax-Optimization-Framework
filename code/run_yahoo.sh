
CUDA_VISIBLE_DEVICES=0 nohup python -u dr_jl_abc_train_norm.py --data_name=yahoo --thres=3 --debias_name=dr_jl_abc_efb_l2_norm --pred_model_name=mf --prop_model_name=logistic_regression --abc_model_name=mlp --copy_model_pred=0 --seed=2024 --embedding_k=64 --batch_size=4096 --batch_size_prop=32768 --num_bins=10 --l2_norm=1.0 --sever=sui --tune_type=val --ex_idx=1 > ../outputs/yahoo/dr_jl_abc_efb_l2_norm/dr_jl_abc_efb_l2_norm_mf_logistic_regression_mlp_64_4096_32768_10_1_grad_type_optuna_float_on_val_ex1.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 nohup python -u dr_jl_abc_train_all_ones.py --data_name=yahoo --thres=3 --debias_name=dr_jl_abc_all_ones --pred_model_name=mf --prop_model_name=logistic_regression --abc_model_name=mlp --copy_model_pred=0 --seed=2024 --embedding_k=64 --batch_size=4096 --batch_size_prop=32768 --num_bins=10 --beta=1.0 --sever=sui --tune_type=val --ex_idx=1 > ../outputs/yahoo/dr_jl_abc_all_ones/dr_jl_abc_all_ones_mf_logistic_regression_mlp_64_4096_32768_10_1_grad_type_optuna_float_on_val_ex1.log 2>&1 &




CUDA_VISIBLE_DEVICES=0 nohup python -u dr_jl_abc_train.py --data_path=/NAS/phang/propensity_framwork/data --thres=3 --debias_name=dr_jl_abc_eqb --pred_model_name=mf --prop_model_name=logistic_regression --abc_model_name=mlp --copy_model_pred=0 --seed=2024 --embedding_k=64 --batch_size=4096 --batch_size_prop=32768 --num_bins=10 --beta=1.0 --sever=han5 --tune_type=val --ex_idx=1 > /data2/phang/projects/propensity_framework/outputs/yahoo/dr_jl_abc_eqb/dr_jl_abc_eqb_mf_logistic_regression_mlp_64_4096_32768_10_1_grad_type_optuna_float_on_val_ex1.log 2>&1 &



CUDA_VISIBLE_DEVICES=5 nohup python -u dr_jl_abc_train.py --data_path=/NAS/phang/propensity_framwork/data --thres=3 --debias_name=dr_jl_abc_eqb_free --pred_model_name=mf --prop_model_name=logistic_regression --abc_model_name=logistic_regression --copy_model_pred=0 --seed=2024 --embedding_k=64 --batch_size=4096 --batch_size_prop=32768 --num_bins=10 --beta=1.0 --sever=han5 --tune_type=val --ex_idx=1 > /data2/phang/projects/propensity_framework/outputs/yahoo/dr_jl_abc_eqb_free/dr_jl_abc_eqb_free_mf_logistic_regression_logistic_regression_64_4096_32768_10_1_grad_type_optuna_float_on_val_ex1.log 2>&1 &
