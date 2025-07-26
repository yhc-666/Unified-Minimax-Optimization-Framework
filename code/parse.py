import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="propensity framework")
    parser.add_argument('--data_path', type=str, default='../data', help="data path")
    parser.add_argument('--data_name', type=str, default='yahoo', help="data name")
    parser.add_argument('--thres', type=int, default=3, help='threshold')
    parser.add_argument('--train_rate', type=float, default=1.0, help='val_rate')
    parser.add_argument('--val_rate', type=float, default=0.1, help='val_rate') 
    parser.add_argument('--tol', type=float, default=1e-4, help='val_rate')

    parser.add_argument('--debias_name', type=str, default='erm', help='mf')
    parser.add_argument('--pred_model_name', type=str, default='mf', help='mf')
    parser.add_argument('--impu_model_name', type=str, default='mf', help='mf')
    parser.add_argument('--prop_model_name', type=str, default='logistic_regression', help='logistic_regression')
    parser.add_argument('--abc_model_name', type=str, default='mlp', help='mlp')

    
    parser.add_argument('--load_param_type', type=str, default='initialized', help='load_para_type')
    parser.add_argument('--copy_model_pred', type=int, default=0, help='copy_model_pred')
    parser.add_argument('--embedding_k', type=int, default=64, help='dimension of all embeddings')
    
    parser.add_argument('--hyper_str', type=str, default='', help='')
    
    parser.add_argument('--grad_type', type=int, default=0, help='grad')
    parser.add_argument('--num_epochs', type=int, default=100, help='the number of iterations')
    parser.add_argument('--batch_size_prop', type=int, default=32768, help='batch size')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
        
    parser.add_argument('--num_bins', type=int, default=10, help='num_bins')
    
    parser.add_argument('--l2_norm', type=float, default=1.0, help='l2_norm')
    parser.add_argument('--beta', type=float, default=1.0, help='beta')
    parser.add_argument('--beta1', type=float, default=1.0, help='beta1')
    parser.add_argument('--beta2', type=float, default=1.0, help='beta2')
    
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('--theta', type=float, default=1.0, help='theta')
    
    parser.add_argument('--gamma', type=float, default=1e-12, help='threshold for clipping propensity')
    parser.add_argument('--G', type=int, default=1, help='counterfactual sample size')
    

    parser.add_argument('--pred_lr', type=float, default=0.01, help='weight decay for learning model_pred')
    parser.add_argument('--pred_lamb', type=float, default=0.0, help='weight decay for learning model_pred')
    
    parser.add_argument('--impu_lr', type=float, default=0.01, help='weight decay for learning model_impu')
    parser.add_argument('--impu_lamb', type=float, default=0.0, help='weight decay for learning model_impu')
    
    parser.add_argument('--prop_lr', type=float, default=0.01, help='weight decay for learning model_prop')
    parser.add_argument('--prop_lamb', type=float, default=0.0, help='weight decay for learning model_prop')
    
    parser.add_argument('--prop_lr2', type=float, default=0.01, help='weight decay for learning model_prop')
    parser.add_argument('--prop_lamb2', type=float, default=0.0, help='weight decay for learning model_prop')
    
    parser.add_argument('--dis_lr', type=float, default=0.01, help='weight decay for learning dis')
    parser.add_argument('--dis_lamb', type=float, default=0.0, help='weight decay for learning dis')
    
    parser.add_argument('--thres_epoch', type=int, default=20, help='iter_impcal')
    
    parser.add_argument('--iter_impcal', type=int, default=20, help='iter_impcal')
    parser.add_argument('--num_experts', type=int, default=20, help='num_experts')
    parser.add_argument('--G_cal', type=int, default=5, help='G_cal')

    parser.add_argument('--impu_cal_lr', type=float, default=0.01, help='weight decay for learning model_impu')
    parser.add_argument('--impu_cal_lamb', type=float, default=0.0, help='weight decay for learning model_impu')
    
    parser.add_argument('--prop_cal_lr', type=float, default=0.01, help='weight decay for learning model_prop')
    parser.add_argument('--prop_cal_lamb', type=float, default=0.0, help='weight decay for learning model_prop')
    
    parser.add_argument('--sever', type=str, default='han5', help='sever')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    
    parser.add_argument('--auc_type', type=str, default='no_weighted', help='tune on ?')
    parser.add_argument('--tune_type', type=str, default='val', help='tune on ?')
    parser.add_argument('--tensorborad_path', type=str, default='../tensorboards/', help='tensorboard path')
    parser.add_argument('--is_tensorboard', type=int, default=1, help='tensorboard')
    parser.add_argument('--ex_idx', type=int, default=1, help='ex_idx')
    



    return parser.parse_args()


