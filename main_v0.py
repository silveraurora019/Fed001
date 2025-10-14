# main_v0.py
import os
import copy
import numpy as np
import random
import time
import torch
import torch.nn as nn

# 导入 v0 组件
from util.options_v0 import args_parser
from util.local_training_v0 import LocalUpdate, globaltest
from util.fedavg import FedAvg # 导入标准 FedAvg (按数据量加权)
from util.util_v0 import add_noise
from util.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Baseline: Standard FedAvg (Version 0)
(H. Brendan McMahan, et al. 2017)
"""

if __name__ == '__main__':

    start_time = time.time()

    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "./record/"

    # --- 1. 加载数据和模型 ---
    dataset_train, dataset_test, dict_users = get_dataset(args)
    
    # --- 2. 添加噪声 (为了与 v1-v7 公平对比) ---
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    # --- 3. 设置日志 ---
    if not os.path.exists(rootpath + 'txtsave/'): os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/V0_FedAvg_%s_%s_NL_%.1f_LB_%.1f_Rnd_%d_E_%d_Frac_%.2f_LR_%.3f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.rounds, 
        args.local_ep, args.frac, args.lr, args.seed)

    if args.iid: txtpath += "_IID"
    else: txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    f_acc = open(txtpath + '_acc.txt', 'a')

    f_acc.write("="*50 + "\n")
    f_acc.write("Training Parameters (V0 - Standard FedAvg):\n")
    f_acc.write(str(args) + "\n")
    f_acc.write("="*50 + "\n")
    f_acc.flush()
    
    # --- 4. 构建模型 ---
    netglob = build_model(args)
    max_accuracy = 0.0

    # ============================ Standard FedAvg Training Loop ============================
    print("\n" + "="*25 + " Stage: Standard FedAvg Training " + "="*25, flush=True)
    final_accuracies = []
    
    # m = 客户端选择数量
    m = max(int(args.frac * args.num_users), 1)

    # 论文中的 T (rounds) 
    for rnd in range(args.rounds):
        print(f"\n--- FedAvg Round: {rnd+1}/{args.rounds} ---", flush=True)
        w_locals = []
        
        # 论文: S_t <- (random set of m clients) 
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"Selected clients for this round: {idxs_users}", flush=True)
        
        # 论文: for each client k in S_t in parallel 
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            
            # 论文: w_k <- ClientUpdate(k, w_t) 
            w, _ = local.update_weights(
                net=copy.deepcopy(netglob).to(args.device), 
                seed=args.seed,
                w_g=netglob.state_dict(),
                epoch=args.local_ep,
                mu=0 # mu=0 确保 FedProx 项被关闭
            )
            w_locals.append({'id': idx, 'w': w})
        
        # 论文: w_t+1 <- sum( (n_k / n) * w_k ) 
        # 您的 util/fedavg.py 中的 FedAvg 函数实现了这一点
        dict_len = [len(dict_users[d['id']]) for d in w_locals]
        w_glob = FedAvg([d['w'] for d in w_locals], dict_len)
        
        netglob.load_state_dict(copy.deepcopy(w_glob))

        # --- 评估 ---
        acc_s3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        final_accuracies.append(acc_s3)
        max_accuracy = max(max_accuracy, acc_s3)
        
        print(f"Test Accuracy after round {rnd+1}: {acc_s3:.4f}", flush=True)
        f_acc.write(f"round {rnd}, test acc  {acc_s3:.4f} \n"); f_acc.flush()

    # ============================ Final Result Output ============================
    print("\n" + "="*30 + " Final Results " + "="*30, flush=True)
    if len(final_accuracies) >= 10:
        last_10_accuracies = final_accuracies[-10:]
        mean_acc = np.mean(last_10_accuracies)
        var_acc = np.var(last_10_accuracies)
        print(f"Mean of last 10 rounds test accuracy: {mean_acc:.4f}", flush=True)
        print(f"Variance of last 10 rounds test accuracy: {var_acc:.6f}", flush=True)
        f_acc.write(f"\nMean of last 10 rounds test accuracy: {mean_acc:.4f}\n")
        f_acc.write(f"Variance of last 10 rounds test accuracy: {var_acc:.6f}\n")
    elif len(final_accuracies) > 0:
        mean_acc = np.mean(final_accuracies)
        var_acc = np.var(final_accuracies)
        print(f"Mean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}", flush=True)
        print(f"Variance of final {len(final_accuracies)} rounds test accuracy: {var_acc:.6f}", flush=True)
        f_acc.write(f"\nMean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}\n")
        f_acc.write(f"Variance of final {len(final_accuracies)} rounds test accuracy: {var_acc:.6f}\n")
    
    print(f"\nMaximum test accuracy achieved: {max_accuracy:.4f}", flush=True)
    f_acc.write(f"\nMaximum test accuracy achieved: {max_accuracy:.4f}\n")

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s", flush=True)
    f_acc.write(f"\nTotal Training Time: {hours}h {minutes}m {seconds}s\n")


    f_acc.close()
    torch.cuda.empty_cache()
    print("\nTraining Finished!", flush=True)