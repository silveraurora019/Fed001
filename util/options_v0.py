# util/options_v0.py
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # === FedAvg 核心参数 (来自 FedAvg 论文) ===
    parser.add_argument('--rounds', type=int, default=150, help="rounds of training (T)")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs (E)")
    parser.add_argument('--frac', type=float, default=0.3, help="fraction of clients to select (C)")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size (B)")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")

    # === 实验环境参数 (用于对比) ===
    parser.add_argument('--num_users', type=int, default=50, help="number of uses: K")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, default: 0")
    parser.add_argument('--seed', type=int, default=13, help="random seed")
    
    # --- 数据分布与噪声 (用于公平对比) ---
    parser.add_argument('--iid', type=bool, default=False, help="i.i.d. (True) or non-i.i.d. (False)")
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=1)
  
    
    # --- 噪声参数 (保持数据一致) ---
    parser.add_argument('--level_n_system', type=float, default=0, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0, help="lower bound of noise level")

    # --- v1+ 的参数 (v0 中不使用) ---
    # parser.add_argument('--iteration1', ...)
    # parser.add_argument('--rounds1', ...)
    # parser.add_argument('--beta', ...) (FedProx, not FedAvg)
    # parser.add_argument('--k_clusters', ...)
    # parser.add_argument('--fairness_alpha', ...)
    # parser.add_argument('--exp_temp', ...)
    # parser.add_argument('--beta_pseudo', ...)
    # parser.add_argument('--correction', ...)
    # parser.add_argument('--fine_tuning', ...)
    # ... (所有 v1-v7 的特定参数均已移除) ...

    return parser.parse_args()