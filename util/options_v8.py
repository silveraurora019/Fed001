# util/options_v8.py
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # === v8: 统一的训练轮次 ===
    parser.add_argument('--rounds', type=int, default=200, help="total number of training rounds")

    # --- 保留v5的核心参数 ---
    # federated arguments
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac', type=float, default=0.3, help="fraction of clients to select each round")
    parser.add_argument('--num_users', type=int, default=50, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")

    # noise arguments
    parser.add_argument('--level_n_system', type=float, default=0, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0, help="lower bound of noise level")

    # correction arguments
    parser.add_argument('--correction', action='store_true', help='whether to correct noisy labels')

    # other arguments
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--iid', type=bool, default=False, help="i.i.d. (True) or non-i.i.d. (False)")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--gpu', type=int, default=2, help="GPU ID, default: 0")

    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=1.0, help="parameter for dirichlet distribution")

    # Method-specific arguments (from v5)
    parser.add_argument('--k_clusters', type=int, default=5, help="Number of sub-prototypes per class")
    parser.add_argument('--fairness_alpha', type=float, default=0.5, help="Alpha for balancing noise and contribution weights (0 to 1)")
    parser.add_argument('--exp_temp', type=float, default=0.1, help="Temperature for exponential weighting based on noise level")
    parser.add_argument('--beta_pseudo', type=float, default=0.5, help="Confidence threshold for pseudo-labeling in Stage 1")
    parser.add_argument('--lambda_mcpcl', type=float, default=0.1, help="Weight for the Multi-Client Prototype Contrastive Loss")
    parser.add_argument('--temp_mcpcl', type=float, default=0.1, help="Temperature for the Multi-Client Prototype Contrastive Loss")
    parser.add_argument('--affinity_beta', type=float, default=1.0, help="Sensitivity parameter for affinity calculation")

    # (移除了 fine_tuning 和多阶段轮次参数)

    return parser.parse_args()