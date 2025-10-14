# util/options_v10.py
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=10, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=150, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=150, help="rounds of training in usual training stage")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    parser.add_argument('--frac1', type=float, default=0.3, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=0.3, help="fration of selected clients in fine-tuning and usual training stage")
    parser.add_argument('--num_users', type=int, default=50, help="number of uses: K")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")
    parser.add_argument('--beta', type=float, default=0, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")

    # noise arguments
    parser.add_argument('--level_n_system', type=float, default=0.5, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.3, help="lower bound of noise level")

    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")

    # ablation study
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')

    # other arguments
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    parser.add_argument('--iid', type=bool, default=False, help="i.i.d. (True) or non-i.i.d. (False)")

    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    parser.add_argument('--alpha_dirichlet', type=float, default=1)

    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=13, help="random seed, default: 1")
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--alpha', type=float, default=1, help="0.1,1,5")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, default: 0")

    # Method-specific arguments
    parser.add_argument('--k_clusters', type=int, default=3, help="Number of sub-prototypes per class")
    parser.add_argument('--fairness_alpha', type=float, default=0.5, help="Alpha for balancing noise and contribution weights (0 to 1)")
    parser.add_argument('--exp_temp', type=float, default=0.1, help="Temperature for exponential weighting based on noise level")
    parser.add_argument('--beta_pseudo', type=float, default=0.5, help="Confidence threshold for pseudo-labeling in Stage 1")
    
    # === v8: Inter-Client Prototype Contrastive Learning ===
    parser.add_argument('--lambda_mcpcl', type=float, default=0.1, help="Weight for the Multi-Client Prototype Contrastive Loss")
    parser.add_argument('--temp_mcpcl', type=float, default=0.1, help="Temperature for the Multi-Client Prototype Contrastive Loss")

    # === v9: Affinity-based Adaptive Prototype Contrastive Learning ===
    parser.add_argument('--affinity_beta', type=float, default=1.0, help="Sensitivity parameter for affinity calculation")

    # === v10: Early Stopping ===
    parser.add_argument('--patience', type=int, default=1000, help="Early stopping patience rounds")



    return parser.parse_args()