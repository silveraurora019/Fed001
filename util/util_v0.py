# util/util_v0.py
import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist

def add_noise(args, y_train, dict_users):
    # (此函数与 v1-v7 版本中的 add_noise 相同)
    np.random.seed(args.seed)
    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)
    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print(f"Client {i}, noise level: {gamma_c[i]:.4f}, real noise ratio: {noise_ratio:.4f}")
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)

def get_output(loader, net, args, latent=False, criterion=None):
    # (此函数与 v1-v7 版本中的 get_output 相同)
    net.eval()
    all_outputs = []
    all_labels = []
    all_loss = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device).long()
            
            final_outputs = net(images, False)
            if criterion is not None:
                loss = criterion(final_outputs, labels)
                all_loss.append(loss.cpu().numpy())

            if latent:
                outputs = net(images, True)
            else:
                outputs = F.softmax(final_outputs, dim=1)
            
            all_outputs.append(outputs.cpu().numpy())

    output_whole = np.concatenate(all_outputs) if len(all_outputs) > 0 else np.array([])
    loss_whole = np.concatenate(all_loss) if len(all_loss) > 0 else np.array([])
    
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole, None

# (移除了 global_sub_prototype_distance)
# (移除了 selective_pseudo_labeling)
# (移除了 calculate_global_prototypes)
# (移除了 prototype_based_correction)
# (移除了 adaptive_mcpcl_loss)