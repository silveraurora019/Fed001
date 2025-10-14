import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist

def add_noise(args, y_train, dict_users):
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
        print(f"Client {i}, noise level: {gamma_c[i]:.4f} ({gamma_c[i] * 0.9:.4f}), real noise ratio: {noise_ratio:.4f}")
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)

def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    all_outputs = []
    all_labels = []
    all_loss = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device).long()
            
            # Get final outputs for loss calculation
            final_outputs = net(images, False)
            if criterion is not None:
                loss = criterion(final_outputs, labels)
                all_loss.append(loss.cpu().numpy())

            # Get the required feature layer output
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


# === v222: Function to calculate distance based on global sub-prototypes ===
def global_sub_prototype_distance(latent_representations, labels, global_sub_prototypes):
    distances = np.zeros(len(labels))
    for i in range(len(latent_representations)):
        label = labels[i]
        if label in global_sub_prototypes:
            representation = latent_representations[i]
            # Get all global sub-prototypes for this class
            sub_prototypes = global_sub_prototypes[label]
            # Calculate the distance to all sub-prototypes and take the minimum
            dists_to_subs = cdist(representation.reshape(1, -1), sub_prototypes)
            distances[i] = np.min(dists_to_subs)
        else:
            # If a label has no global prototype, set the distance to nan
            distances[i] = np.nan 
    return distances

# === v1: Selective pseudo-labeling inspired by FedLabel ===
def selective_pseudo_labeling(args, local_net, global_net, client_data, client_indices):
    """
    Generates pseudo-labels for a client's data by selectively choosing between the local and global models.
    """
    local_net.eval()
    global_net.eval()

    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(client_data, client_indices), batch_size=100, shuffle=False)
    
    # Get predictions from both models
    local_outputs, _ = get_output(loader, local_net.to(args.device), args, False, None)
    global_outputs, _ = get_output(loader, global_net.to(args.device), args, False, None)

    corrected_labels = np.array(client_data.targets)
    corrected_count = 0

    for i in range(len(client_indices)):
        local_pred_probs = local_outputs[i]
        global_pred_probs = global_outputs[i]

        # Use variance as the confidence score, as mentioned in the FedLabel paper
        local_confidence = np.var(local_pred_probs)
        global_confidence = np.var(global_pred_probs)

        # Select the model with higher confidence
        if local_confidence > global_confidence:
            selected_probs = local_pred_probs
        else:
            selected_probs = global_pred_probs
        
        # Apply thresholding to generate pseudo-label
        if np.max(selected_probs) > args.beta_pseudo:
            pseudo_label = np.argmax(selected_probs)
            original_label_index = client_indices[i]

            if corrected_labels[original_label_index] != pseudo_label:
                corrected_labels[original_label_index] = pseudo_label
                corrected_count += 1
    
    if corrected_count > 0:
        print(f"  - Corrected {corrected_count} labels.")

    return corrected_labels