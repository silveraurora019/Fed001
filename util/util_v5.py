# util/util_v9.py
import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist

# Functions add_noise, get_output, global_sub_prototype_distance, selective_pseudo_labeling, 
# calculate_global_prototypes, prototype_based_correction remain the same as in util_v8.py.

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
        print(f"Client {i}, noise level: {gamma_c[i]:.4f}, real noise ratio: {noise_ratio:.4f}")
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)

def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    all_outputs = []
    all_loss = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels_ = images.to(args.device), labels.to(args.device).long()
            
            features = net(images, latent_output=True)
            final_outputs = net(images)
            if criterion is not None:
                loss = criterion(final_outputs, labels_)
                all_loss.append(loss.cpu().numpy())

            if latent:
                outputs = features
            else:
                outputs = F.softmax(final_outputs, dim=1)
            
            all_outputs.append(outputs.cpu().numpy())

    output_whole = np.concatenate(all_outputs) if len(all_outputs) > 0 else np.array([])
    loss_whole = np.concatenate(all_loss) if len(all_loss) > 0 else np.array([])
    
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole, None

def global_sub_prototype_distance(latent_representations, labels, global_sub_prototypes):
    distances = np.zeros(len(labels))
    for i in range(len(latent_representations)):
        label = labels[i]
        if label in global_sub_prototypes:
            representation = latent_representations[i]
            sub_prototypes = global_sub_prototypes[label]
            dists_to_subs = cdist(representation.reshape(1, -1), sub_prototypes)
            distances[i] = np.min(dists_to_subs)
        else:
            distances[i] = np.nan 
    return distances

def selective_pseudo_labeling(args, local_net, global_net, client_data, client_indices):
    local_net.eval()
    global_net.eval()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(client_data, client_indices), batch_size=100, shuffle=False)
    local_outputs, _ = get_output(loader, local_net.to(args.device), args, False, None)
    global_outputs, _ = get_output(loader, global_net.to(args.device), args, False, None)
    corrected_labels = np.array(client_data.targets)

    corrected_count = 0

    for i in range(len(client_indices)):
        local_pred_probs = local_outputs[i]
        global_pred_probs = global_outputs[i]

        local_confidence = np.var(local_pred_probs)
        global_confidence = np.var(global_pred_probs)

        if local_confidence > global_confidence:
            selected_probs = local_pred_probs
        else:
            selected_probs = global_pred_probs
        
        if np.max(selected_probs) > args.beta_pseudo:
            pseudo_label = np.argmax(selected_probs)
            original_label_index = client_indices[i]

            if corrected_labels[original_label_index] != pseudo_label:
                corrected_labels[original_label_index] = pseudo_label
                corrected_count += 1
    
    if corrected_count > 0:
        print(f"  - Corrected {corrected_count} labels.")

    return corrected_labels

def calculate_global_prototypes(net, clean_data_subset, args):
    net.eval()
    global_prototypes = {}
    loader = torch.utils.data.DataLoader(clean_data_subset, batch_size=100, shuffle=False)
    
    latent_representations, _ = get_output(loader, net.to(args.device), args, latent=True, criterion=None)
    labels = np.array(clean_data_subset.dataset.targets)[clean_data_subset.indices]

    for label in np.unique(labels):
        class_mask = (labels == label)
        class_reps = latent_representations[class_mask]
        if class_reps.shape[0] > 0:
            global_prototypes[label] = np.mean(class_reps, axis=0)
            
    return global_prototypes

def prototype_based_correction(net, noisy_data_subset, global_prototypes, args):
    net.eval()
    
    loader = torch.utils.data.DataLoader(noisy_data_subset, batch_size=100, shuffle=False)
    
    # Get latent representations and final predictions
    latent_reps, _ = get_output(loader, net.to(args.device), args, latent=True, criterion=None)
    final_outputs, _ = get_output(loader, net.to(args.device), args, latent=False, criterion=None)
    
    global_pred_labels = np.argmax(final_outputs, axis=1)

    corrected_labels = np.array(noisy_data_subset.dataset.targets)
    corrected_count = 0
    
    proto_labels = list(global_prototypes.keys())
    proto_vectors = np.array(list(global_prototypes.values()))

    for i in range(len(latent_reps)):
        sample_rep = latent_reps[i].reshape(1, -1)
        
        # Calculate L2 norm (Euclidean distance) to all global prototypes
        distances = cdist(sample_rep, proto_vectors)
        
        # Find the label of the closest prototype
        min_dist_idx = np.argmin(distances)
        prototype_predicted_label = proto_labels[min_dist_idx]
        
        # Get the model's direct prediction
        global_predicted_label = global_pred_labels[i]

        # Check if both predictions match
        if prototype_predicted_label == global_predicted_label:
            original_label_index = noisy_data_subset.indices[i]
            if corrected_labels[original_label_index] != global_predicted_label:
                corrected_labels[original_label_index] = global_predicted_label
                corrected_count += 1
    
    print(f"  - Corrected {corrected_count} labels based on prototype agreement.")
    
    return corrected_labels


# === NEW v9: Affinity-based Adaptive Multi-Client Prototype Contrastive Loss ===
def adaptive_mcpcl_loss(features, labels, multi_client_prototypes, affinity_vector, temp, device):
    """
    Calculates the Affinity-based Adaptive Multi-Client Prototype Contrastive Loss.
    """
    if not multi_client_prototypes:
        return torch.tensor(0.0).to(device)

    # Normalize features
    features = F.normalize(features, p=2, dim=1)
    
    # Prepare all prototypes and their origins
    all_prototypes = []
    proto_labels = []
    proto_origins = [] # NEW: Keep track of which client a prototype comes from
    
    for label, protos_info in multi_client_prototypes.items():
        protos = protos_info['prototypes']
        origins = protos_info['origins']
        protos_tensor = torch.tensor(np.array(protos), dtype=torch.float32).to(device)
        protos_tensor = F.normalize(protos_tensor, p=2, dim=1)
        all_prototypes.append(protos_tensor)
        proto_labels.extend([label] * protos_tensor.shape[0])
        proto_origins.extend(origins)

    if not all_prototypes:
        return torch.tensor(0.0).to(device)
        
    all_prototypes = torch.cat(all_prototypes, dim=0)
    proto_labels = torch.tensor(proto_labels).to(device)
    proto_origins = torch.tensor(proto_origins).to(device)

    # Calculate raw similarity matrix (logits)
    logits = torch.matmul(features, all_prototypes.T)

    # === NEW: Create affinity weights for logits ===
    # affinity_vector is (num_users,), proto_origins is (num_prototypes,)
    # We use advanced indexing to create a weight matrix for broadcasting
    affinity_weights = affinity_vector[proto_origins] # Shape: (num_prototypes,)
    affinity_weights = affinity_weights.unsqueeze(0).expand(features.shape[0], -1) # Shape: (batch_size, num_prototypes)
    
    # Apply affinity weights and temperature
    weighted_logits = (logits * affinity_weights) / temp
    
    # Create mask for positive pairs
    labels_reshaped = labels.view(-1, 1)
    proto_labels_reshaped = proto_labels.view(1, -1)
    pos_mask = (labels_reshaped == proto_labels_reshaped).float()
    
    # Handle cases with no positive prototypes for a given sample
    if pos_mask.sum(1).min() == 0:
        return torch.tensor(0.0).to(device)

    # Calculate log probabilities using the weighted logits
    log_prob = weighted_logits - torch.log(torch.exp(weighted_logits).sum(1, keepdim=True))
    
    # Mean log-likelihood for positive pairs
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

    loss = -mean_log_prob_pos.mean()
    
    return loss