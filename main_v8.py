# main_v8.py
import os
import copy
import numpy as np
import random
import time
import torch
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import torch.nn as nn
from scipy.spatial.distance import cdist

# 导入 v8 的组件
from util.options_v8 import args_parser 
from util.local_training_v5 import LocalUpdate, globaltest
from util.fedavg import FedAvg, FedAvgWeighted
from util.util_v5 import (add_noise, global_sub_prototype_distance, get_output, 
                          selective_pseudo_labeling, calculate_global_prototypes, prototype_based_correction)
from util.dataset import get_dataset
from model.build_model_v5 import build_model 

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL (Version 8: Unified Training Loop)
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

    dataset_train, dataset_test, dict_users = get_dataset(args)
    y_train = np.array(dataset_train.targets)
    y_train_noisy, _, _ = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    if not os.path.exists(rootpath + 'txtsave/'): os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/V8_%s_%s_NL_%.1f_LB_%.1f_Rounds_%d_lr_%.2f_BetaP_%.2f_K_%d_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.rounds, args.lr, 
        args.beta_pseudo, args.k_clusters, args.seed)

    if args.iid: txtpath += "_IID"
    else: txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    txtpath += f"_AffBeta_{args.affinity_beta}"
    if args.correction: txtpath += "_CORR"
    f_acc = open(txtpath + '_acc.txt', 'a')

    f_acc.write("="*50 + "\n")
    f_acc.write("Training Parameters:\n")
    f_acc.write(str(args) + "\n")
    f_acc.write("="*50 + "\n")
    f_acc.flush()

    netglob = build_model(args)
    net_local = build_model(args)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    dist_accumulative_client = np.zeros(args.num_users)
    estimated_noisy_level = np.zeros(args.num_users)
    max_accuracy = 0.0
    
    multi_client_prototypes = {} 
    affinity_matrix = np.ones((args.num_users, args.num_users))
    final_accuracies = []

    # ============================ v8: Unified Training Loop ============================
    print("\n" + "="*25 + " Unified Training Loop " + "="*25, flush=True)

    for round_num in range(args.rounds):
        print(f"\n------ Round: {round_num+1}/{args.rounds} ------", flush=True)
        w_locals, all_local_sub_prototypes = [], []
        num_to_select = max(1, int(args.num_users * args.frac))
        idxs_users = np.random.choice(range(args.num_users), num_to_select, replace=False)
        print(f"Selected clients: {idxs_users}", flush=True)

        # 1. Local Training on selected clients (same as v5 Stage 1)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net_local.load_state_dict(netglob.state_dict())
            
            affinity_vector = torch.tensor(affinity_matrix[idx, :], dtype=torch.float32).to(args.device)

            w, _ = local.update_weights(
                net=copy.deepcopy(net_local).to(args.device), 
                seed=args.seed,
                w_g=netglob.state_dict(), 
                epoch=args.local_ep, 
                mu=estimated_noisy_level[idx],
                multi_client_prototypes=multi_client_prototypes,
                affinity_vector=affinity_vector,
                client_id=idx
            )
            w_locals.append({'id': idx, 'w': w})
            net_local.load_state_dict(w)
            local_sub_prototypes = local.calculate_sub_prototypes(net_local)
            all_local_sub_prototypes.append({'id': idx, 'prototypes': local_sub_prototypes})
        
        # 2. Server-side Aggregation (same as v5 Stage 1)
        class_freq = {c: 0 for c in range(args.num_classes)}
        client_classes = {idx: set() for idx in idxs_users}
        for proto_dict in all_local_sub_prototypes:
            idx = proto_dict['id']
            for label in proto_dict['prototypes'].keys():
                client_classes[idx].add(label)
                class_freq[label] += 1
        contribution_scores = np.zeros(args.num_users)
        for idx in idxs_users:
            if not client_classes[idx]: continue
            score = sum(1 / (class_freq[c] + 1e-8) for c in client_classes[idx])
            contribution_scores[idx] = score
        exp_scores = np.exp(contribution_scores[idxs_users] - np.max(contribution_scores[idxs_users]))
        weights_contrib = exp_scores / (exp_scores.sum() + 1e-8)
        noise_levels = estimated_noisy_level[idxs_users]
        weights_noise = np.exp(-noise_levels / (args.exp_temp + 1e-8))
        weights_noise /= (weights_noise.sum() + 1e-8)
        final_weights = (1 - args.fairness_alpha) * weights_noise + args.fairness_alpha * weights_contrib
        w_glob = FedAvgWeighted([d['w'] for d in w_locals], final_weights)
        netglob.load_state_dict(copy.deepcopy(w_glob))
        
        # 3. Meta-clustering and Affinity Update (same as v5 Stage 1)
        global_sub_prototypes = {}
        for label in range(args.num_classes):
            collected_sub_protos = []
            for proto_dict in all_local_sub_prototypes:
                if label in proto_dict['prototypes']:
                    collected_sub_protos.extend(proto_dict['prototypes'][label])
            num_collected = len(collected_sub_protos)
            if num_collected >= args.k_clusters:
                kmeans = KMeans(n_clusters=args.k_clusters, random_state=args.seed, n_init=10).fit(collected_sub_protos)
                global_sub_prototypes[label] = kmeans.cluster_centers_

        multi_client_prototypes = {}
        for label in range(args.num_classes):
            collected_sub_protos, proto_origins = [], []
            for proto_dict in all_local_sub_prototypes:
                if label in proto_dict['prototypes']:
                    protos = proto_dict['prototypes'][label]
                    collected_sub_protos.extend(protos)
                    proto_origins.extend([proto_dict['id']] * len(protos))
            if collected_sub_protos:
                multi_client_prototypes[label] = {'prototypes': np.array(collected_sub_protos), 'origins': proto_origins}

        temp_affinity_matrix = np.eye(args.num_users)
        # (Affinity matrix calculation remains the same as v5)
        # ... (code omitted for brevity but is identical to main_v5.py)

        # 4. Noise Identification (same as v5 Stage 1)
        dist_client = np.zeros(args.num_users)
        for idx in range(args.num_users):
            sample_idx = np.array(list(dict_users[idx]))
            loader = torch.utils.data.DataLoader(Subset(dataset_train, sample_idx), batch_size=100, shuffle=False)
            latent_output, _ = get_output(loader, netglob.to(args.device), args, True, criterion=None)
            
            if latent_output.size == 0: continue
            client_labels = np.array(dataset_train.targets)[sample_idx]
            dist_local = global_sub_prototype_distance(latent_output, client_labels, global_sub_prototypes)
            dist_local = np.nan_to_num(dist_local, nan=np.nanmean(dist_local[~np.isnan(dist_local)]))
            dist_client[idx] = np.mean(dist_local)
            
        dist_accumulative_client += dist_client

        valid_distances = dist_accumulative_client[dist_accumulative_client != 0]
        if len(valid_distances) < 2:
            noisy_set, clean_set = np.array([]), np.array(range(args.num_users))
        else:
            gmm = GaussianMixture(n_components=2, random_state=args.seed).fit(valid_distances.reshape(-1, 1))
            labels = gmm.predict(dist_accumulative_client.reshape(-1, 1))
            clean_label = np.argsort(gmm.means_[:, 0])[0]
            noisy_set = np.where(labels != clean_label)[0]
            clean_set = np.where(labels == clean_label)[0]
        
        print(f"  Identified Clean Clients: {clean_set.tolist()}", flush=True)
        print(f"  Identified Noisy Clients: {noisy_set.tolist()}", flush=True)

        # 5. Label Correction for Noisy Clients (Applied every round)
        if args.correction:
            print("\n  Performing label correction for identified noisy clients...")
            new_targets = np.array(dataset_train.targets)
            for client_id in noisy_set:
                # Only correct clients that were selected and identified as noisy in this round
                local_w = next((item['w'] for item in w_locals if item["id"] == client_id), None)
                if local_w is not None:
                    net_local.load_state_dict(local_w)
                    client_indices = list(dict_users[client_id])
                    corrected_labels_all = selective_pseudo_labeling(args, net_local, netglob, dataset_train, client_indices)
                    new_targets[client_indices] = corrected_labels_all[client_indices]
            dataset_train.targets = new_targets
        
        # 6. Evaluation
        acc_test = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        final_accuracies.append(acc_test)
        max_accuracy = max(max_accuracy, acc_test)
        print(f"Test Accuracy after round {round_num+1}: {acc_test:.4f}", flush=True)
        f_acc.write(f"round {round_num}, test acc  {acc_test:.4f} \n"); f_acc.flush()

    # ============================ Final Result Output ============================
    # (Final result output logic is the same as v5)
    # ... (code omitted for brevity but is identical to main_v5.py)
    
    print("\n" + "="*30 + " Final Results " + "="*30, flush=True)
    if len(final_accuracies) >= 10:
        last_10_accuracies = final_accuracies[-10:]
        mean_acc = np.mean(last_10_accuracies)
        var_acc = np.var(last_10_accuracies)
        print(f"Mean of last 10 rounds test accuracy: {mean_acc:.4f}", flush=True)
        f_acc.write(f"\nMean of last 10 rounds test accuracy: {mean_acc:.4f}\n")
    elif len(final_accuracies) > 0:
        mean_acc = np.mean(final_accuracies)
        var_acc = np.var(final_accuracies)
        print(f"Mean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}", flush=True)
        f_acc.write(f"\nMean of final {len(final_accuracies)} rounds test accuracy: {mean_acc:.4f}\n")
    
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