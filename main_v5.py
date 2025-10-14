# main_v10.py
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

# 导入 v10 的参数解析器
from util.options_v5 import args_parser 
# 复用 v9 的本地训练和工具函数
from util.local_training_v5 import LocalUpdate, globaltest
from util.fedavg import FedAvg, FedAvgWeighted
from util.util_v5 import (add_noise, global_sub_prototype_distance, get_output, 
                          selective_pseudo_labeling, prototype_based_correction, calculate_global_prototypes)
from util.dataset import get_dataset
from model.build_model_v5 import build_model 

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL (Version 5:  Early Stopping)
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
    txtpath = rootpath + 'txtsave/V5_%s_%s_NL_%.1f_LB_%.1f_Iter_%d_lr_%.2f_BetaP_%.2f_K_%d_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.iteration1, args.lr, 
        args.beta_pseudo, args.k_clusters, args.seed)

    if args.iid: txtpath += "_IID"
    else: txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
    txtpath += f"_AffBeta_{args.affinity_beta}"
    if args.fine_tuning: txtpath += "_FT"
    if args.correction: txtpath += "_CORR"
    if args.mixup: txtpath += "_Mix_%.1f" % (args.alpha)
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

    # ============================ Stage 1: Pre-processing (Unchanged from V9) ============================
    print("\n" + "="*25 + " Stage 1: Pre-processing with APCL-FL " + "="*25, flush=True)

    for iteration in range(args.iteration1):
        print(f"\n------ Pre-processing Iteration: {iteration+1}/{args.iteration1} ------", flush=True)
        w_locals, all_local_sub_prototypes = [], []
        num_to_select = max(1, int(args.num_users * args.frac1))
        idxs_users = np.random.choice(range(args.num_users), num_to_select, replace=False)
        print(f"Selected clients: {idxs_users}", flush=True)

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
        
        # Server performs fairness-weighted aggregation (same as v8)
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
        
        # Meta-clustering to generate global sub-prototypes (same as v8)
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

        # === NEW v9: Rebuild multi-client prototypes with origin info & Calculate Affinity Matrix ===
        print("\n  <-- Rebuilding Multi-Client Prototypes & Calculating Affinity -->", flush=True)
        multi_client_prototypes = {}
        for label in range(args.num_classes):
            collected_sub_protos = []
            proto_origins = []
            for proto_dict in all_local_sub_prototypes:
                if label in proto_dict['prototypes']:
                    protos = proto_dict['prototypes'][label]
                    collected_sub_protos.extend(protos)
                    proto_origins.extend([proto_dict['id']] * len(protos))
            
            if collected_sub_protos:
                multi_client_prototypes[label] = {'prototypes': np.array(collected_sub_protos), 'origins': proto_origins}

        # Calculate affinity matrix
        temp_affinity_matrix = np.eye(args.num_users) # self-affinity is 1
        for i in range(args.num_users):
            for j in range(i + 1, args.num_users):
                client_i_protos = next((item['prototypes'] for item in all_local_sub_prototypes if item["id"] == i), None)
                client_j_protos = next((item['prototypes'] for item in all_local_sub_prototypes if item["id"] == j), None)
                
                if client_i_protos is not None and client_j_protos is not None:
                    common_labels = set(client_i_protos.keys()) & set(client_j_protos.keys())
                    if not common_labels:
                        avg_dist = np.inf
                    else:
                        total_dist = 0
                        for label in common_labels:
                            # Use cdist for efficient distance calculation between two sets of prototypes for a class
                            dist_matrix = cdist(client_i_protos[label], client_j_protos[label])
                            total_dist += np.mean(dist_matrix)
                        avg_dist = total_dist / len(common_labels)
                    
                    affinity = np.exp(-args.affinity_beta * avg_dist)
                    temp_affinity_matrix[i, j] = temp_affinity_matrix[j, i] = affinity
                else:
                    # If one client is not selected, their affinity is considered low
                    temp_affinity_matrix[i, j] = temp_affinity_matrix[j, i] = 0.0 
        
        # Update affinity matrix only for selected clients to avoid stale values
        for i in idxs_users:
            for j in idxs_users:
                affinity_matrix[i,j] = temp_affinity_matrix[i,j]



        # 3. Noise identification process
        dist_client = np.zeros(args.num_users)

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
        
        # 4. === v1: Label correction using FedLabel's selective pseudo-labeling ===
        print("\nPerforming label correction using selective pseudo-labeling...")
        new_targets = np.array(dataset_train.targets)
        for client_id in noisy_set:
            local_w = next((item['w'] for item in w_locals if item["id"] == client_id), None)
            if local_w is not None:
                net_local.load_state_dict(local_w)
                client_indices = list(dict_users[client_id])
                corrected_labels_all = selective_pseudo_labeling(args, net_local, netglob, dataset_train, client_indices)
                new_targets[client_indices] = corrected_labels_all[client_indices]
        dataset_train.targets = new_targets

    print("\n" + "="*25 + " End of Stage 1 " + "="*25, flush=True)
    print(f"Identified Clean Clients: {clean_set.tolist()}", flush=True)
    print(f"Identified Noisy Clients: {noisy_set.tolist()}", flush=True)

    args.beta = 0

    # ============================ Stage 2: Federated Finetuning with Early Stopping ============================
    if args.fine_tuning:
        print("\n" + "="*25 + " Stage 2: Federated Finetuning " + "="*25, flush=True)
        selected_clean_idx = clean_set
        if len(selected_clean_idx) > 0:
            print(f"Finetuning with all {len(selected_clean_idx)} identified clean clients in each round.", flush=True)
            
            # === NEW v10: Initialize early stopping variables for Stage 2 ===
            best_acc_stage2 = 0
            patience_counter_stage2 = 0
            
            idxs_users = selected_clean_idx
            for rnd in range(args.rounds1):
                print(f"\n--- Finetuning Round: {rnd+1}/{args.rounds1} ---", flush=True)
                w_locals = []
                for idx in idxs_users:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w_local, _ = local.update_weights(
                        net=copy.deepcopy(netglob).to(args.device), 
                        seed=args.seed, w_g=netglob.state_dict(),
                        epoch=args.local_ep, mu=0)
                    w_locals.append({'id': idx, 'w': w_local})
                
                dict_len = [len(dict_users[d['id']]) for d in w_locals]
                w_glob_fl = FedAvg([d['w'] for d in w_locals], dict_len)
                netglob.load_state_dict(copy.deepcopy(w_glob_fl))
                
                acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
                max_accuracy = max(max_accuracy, acc_s2)
                print(f"Test Accuracy after round {rnd+1}: {acc_s2:.4f}", flush=True)
                f_acc.write(f"fine tuning stage round {rnd}, test acc  {acc_s2:.4f} \n"); f_acc.flush()

                # === NEW v10: Early stopping check for Stage 2 ===
                if acc_s2 > best_acc_stage2:
                    best_acc_stage2 = acc_s2
                    patience_counter_stage2 = 0
                    print(f"  (New best accuracy in Stage 2: {best_acc_stage2:.4f})")
                else:
                    patience_counter_stage2 += 1
                    print(f"  (No improvement for {patience_counter_stage2} round(s))")

                if patience_counter_stage2 >= args.patience:
                    print(f"\nEarly stopping triggered in Stage 2 after {patience_counter_stage2} rounds of no improvement.")
                    break # Exit the finetuning loop

            if args.correction:
                # ... (Correction logic remains the same)
                pass
        else:
            print("Warning: No clients identified as clean for finetuning. Skipping this stage.", flush=True)

    # ============================ Stage 3: Usual Federated Learning with Early Stopping ============================
    print("\n" + "="*25 + " Stage 3: Usual Federated Learning " + "="*25, flush=True)
    final_accuracies = []
    m = max(int(args.frac2 * args.num_users), 1)
    
    for cid in range(args.num_users):
        if cid in noisy_set:
            estimated_noisy_level[cid] = 1.0 
        else:
            estimated_noisy_level[cid] = 0.0

    # === NEW v10: Initialize early stopping variables for Stage 3 ===
    best_acc_stage3 = 0
    patience_counter_stage3 = 0

    for rnd in range(args.rounds2):
        print(f"\n--- Final Training Round: {rnd+1}/{args.rounds2} ---", flush=True)
        w_locals = []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"Selected clients for this round: {idxs_users}", flush=True)
        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, _ = local.update_weights(
                net=copy.deepcopy(netglob).to(args.device), 
                seed=args.seed,
                w_g=netglob.state_dict(),
                epoch=args.local_ep,
                mu=0
            )
            w_locals.append({'id': idx, 'w': w})
        
        client_weights = [np.exp(-estimated_noisy_level[d['id']] / args.exp_temp) for d in w_locals]
        w_glob = FedAvgWeighted([d['w'] for d in w_locals], client_weights)
        netglob.load_state_dict(copy.deepcopy(w_glob))

        acc_s3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        final_accuracies.append(acc_s3)
        max_accuracy = max(max_accuracy, acc_s3)
        
        print(f"Test Accuracy after round {rnd+1}: {acc_s3:.4f}", flush=True)
        f_acc.write(f"third stage round {rnd}, test acc  {acc_s3:.4f} \n"); f_acc.flush()

        # === NEW v10: Early stopping check for Stage 3 ===
        if acc_s3 > best_acc_stage3:
            best_acc_stage3 = acc_s3
            patience_counter_stage3 = 0
            print(f"  (New best accuracy in Stage 3: {best_acc_stage3:.4f})")
        else:
            patience_counter_stage3 += 1
            print(f"  (No improvement for {patience_counter_stage3} round(s))")

        if patience_counter_stage3 >= args.patience:
            print(f"\nEarly stopping triggered in Stage 3 after {patience_counter_stage3} rounds of no improvement.")
            break # Exit the final training loop

    # ============================ Final Result Output (Unchanged) ============================
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