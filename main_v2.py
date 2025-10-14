# main_v4.py
# python version 3.7.1
# -*- coding: utf-8 -*-

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

from util.options_v2 import args_parser
from util.local_training_v2 import LocalUpdate, globaltest
from util.fedavg import FedAvg, FedAvgWeighted
from util.util_v2 import (add_noise, global_sub_prototype_distance, get_output, 
                          selective_pseudo_labeling, calculate_global_prototypes, prototype_based_correction)
from util.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL (Version 2: Prototype-based correction in Stage 2)
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
    txtpath = rootpath + 'txtsave/V2_%s_%s_NL_%.1f_LB_%.1f_Iter_%d_lr_%.2f_BetaP_%.2f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.iteration1, args.lr, 
        args.beta_pseudo, args.seed)

    if args.iid: txtpath += "_IID"
    else: txtpath += "_nonIID_p_%.1f_dirich_%.1f"%(args.non_iid_prob_class,args.alpha_dirichlet)
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

    print("\n" + "="*25 + " Stage 1: Pre-processing (v4) " + "="*25, flush=True)
    # ... (Stage 1 remains unchanged)
    for iteration in range(args.iteration1):
        print(f"\n------ Pre-processing Iteration: {iteration+1}/{args.iteration1} ------", flush=True)
        w_locals, all_local_sub_prototypes = [], []
        num_to_select = max(1, int(args.num_users * args.frac1))
        idxs_users = np.random.choice(range(args.num_users), num_to_select, replace=False)
        print(f"Selected clients: {idxs_users}", flush=True)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            net_local.load_state_dict(netglob.state_dict())
            w, _ = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                        w_g=netglob.state_dict(), epoch=args.local_ep, mu=estimated_noisy_level[idx])
            w_locals.append({'id': idx, 'w': w})
            net_local.load_state_dict(w)
            local_sub_prototypes = local.calculate_sub_prototypes(net_local)
            all_local_sub_prototypes.append({'id': idx, 'prototypes': local_sub_prototypes})
        print("\n  <-- Server Aggregation (Fairness-aware) -->", flush=True)
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

    # ============================ Stage 2: Federated Finetuning ============================
    if args.fine_tuning:
        print("\n" + "="*25 + " Stage 2: Federated Finetuning " + "="*25, flush=True)
        
        selected_clean_idx = clean_set
        
        if len(selected_clean_idx) > 0:
            print(f"Finetuning with all {len(selected_clean_idx)} identified clean clients in each round.", flush=True)
            
            idxs_users = selected_clean_idx
            
            for rnd in range(args.rounds1):
                print(f"\n--- Finetuning Round: {rnd+1}/{args.rounds1} ---", flush=True)
                w_locals = []
                # print(f"Selected clients for this round: {idxs_users}", flush=True)
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
            
            # === MODIFIED CODE: v4 Label correction after finetuning ===
            if args.correction:
                print("\nPerforming label correction after finetuning using prototype agreement...", flush=True)
                
                # 1. Create a dataset of all clean clients
                all_clean_indices = []
                for idx in clean_set:
                    all_clean_indices.extend(list(dict_users[idx]))
                clean_subset = Subset(dataset_train, all_clean_indices)
                
                # 2. Calculate global prototypes using the fine-tuned model on clean data
                global_prototypes = calculate_global_prototypes(netglob, clean_subset, args)
                
                # 3. Correct labels on noisy clients
                y_train_corrected = np.array(dataset_train.targets)
                for idx in noisy_set:
                    noisy_indices = list(dict_users[idx])
                    noisy_subset = Subset(dataset_train, noisy_indices)
                    
                    # This function returns the corrected labels for the entire dataset
                    corrected_labels_all = prototype_based_correction(netglob, noisy_subset, global_prototypes, args)
                    
                    # Only update the labels for the current noisy client
                    y_train_corrected[noisy_indices] = corrected_labels_all[noisy_indices]
                
                dataset_train.targets = y_train_corrected

        else:
            print("Warning: No clients identified as clean for finetuning. Skipping this stage.", flush=True)

    # ============================ Stage 3: Usual Federated Learning ============================
    print("\n" + "="*25 + " Stage 3: Usual Federated Learning " + "="*25, flush=True)
    final_accuracies = []
    m = max(int(args.frac2 * args.num_users), 1)
    
    for cid in range(args.num_users):
        if cid in noisy_set:
            estimated_noisy_level[cid] = 1.0 
        else:
            estimated_noisy_level[cid] = 0.0

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