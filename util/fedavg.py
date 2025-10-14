# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
import torch


def FedAvg(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0] 
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def FedAvgWeighted(w, weights):
    """
    Performs weighted averaging for model parameters.
    Args:
        w: list of model state_dicts.
        weights: list of weights corresponding to each model.
    """
    if sum(weights) == 0:
        # Fallback to standard FedAvg if all weights are zero
        print("Warning: All aggregation weights are zero. Falling back to standard FedAvg.")
        dict_len = [1] * len(w) # Assume equal data size
        return FedAvg(w, dict_len)

    # Normalize weights to sum to 1
    weights_norm = [float(i) / sum(weights) for i in weights]

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys(): # k is the layer name (e.g., 'conv1.weight')
        # Initialize with the first weighted model
        w_avg[k] = w[0][k] * weights_norm[0]
        # Accumulate the rest
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * weights_norm[i]
            
    return w_avg
