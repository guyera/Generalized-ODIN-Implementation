# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc
from sklearn.metrics import roc_auc_score


def tpr95(name):
    #calculate the falsepositive error when tpr is 95%
    # calculate baseline
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 1 
    if name == "CIFAR-100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fpr_base = fpr/total

    # calculate our algorithm
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR-100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fpr_new = fpr/total
            
    return fpr_base, fpr_new

def auroc_orig(name):
    #calculate the AUROC
    """
    # calculate baseline
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 1 
    if name == "CIFAR-100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auroc_base = 0.0
    fpr_temp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        auroc_base += (-fpr+fpr_temp)*tpr
        fpr_temp = fpr
    auroc_base += fpr * tpr
    """
    # calculate our algorithm
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR-100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auroc_new = 0.0
    fpr_temp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        auroc_new += (-fpr+fpr_temp)*tpr
        fpr_temp = fpr
    auroc_new += fpr * tpr
    return auroc_new
    # return auroc_base, auroc_new

def auroc(name):
    #calculate the AUROC
    # calculate baseline
    """
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    scores = np.concatenate((Y1, X1))
    trues = np.array(([0] * len(Y1)) + ([1] * len(X1)))
    auroc_base = roc_auc_score(trues, scores)
    """
    # calculate our algorithm
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    scores = np.concatenate((Y1, X1))
    trues = np.array(([0] * len(Y1)) + ([1] * len(X1)))
    auroc_new = roc_auc_score(trues, scores)
    return auroc_new
    # return auroc_base, auroc_new

def aupr_in(name):
    #calculate the AUPR
    # calculate baseline
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 1 
    if name == "CIFAR-100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    precision_vec = []
    recallVec = []
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aupr_base = 0.0
    recall_temp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precision_vec.append(precision)
        recallVec.append(recall)
        aupr_base += (recall_temp-recall)*precision
        recall_temp = recall
    aupr_base += recall * precision
    #print(recall, precision)

    # calculate our algorithm
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR-100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aupr_new = 0.0
    recall_temp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        #precision_vec.append(precision)
        #recallVec.append(recall)
        aupr_new += (recall_temp-recall)*precision
        recall_temp = recall
    aupr_new += recall * precision
    return aupr_base, aupr_new

def aupr_out(name):
    #calculate the AUPR
    # calculate baseline
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 1 
    if name == "CIFAR-100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aupr_base = 0.0
    recall_temp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        aupr_base += (recall_temp-recall)*precision
        recall_temp = recall
    aupr_base += recall * precision
        
    
    # calculate our algorithm
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR-100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    aupr_new = 0.0
    recall_temp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        aupr_new += (recall_temp-recall)*precision
        recall_temp = recall
    aupr_new += recall * precision
    return aupr_base, aupr_new



def detection(name):
    #calculate the minimum detection error
    # calculate baseline
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 1 
    if name == "CIFAR-100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    error_base = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        error_base = np.minimum(error_base, (tpr+error2)/2.0)

    # calculate our algorithm
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR-10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR-100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    error_new = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        error_new = np.minimum(error_new, (tpr+error2)/2.0)
            
    return error_base, error_new




def metric(nn, data):
    indis = "CIFAR-10"
    if data == "Imagenet": dataName = "Tiny-ImageNet (crop)"
    if data == "Imagenet_resize": dataName = "Tiny-ImageNet (resize)"
    if data == "LSUN": dataName = "LSUN (crop)"
    if data == "LSUN_resize": dataName = "LSUN (resize)"
    if data == "iSUN": dataName = "iSUN"
    if data == "Gaussian": dataName = "Gaussian noise"
    if data == "Uniform": dataName = "Uniform Noise"
    #fpr_base, fpr_new = tpr95(indis)
    #error_base, error_new = detection(indis)
    auc_orig = auroc_orig(indis)
    auc = auroc(indis)
    #aupr_in_base, aupr_in_new = aupr_in(indis)
    #aupr_out_base, aupr_out_new = aupr_out(indis)
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    # print("{:>34}{:>19}".format("Baseline", "Our Method"))
    #print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:",fpr_base*100, fpr_new*100))
    #print("{:20}{:13.1f}%{:>18.1f}%".format("Detection error:",error_base*100, error_new*100))
    print("AUROC (Original):", auc_orig*100)
    print("AUROC:", auc*100)
    #print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:",aupr_in_base*100, aupr_in_new*100))
    #print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:",aupr_out_base*100, aupr_out_new*100))










