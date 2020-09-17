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

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import argparse

from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

from torch.utils.data import DataLoader, Subset

from densenet import DenseNet3
from deconfnet import DeconfNet, CosineDeconf, InnerDeconf, EuclideanDeconf

from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])


h_dict = {
    "cosine":   CosineDeconf,
    "inner":    InnerDeconf,
    "euclid":   EuclideanDeconf
    }



def kliep_loss(logits, labels, max_ratio=50):
    softplus = nn.Softplus()
    logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
    
    #preds  = torch.softmax(logits,dim=1)
    preds  = softplus(logits)
    #preds  = torch.sigmoid(logits) * 10

    maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
    
    y = torch.eye(preds.size(1))
    labels = y[labels].to(preds.device)

    inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
    outlier_loss = ((1-labels) * (preds)).mean(1)
    loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

    return loss

losses_dict = {
        'ce':nn.CrossEntropyLoss(),
        'kliep': kliep_loss, 
    }

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    parser.add_argument('--out-dataset', default = "Imagenet", type = str,
                        help = 'out-of-distribution dataset')
    parser.add_argument('--h-type', default = "cosine", type = str,
                        help = 'cosine|inner|euclid')
    parser.add_argument('--loss-type', default = "ce", type = str,
                        help = 'ce|kliep')
    parser.add_argument('--gpu', default = 0, type = int,
              help = 'gpu index')
    parser.add_argument('--magnitudes', nargs = '+', default = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08], type = float,
                        help = 'perturbation magnitudes')
    parser.add_argument('--epochs', default = 300, type = int,
               help = 'gpu index')
    parser.add_argument('-bs', '--batch_size', default = 64, type = int,
               help = 'gpu index')
    parser.add_argument('--eval-only', action='store_true',
                help='Load a model from file')
    parser.set_defaults(argument=True)
    return parser.parse_args()



def get_datasets(data_name, batchsize):

    train_set_in = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    test_set_in  = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    outlier_set  = torchvision.datasets.ImageFolder("./data/{}".format(data_name), transform=transform)
    
    test_indices      = list(range(len(test_set_in)))
    validation_set_in = Subset(test_set_in, test_indices[:1000])
    test_set_in       = Subset(test_set_in, test_indices[1000:])


    train_loader_in      =  DataLoader(train_set_in,      batch_size=batchsize, shuffle=True,  num_workers=4)
    validation_loader_in =  DataLoader(validation_set_in, batch_size=batchsize, shuffle=False, num_workers=4)
    test_loader_in       =  DataLoader(test_set_in,       batch_size=batchsize, shuffle=False, num_workers=4)
    outlier_loader       =  DataLoader(outlier_set,       batch_size=batchsize, shuffle=False, num_workers=4)

    return train_loader_in, validation_loader_in, test_loader_in, outlier_loader


def main():
    args = get_args()

    data_name        = args.out_dataset
    device           = args.gpu
    noise_magnitudes = args.magnitudes 
    epochs           = args.epochs
    h_type           = args.h_type
    eval_only        = args.eval_only
    batchsize        = args.batch_size

    #def __init__(self, depth, num_classes, growth_rate=12,
    #             reduction=0.5, bottleneck=True, dropRate=0.0):
    dense_net = DenseNet3(depth = 100, num_classes = 10).to(device)

    # Construct g, h, and the composed deconf net
    h = h_dict[h_type](dense_net.in_planes, 10).to(device)
    deconf_net = DeconfNet(dense_net, dense_net.in_planes, 10, h).to(device)

    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        if name == "h.h.weight" or name == "h.h.bias":
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    optimizer = optim.SGD(parameters, lr = 0.1, momentum = 0.9, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    h_optimizer = optim.SGD(h_parameters, lr = 0.1, momentum = 0.9) # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)

    #get outlier data
    train_data, val_data, test_data, open_data =get_datasets(data_name, batchsize)
  
    criterion = losses_dict[args.loss_type]

    # Train the model
    if not eval_only:
        deconf_net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_data)):
                inputs = inputs.to(device)
                targets = targets.to(device)
                h_optimizer.zero_grad()
                optimizer.zero_grad()

                logits, _, _ = deconf_net(inputs)
                loss = criterion(logits, targets)
                loss.backward()

                optimizer.step()
                h_optimizer.step()
                total_loss += loss.item()

            print("Epoch{} Loss: {}".format(epoch + 1, total_loss))
            h_scheduler.step()
            scheduler.step()
        torch.save(deconf_net.state_dict(), f"desnet{epochs}_{h_type}_{args.loss_type}.pth")
    else:
        deconf_net.load_state_dict(torch.load(f"desnet{epochs}_{h_type}_{args.loss_type}.pth"))


    deconf_net.eval()
    best_val_score = 0
    best_auc = 0
    for noise_magnitude in noise_magnitudes:

        print("Noise magnitude {:.5f}         ".format(noise_magnitude))
        validation_results =  np.average(testData(deconf_net, device, val_data, noise_magnitude, criterion)) 
        print("ID Validation Score:",validation_results)
       

        print("Getting Nominals     ")
        id_test_results = testData(deconf_net, device, test_data, noise_magnitude, criterion) 
        print("Getting Anomalies")

        ood_test_results = testData(deconf_net, device, open_data, noise_magnitude, criterion) 
        auroc = calc_auroc(id_test_results, ood_test_results)*100
        tnrATtpr95 = calc_tnr(id_test_results, ood_test_results)
        print("AUROC:", auroc, "TNR@TPR95:", tnrATtpr95)
        best_auc = max(best_auc, auroc)
        if validation_results > best_val_score:
            best_val_score = validation_results
            best_val_auc = auroc
            best_tnr = tnrATtpr95


    print("supposedly best auc: ", best_val_auc, " and tnr@tpr95 ", best_tnr)
    print("true best auc:"      , best_auc)

def calc_tnr(id_test_results, ood_test_results):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=.95)]



def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    print(scores)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result

def testData(model, CUDA_DEVICE, data_loader, noise_magnitude, criterion):
    model.eval()
    num_batches = len(data_loader)
    results = []
    for j, (images, _) in enumerate(tqdm(data_loader)):
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        logits, outputs, _ = model(images)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of the numerator w.r.t. input

        max_numerators, _ = torch.max(outputs, dim = 1)
        max_numerators.backward(torch.ones(len(max_numerators)).to(CUDA_DEVICE))
        
        # Normalizing the gradient to binary in {-1, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
        gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
        gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data, gradient, alpha=noise_magnitude)
        
        # Now calculate score
        _, outputs, _ = model(tempInputs)

        results.extend(torch.max(outputs, dim=1)[0].data.cpu().numpy())
        
        
    return np.array(results)

if __name__ == '__main__':
    main()
