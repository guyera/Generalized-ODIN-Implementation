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
import calMetric as m
import calData as d
import math

from torch.utils.data import DataLoader, Subset

from densenet import DenseNet3
from deconfnet import DeconfNet, CosineDeconf, InnerDeconf, EuclideanDeconf

from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])

criterion = nn.CrossEntropyLoss()

h_dict = {
    "cosine":   CosineDeconf,
    "inner":    InnerDeconf,
    "euclid":   EuclideanDeconf
    }

def get_datasets(data_name):

    train_set_in = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    test_set_in  = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)
    outlier_set  = torchvision.datasets.ImageFolder("./data/{}".format(data_name), transform=transform)
    
    test_indices      = list(range(len(test_set_in)))
    validation_set_in = Subset(test_set_in, test_indices[:1000])
    test_set_in       = Subset(test_set_in, test_indices[1000:])


    train_loader_in      =  DataLoader(train_set_in,      batch_size=64, shuffle=True,  num_workers=4)
    validation_loader_in =  DataLoader(validation_set_in, batch_size=128,shuffle=False, num_workers=4)
    test_loader_in       =  DataLoader(test_set_in,       batch_size=128,shuffle=False, num_workers=4)
    outlier_loader       =  DataLoader(outlier_set,       batch_size=128,shuffle=False, num_workers=4)

    return train_loader_in, validation_loader_in, test_loader_in, outlier_loader


def test(data_name, device, noise_magnitudes, epochs, h_type='cosine', validation_proportion = 0.10):
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
    train_data, val_data, test_data, open_data =get_datasets(data_name)
  
    # Train the model
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

    torch.save(deconf_net.state_dict(), f"desnet{epochs}_{h_type}.pth")

    deconf_net.eval()
    for noise_magnitude in noise_magnitudes:
        print("----------------------------------------")
        print("        Noise magnitude {:.5f}         ".format(noise_magnitude))
        print("----------------------------------------")
        print("------------------------")
        print("       Validation       ")
        print("------------------------")
        validation_results = d.testData(deconf_net, device, val_data, noise_magnitude) 
        m.validate(validation_results)
        print("------------------------")
        print("        Testing         ")
        print("------------------------")
        print("------------------")
        print("     Nominals     ")
        print("------------------")
        id_test_results = d.testData(deconf_net, device, test_data, noise_magnitude) 
        print("------------------")
        print("    Anomalies     ")
        print("------------------")
        ood_test_results = d.testData(deconf_net, device, open_data, noise_magnitude) 
        m.test(id_test_results, ood_test_results)
