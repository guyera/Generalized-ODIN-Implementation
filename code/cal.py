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
from densenet import DenseNet3
from deconfnet import DeconfNet, CosineDeconf

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])

criterion = nn.CrossEntropyLoss()

def test(data_name, device, noise_magnitudes, epochs, validation_proportion = 0.15):
    #def __init__(self, depth, num_classes, growth_rate=12,
    #             reduction=0.5, bottleneck=True, dropRate=0.0):
    dense_net = DenseNet3(depth = 100, num_classes = 10)
    dense_net.to(device)
    # Construct g, h, and the composed deconf net
    h = CosineDeconf(dense_net.in_planes, 10)
    h.to(device)
    deconf_net = DeconfNet(dense_net, dense_net.in_planes, 10, h)
    deconf_net.to(device)

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

    test_set_out = torchvision.datasets.ImageFolder("../data/{}".format(data_name), transform=transform)
    test_loader_out = torch.utils.data.DataLoader(test_set_out, batch_size=100,
                                     shuffle=False, num_workers=2)

    train_set_in = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    # Don't get the train_loader_in yet, since the train_set_in will be split into validation and training data

    test_set_in = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader_in = torch.utils.data.DataLoader(test_set_in, batch_size=100,
                                     shuffle=False, num_workers=2)
    
    # Split the train_set_in into a training set and a validation set, based on the validation_proportion
    # Stagger the elements used for validation
    validation_range = np.arange(0, len(train_set_in), 1. / validation_proportion)
    validation_indices = [math.floor(x) for x in validation_range]
    
    # Use the remaining (complementary) indices as the training set
    validation_indices_set = set(validation_indices)
    training_indices = [x for x in range(len(train_set_in)) if x not in validation_indices_set]
    
    # Construct the subsets and dataloaders
    validation_set_in = torch.utils.data.Subset(train_set_in, validation_indices)
    validation_loader_in = torch.utils.data.DataLoader(validation_set_in, batch_size=100,
                                     shuffle=False, num_workers=2)
    train_set_in = torch.utils.data.Subset(train_set_in, training_indices)
    train_loader_in = torch.utils.data.DataLoader(train_set_in, batch_size=64,
                                     shuffle=True, num_workers=2)

    # Train the model
    
    deconf_net.train()
    for epoch in range(epochs):
        print("Epoch #{}".format(epoch + 1))
        for batch_idx, (inputs, targets) in enumerate(train_loader_in):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            softmax, _, _ = deconf_net(inputs)
            loss = criterion(softmax, targets)
            loss.backward()
            optimizer.step()
            h_optimizer.step()
            total_loss = loss.item()
            print("Batch #{} Loss: {}".format(batch_idx + 1, total_loss))
        h_scheduler.step()
        scheduler.step()

    for noise_magnitude in noise_magnitudes:
        print("----------------------------------------")
        print("        Noise magnitude {:.5f}         ".format(noise_magnitude))
        print("----------------------------------------")
        print("------------------------")
        print("       Validation       ")
        print("------------------------")
        validation_results = d.testData(deconf_net, device, validation_loader_in, noise_magnitude) 
        m.validate(validation_results)
        print("------------------------")
        print("        Testing         ")
        print("------------------------")
        print("------------------")
        print("     Nominals     ")
        print("------------------")
        id_test_results = d.testData(deconf_net, device, test_loader_in, noise_magnitude) 
        print("------------------")
        print("    Anomalies     ")
        print("------------------")
        ood_test_results = d.testData(deconf_net, device, test_loader_out, noise_magnitude) 
        m.test(id_test_results, ood_test_results)
