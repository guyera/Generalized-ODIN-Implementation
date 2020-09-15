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
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
import math
from densenet import DenseNet3
from densenetclassifier import DenseNetClassifier
from cosinedeconf import CosineDeconf
from deconfnet import DeconfNet
#device = 0

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])




# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         dense_net10
# Densenet trained on CIFAR-100:        dense_net100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nn_name = "dense_net10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()

def test(nn_name, data_name, device, noise_magnitudes, temperature, validate = False, validation_proportion = 0.4):
    
    #dense_net = torch.load("../models/{}.pth".format(nn_name))
    #def __init__(self, depth, num_classes, growth_rate=12,
    #             reduction=0.5, bottleneck=True, dropRate=0.0):
    dense_net = DenseNet3(depth = 100, num_classes = 10)
    dense_net.to(device)
    dense_net_classifier = DenseNetClassifier(dense_net, temperature)
    dense_net_classifier.to(device)
    # Construct g, h, and the composed deconf net
    h = CosineDeconf(dense_net.in_planes, 10)
    h.to(device)
    deconf_net = DeconfNet(dense_net, dense_net.in_planes, 10, h)
    deconf_net.to(device)

    optimizer = optim.SGD(deconf_net.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0001)
    #optimizer = optim.SGD(dense_net_classifier.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0001)

    test_set_out = None
    test_set_in = None

    test_set_out = torchvision.datasets.ImageFolder("../data/{}".format(data_name), transform=transform)
    test_loader_out = torch.utils.data.DataLoader(test_set_out, batch_size=1,
                                     shuffle=False, num_workers=2)

    train_set_in = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_set_in = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    
    # Split the test_loader_in into a validation set and a test set, based on the validation_proportion
    validation_range = np.arange(0, len(test_set_in), 1. / validation_proportion)
    validation_indices = [math.floor(x) for x in validation_range]
    if validate:
        indices = validation_indices
    else:
        validation_indices_set = set(validation_indices)
        indices = [x for x in range(len(test_set_in)) if x not in validation_indices_set]
    
    test_set_in = torch.utils.data.Subset(test_set_in, indices)
    test_loader_in = torch.utils.data.DataLoader(test_set_in, batch_size=1,
                                     shuffle=False, num_workers=2)
    train_loader_in = torch.utils.data.DataLoader(train_set_in, batch_size=100,
                                     shuffle=True, num_workers=2)

    # Train the model
    
    deconf_net.train()
    for epoch in range(100):
        print("Epoch #{}".format(epoch + 1))
        for batch_idx, (inputs, targets) in enumerate(train_loader_in):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            softmax, _, _ = deconf_net(inputs)
            loss = criterion(softmax, targets)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            print("Batch #{} Loss: {}".format(batch_idx + 1, total_loss))
    
    """
    dense_net_classifier.train()
    for epoch in range(600):
        print("Epoch #{}".format(epoch + 1))
        for batch_idx, (inputs, targets) in enumerate(train_loader_in):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = dense_net_classifier(inputs)
            loss = criterion(outputs, targets)
            total_loss = loss.item()
            loss.backward()
            optimizer.step()
            print("Batch #{} Loss: {}".format(batch_idx + 1, total_loss))
    """

    #d.testData(deconf_net, criterion, device, test_loader_in, test_loader_out, nn_name, data_name, epsilon, temperature) 
    for magnitude in noise_magnitudes:
        print("----------------------------------------")
        print("Testing magnitude {:.5f}".format(magnitude))
        print("----------------------------------------")
        d.testData(deconf_net, criterion, device, test_loader_in, test_loader_out, nn_name, data_name, magnitude, temperature) 
        # d.testData(dense_net_classifier, criterion, device, test_loader_in, test_loader_out, nn_name, data_name, magnitude, temperature) 
        m.metric(nn_name, data_name)
