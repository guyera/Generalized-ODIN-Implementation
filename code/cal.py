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
from cosinedeconf import CosineDeconf
from deconfnet import DeconfNet
#CUDA_DEVICE = 0

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])




# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nn_name = "densenet10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()

def update_module(module):
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = update_module(module1)
    return module

def test(nn_name, data_name, CUDA_DEVICE, epsilon, temperature, validate = False, validation_proportion = 0.4):
    
    net1 = torch.load("../models/{}.pth".format(nn_name))

    # The provided pretrained models were based on an old version of pytorch.
    # The following code updates their BatchNorm2d layers to be compatible
    # with the latest version.
    for i, (name, module) in enumerate(net1._modules.items()):
        module = update_module(module)
    
    # Construct g, h, and the composed deconf net
    h = CosineDeconf(10, 10)
    h.cuda(CUDA_DEVICE)
    deconf_net = DeconfNet(net1, 10, 10, h)

    optimizer1 = optim.SGD(deconf_net.parameters(), lr = 0.01, momentum = 0.9)
    deconf_net.cuda(CUDA_DEVICE)

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
    for epoch in range(1):
        print("Epoch #{}".format(epoch + 1))
        for batch_idx, (inputs, targets) in enumerate(train_loader_in):
            break
            inputs = inputs.cuda(CUDA_DEVICE)
            targets = targets.cuda(CUDA_DEVICE)
            optimizer1.zero_grad()
            softmax, _, _ = deconf_net(inputs)
            loss = criterion(softmax, targets)
            loss.backward()
            optimizer1.step()
            total_loss = loss.item()
            print("Batch #{} Loss: {}".format(batch_idx + 1, total_loss))
    d.testData(deconf_net, criterion, CUDA_DEVICE, test_loader_in, test_loader_out, nn_name, data_name, epsilon, temperature) 
    m.metric(nn_name, data_name)
