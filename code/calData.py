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

def testData(model, criterion, CUDA_DEVICE, test_loader_in, test_loader_out, nn_name, data_name, noise_magnitude, temperature):
    model.eval()
    t0 = time.time()
    g1 = open("./softmax_scores/confidence_Our_In.txt", 'w')
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    N = len(test_loader_in)
    print("Processing in-distribution images")
########################################In-distribution###########################################
    for j, (images, _) in enumerate(test_loader_in):
        images = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        #images = images.cuda(CUDA_DEVICE)
        
        outputs, _, _ = model(images)
        # outputs = model(images)
        
        # Calculating the confidence of the output, no perturbation added here, no temperatureature scaling used
        nn_outputs = outputs.data.cpu()
        nn_outputs = nn_outputs.numpy()
        nn_outputs = nn_outputs[0]
        nn_outputs = nn_outputs - np.max(nn_outputs)
        nn_outputs = np.exp(nn_outputs)/np.sum(np.exp(nn_outputs))
        
        # Using temperatureature scaling
        outputs = outputs / temperature
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nn_outputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data,  -noise_magnitude, gradient)
        _, outputs, _ = model(Variable(tempInputs))
        # outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nn_outputs = outputs.data.cpu()
        nn_outputs = nn_outputs.numpy()
        nn_outputs = nn_outputs[0]
        # nn_outputs = nn_outputs - np.max(nn_outputs)
        # nn_outputs = np.exp(nn_outputs)/np.sum(np.exp(nn_outputs))
        g1.write("{}, {}, {}\n".format(temperature, noise_magnitude, np.max(nn_outputs)))
        
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1, N, time.time()-t0))
            t0 = time.time()
        
        if j == N - 1: break
    
    
    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    N = len(test_loader_out)
    for j, (images, _) in enumerate(test_loader_out):
        images = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        #images = images.cuda(CUDA_DEVICE)
        outputs, _, _ = model(images)
        # outputs = model(images)
        
        # Calculating the confidence of the output, no perturbation added here, no temperatureature scaling used
        nn_outputs = outputs.data.cpu()
        nn_outputs = nn_outputs.numpy()
        nn_outputs = nn_outputs[0]
        nn_outputs = nn_outputs - np.max(nn_outputs)
        nn_outputs = np.exp(nn_outputs)/np.sum(np.exp(nn_outputs))
        
        # Using temperatureature scaling
        outputs = outputs / temperature
        
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nn_outputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data,  -noise_magnitude, gradient)
        _, outputs, _ = model(Variable(tempInputs))
        # outputs = model(Variable(tempInputs))
        outputs = outputs / temperature
        # Calculating the confidence after adding perturbations
        nn_outputs = outputs.data.cpu()
        nn_outputs = nn_outputs.numpy()
        nn_outputs = nn_outputs[0]
        # nn_outputs = nn_outputs - np.max(nn_outputs)
        # nn_outputs = np.exp(nn_outputs)/np.sum(np.exp(nn_outputs))
        g2.write("{}, {}, {}\n".format(temperature, noise_magnitude, np.max(nn_outputs)))
        if j % 100 == 99:
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1, N, time.time()-t0))
            t0 = time.time()
