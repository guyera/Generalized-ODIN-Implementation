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

def testData(model, CUDA_DEVICE, data_loader, noise_magnitude):
    model.eval()
    t0 = time.time()
    num_batches = len(data_loader)
    results = None
    for j, (images, _) in enumerate(data_loader):
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        _, outputs, _ = model(images)
        
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
        gradient[::, 2] = (gradient[::, 2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data, noise_magnitude, gradient)
        
        # Now calculate score
        _, outputs, _ = model(tempInputs)
        nn_outputs = outputs.data.cpu()
        nn_outputs = nn_outputs.numpy()
        temp_results = np.max(nn_outputs, axis = 1).tolist()
        if results is None:
            results = temp_results
        else:
            results += temp_results
        
        print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j + 1, num_batches, time.time()-t0))
        t0 = time.time()
        
        if j == num_batches - 1:
            break
    return np.array(results)

"""
    # In case of validation (in-distribution) for hyperparameter tuning, this may be the end
    if test_loader_out is None:
        return
    
    # Otherwise, keep going with the out of distribution data
    g2 = open("./softmax_scores/confidence_Our_Out.txt", 'w')
    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    num_batches = len(test_loader_out)
    for j, (images, _) in enumerate(test_loader_out):
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        _, outputs, _ = model(images)
        
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
        gradient[::, 2] = (gradient[::, 2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(images.data,  noise_magnitude, gradient)
        
        # Now calculate score
        _, outputs, _ = model(tempInputs)
        nn_outputs = outputs.data.cpu()
        nn_outputs = nn_outputs.numpy()
        for nn_output in nn_outputs:
            g2.write("{}, {}\n".format(noise_magnitude, np.max(nn_output)))
            
        print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j + 1, num_batches, time.time()-t0))
        t0 = time.time()
        
        if j == num_batches - 1:
            break
"""
