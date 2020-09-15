# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        return None

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)

        ret = (torch.matmul(x,w.T))
        return ret

class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        return None

    def forward(self, x):
        ret = -((x -self.h.weight)**2)
        return ret
        
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        return None

    def forward(self, x):
        return self.h(x)


class DeconfNet(nn.Module):
    def __init__(self, underlying_model, in_features, num_classes, h):
        super(DeconfNet, self).__init__()
        
        self.num_classes = num_classes

        self.underlying_model = underlying_model
        
        self.g = nn.Sequential(
            nn.Linear(in_features, 1),
            #nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        
        self.h = h
        
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        output = self.underlying_model(x)
        numerators = self.h(output)
        denominators = self.g(output)
        # denominators is an N x 1 tensor.
        # Expand the denominators so that they repeat
        # across classes for each image (N x M)
        expanded_denominators = denominators.expand(-1, self.num_classes)
        
        # Now, broadcast the denominators per image across the numerators by division
        quotients = numerators / denominators
        
        # Pass through softmax layer
        softmax = self.softmax(quotients)

        # Return the softmax (used during training), numerators, and denominators
        return softmax, numerators, denominators
