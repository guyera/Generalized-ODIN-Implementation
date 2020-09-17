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

#self.weights = torch.nn.Parameter(torch.randn(size = (num_classes, in_features)) * math.sqrt(2 / (in_features)))
class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

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
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        x = x.unsqueeze(2) #(batch, latent, 1)
        h = self.h.weight.T.unsqueeze(0) #(1, latent, num_classes)
        ret = -((x -h).pow(2)).mean(1)
        return ret
        
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.h(x)


class DeconfNet(nn.Module):
    def __init__(self, underlying_model, in_features, num_classes, h):
        super(DeconfNet, self).__init__()
        
        self.num_classes = num_classes

        self.underlying_model = underlying_model
        
        self.g = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.BatchNorm1d(1),
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
        
        # Now, broadcast the denominators per image across the numerators by division
        quotients = numerators / denominators
        

        # logits, numerators, and denominators
        return quotients, numerators, denominators
