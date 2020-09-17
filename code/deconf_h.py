# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

    def init_weights():
        return None

    def forward(self, x):
        x = norm(x)
        w = norm(self.fc2.weight)

        ret = (torch.matmul(x,w.T))
        return ret

class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights():
        return None

    def forward(self, x):
        ret = -((x -self.h.weight)**2)
        return ret
        
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights():
        return None

    def forward(self, x):
        return self.h(x)
        
