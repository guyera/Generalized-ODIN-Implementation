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

class DenseNetClassifier(nn.Module):
    def __init__(self, dense_net, temperature):
        super(DenseNetClassifier, self).__init__()
        
        self.dense_net = dense_net
        self.temperature = temperature
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        output = self.dense_net(x)
        output = output / self.temperature
        output = self.softmax(output)
        return output
