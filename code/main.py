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
import argparse
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
#import lmdb
from scipy import misc
import cal as c


parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

parser.add_argument('--out-dataset', default = "Imagenet", type = str,
                    help = 'out-of-distribution dataset')
parser.add_argument('--h-type', default = "cosine", type = str,
                    help = 'cosine|inner|euclid')
parser.add_argument('--gpu', default = 0, type = int,
          help = 'gpu index')
parser.add_argument('--magnitudes', nargs = '+', default = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08], type = float,
                    help = 'perturbation magnitudes')
parser.add_argument('--epochs', default = 300, type = int,
           help = 'gpu index')
parser.add_argument('--eval-only', action='store_true',
            help='Load a model from file')
parser.set_defaults(argument=True)

def main():
    args = parser.parse_args()
    c.test(args.out_dataset, args.gpu, args.magnitudes, args.epochs, h_type=args.h_type, eval_only=args.eval_only)

if __name__ == '__main__':
    main()

















