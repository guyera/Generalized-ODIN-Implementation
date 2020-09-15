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

class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.randn(size = (num_classes, in_features), requires_grad = True), requires_grad = True).cuda()

    def forward(self, x):
        # Compute hadamard product of each image in x (actually fp(x))
        # with each row in the weight matrix. First, we expand x so that
        # each image is repeated several times (once for every row in the weight
        # matrix). Originally, x is N x K, where N is the batch size and K is
        # the number of features in the representation of an image (in_features). We
        # need it to be N x M x K, where each image is entirely repeated M times, to
        # be broadcastable with the weight matrix.
        expanded_x = x.unsqueeze(1).expand(-1, len(self.weights), -1)
        
        # Now, we can broadcast the weight matrix against the list of images to
        # apply the entire weight matrix to every image
        products = expanded_x * self.weights

        # At this point, out is an N x M x K tensor, where M is the number of classes,
        # and K is the number of features in the feature representation space supplied
        # to this function (e.g. the penultimate layer of an NN, AKA in_features)
        # Now, we should sum across the last dimension for an N x M matrix. This is
        # really just a custom, broadcasted dot product
        numerator = torch.sum(input = products, dim = 2)

        # out is now the numerator of our cosine distance formula (or rather, a matrix
        # of numerators, one for every image / class pair)

        # Next, take the l2 norm of every image representation vector
        x_norm = torch.norm(x, dim = 1)

        # Similarly, take the l2 norm of each class' weight vector
        weight_norm = torch.norm(self.weights, dim = 1)
        
        # Once again, we need to broadcast the weight norms across every image
        x_norm = x_norm.unsqueeze(1).expand(-1, len(self.weights))
        
        # Broadcast the weight norms against the expanded image norms
        denominator = weight_norm * x_norm

        # The numerator and denominator are both N x M; for a given numerator[i][j]
        # or denominator[i][j], the quotient is between the dot product of the i'th
        # image feature vector and the j'th class weight vector, and the product of the
        # i'th image feature vector norm and the j'th class weight vector norm. Now,
        # just divide them element-wise and return the result. The [i][j] component
        # of the returned tensor will represent h_j(x[i]).
        return numerator / denominator
