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

# a is an N x 3 tensor, where N is the number of images
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]).reshape(10, 3)

# b is an M x 3 tensor, where M is the number of columns of an image representation to operate on
b = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(5, 3)

# coerce a to an N x M x 3 tensor by unsqueezing dim 1 and expanding it
a = a.unsqueeze(1).expand(-1, 5, -1)

# Now, b can be broadcasted with a to make it apply to every
# image
print(a + b)
