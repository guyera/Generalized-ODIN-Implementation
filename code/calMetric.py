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

import numpy as np
from sklearn.metrics import roc_auc_score

def auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    print(scores)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)
    return result

def average_max_id_score(validation_results):
    # calculate the average max in-distribution anomaly score for validating hyperparameters
    return np.average(validation_results)

def validate(validation_results):
    id_validation_score = average_max_id_score(validation_results)
    print("ID Validation Score:", id_validation_score)

def test(id_test_results, ood_test_results):
    auc = auroc(id_test_results, ood_test_results)
    print("AUROC:", auc*100)
