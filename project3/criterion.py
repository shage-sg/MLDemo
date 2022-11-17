# -*- coding: utf-8 -*-
import torch as th
from torch.nn import BCEWithLogitsLoss

"""
This loss combines a Sigmoid layer and the BCELoss in one single class. 
This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
by combining the operations into one layer, we take advantage of the log-sum-exp trick 
for numerical stability.
"""


def bce(prediction, label):
    return BCEWithLogitsLoss()(prediction, label)



def balanced_bce(prediction, label, beta):
    # beta in (0,1)
    return -th.mean(beta * label * th.log(th.sigmoid(prediction)) + (1-beta) * (1-label) * th.log(1-th.sigmoid(prediction)))