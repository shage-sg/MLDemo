# -*- coding: utf-8 -*-
import torch as th
from torch.nn.functional import mse_loss


def lossFunc(prediction, label, model, lambd, norm='L1'):
    mse = mse_loss(prediction, label)

    if norm == 'L1':
        # penalty = model.weight.abs().sum()
        penalty = th.linalg.vector_norm(model.weight, ord=1)

    elif norm == 'L2':
        # penalty = model.weight.pow(2.0).sum()
        penalty = th.linalg.vector_norm(model.weight, ord=2)

    return mse + lambd * penalty
