#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:17:48 2022

@author: dl
"""
import torch as th

# 勿修改此行
th.manual_seed(1)


def meanSquareError(pred, target):
    # TODO: 实现均方误差的计算
    return ((pred - target) ** 2).sum() / (pred.shape[0] * 2)


if __name__ == '__main__':
    pred = th.normal(0, 1, (5, 1), dtype=th.float32)
    target = th.normal(0, 1, (5, 1), dtype=th.float32)
    loss = meanSquareError(pred, target)
    assert th.norm(loss - th.tensor(0.6721)) < 1e-4, \
        '损失函数实现有误'
