#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:25:37 2022

@author: dl
"""
import torch as th

# 勿修改此行
th.manual_seed(1)


class UnivariateLinearRegression(object):
    def __init__(self):
        self.weight = th.normal(110, 20, (1,), dtype=th.float32)
        self.bias = th.tensor([1700], dtype=th.float32)
        self.mean = None
        self.std = None

    def standardize(self, feature, stage='train'):
        if stage == 'train':
            self.mean = feature.mean()
            self.std = feature.std()
            return (feature - self.mean) / self.std
        elif self.mean is not None and self.std is not None:
            return (feature - self.mean) / self.std
        else:
            return feature

    def predict(self, feature):
        # TODO: 根据模型的数学定义实现预测值的输出（模型 y = wx + b，x 是特征，y 是预测值）
        y = self.weight * feature + self.bias
        return y

    def get_grad(self, feature, target):
        # TODO: 在损失函数为MSE的前提下实现梯度的计算（分别计算 weight 和 bias 的梯度）
        error = self.predict(feature) - target
        # 对于 MSE loss 计算梯度，梯度向量表示成元组 (dw, db)
        dw = (error * feature).sum() / (error.shape[0])
        db = error.sum() / (error.shape[0])
        return dw, db

    def update(self, feature, target, lr):
        dw, db = self.get_grad(feature, target)
        # 梯度下降法
        self.weight = self.weight - lr * dw
        self.bias = self.bias - lr * db


# 运行 model.py 检查各方法是否正确实现
if __name__ == '__main__':
    model = UnivariateLinearRegression()
    feature = th.normal(0, 1, (4,), dtype=th.float32)
    target = th.normal(0, 1, (4,), dtype=th.float32)
    pred = model.predict(feature)
    assert th.norm(pred - th.tensor([1732.8922, 1707.6003, 1776.5631, 1644.3130])) < 1e-4, \
        'predict 方法实现有误'

    dw, db = model.get_grad(feature, target)
    assert th.norm(dw - th.tensor(232.0105)) < 1e-4 and th.norm(db - th.tensor(1715.9260)) < 1e-4, \
        'get_grad 方法实现有误'
