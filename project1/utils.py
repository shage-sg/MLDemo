#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 20:15:33 2022

@author: dl
"""
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from criterion import meanSquareError


def show_data_model(model, X, y):
    pred = model.weight * X + model.bias
    plt.figure()
    plt.plot(X.numpy(), pred.numpy(), 'r-')
    plt.scatter(X.numpy(), y.numpy())
    plt.xlabel('feature')
    plt.ylabel('height')
    plt.legend(['model', 'data points'], loc='lower right')
    plt.show()


def show_contour_model(model, X, y):
    l = 100
    k = th.linspace(20, 80, l)
    b = th.linspace(1690, 1810, l)
    K, B = th.meshgrid(k, b, indexing='ij')
    k0, b0 = model.weight, model.bias

    J = th.zeros((l, l))
    for i in range(l):
        for j in range(l):
            model.weight, model.bias = k[i], b[j]
            J[i, j] = meanSquareError(model.predict(X), y)

    model.weight, model.bias = k0, b0
    plt.figure()
    plt.contour(K.numpy(), B.numpy(), J.numpy(), 20)
    plt.scatter(model.weight.numpy(), model.bias.numpy())
    plt.xlabel('model weight')
    plt.ylabel('model bias')
    plt.legend(['model parameter'], loc='lower right')
    plt.show()


def show_losses(iteration, train_loss, test_loss):
    plt.figure()
    plt.plot(np.arange(1, iteration + 1), train_loss.numpy(), 'b')
    plt.plot(np.arange(1, iteration + 1), test_loss.numpy(), 'r')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend(['train loss', 'test loss'], loc='upper right')
    plt.title('training process')
    plt.show()
