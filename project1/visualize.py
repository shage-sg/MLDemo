#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 10:52:36 2022

@author: dl
"""
import numpy as np
import pandas as pd
import torch as th

import plotly.graph_objects as go
import plotly.io as pio

from model import UnivariateLinearRegression
from criterion import meanSquareError

np.random.seed(1)

#%%

data = pd.read_csv('stature_hand_foot.csv')
male = data[data.gender==1]
male = male.drop(['gender','idGen'], axis=1) # 删除gender和idGen列
# 转换为Numpy数组
dataset = male.to_numpy()

# 置乱数据（并未实际置乱数据，只是把序号打乱）
population = dataset.shape[0]
idx = np.random.permutation(population)

# 划分数据为训练集和测试集，比例4：1， 然后转换成 PyTorch tensor
trainset = dataset[idx[: int(population * 0.8)]]
testset = dataset[idx[int(population * 0.8) :]]


#%% contour graph
model = UnivariateLinearRegression()

# 取出特征和目标值，特征值做标准化操作
# train_feature = model.standardize(trainset[:, 2], stage='train') # footLen
train_feature = trainset[:, 2] # no standardize

train_target = trainset[:, 0]  # height


l = 100
# k = np.linspace(10, 90, l)
# b = np.linspace(1710, 1790, l)

k = np.linspace(-2, 2, l)
b = np.linspace(1748, 1752, l)

K, B = np.meshgrid(k, b)

J = th.zeros((l,l))
for i in range(l):
    for j in range(l):
        model.weight, model.bias = th.tensor([k[i]]), th.tensor([b[j]])
        J[i, j] = meanSquareError(model.predict(th.tensor(train_feature)), th.tensor(train_target))
        
fig = go.Figure()

fig.add_trace(go.Surface(
    x = k,
    y = b,
    z = J.numpy()
))

fig.update_scenes(
    xaxis_title_text='w',
    yaxis_title_text='b',
    zaxis_title_text='L',
)

fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                  highlightcolor="limegreen", project_z=True))

fig.update_layout(
    title="参数空间的损失函数"
)

fig.show(renderer="png")
pio.write_html(fig, 'lossFunction-unstandadize.html')