#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 08:27:07 2022

@author: dl
"""
import numpy as np
import torch as th
import pandas as pd
import seaborn as sns

np.random.seed(1)
th.manual_seed(1)

sns.set_style("darkgrid")

# %% 载入数据
data = pd.read_csv('stature_hand_foot.csv')
data.loc[data.gender == 1, 'gender'] = 'male'
data.loc[data.gender == 2, 'gender'] = 'female'

data.info()
data.describe()

# 基本情况：155条数据，男性80条，女性75条；5列数据；手、脚、身高长度单位：毫米


# %% 探索性数据分析
sns.relplot(data=data, x='handLen', y='height', hue='gender')
sns.relplot(data=data, x='footLen', y='height', hue='gender')
# 两性的数据分布是不同的


# 从分布及直方图进一步认识数据
sns.displot(data=data, x='handLen', hue='gender', kde=True)
sns.displot(data, x='handLen', y='height', hue='gender', kind='kde')

sns.displot(data=data, x='footLen', hue='gender', kde=True)
sns.displot(data, x='footLen', y='height', hue='gender', kind='kde')

# %% 划分数据
# 取出男性数据
male = data[data.gender == 'male']
male = male.drop(['gender', 'idGen'], axis=1)  # 删除gender和idGen列

# 转换为Numpy数组
dataset = male.to_numpy()

# 置乱数据（并未实际置乱数据，只是把序号打乱）
population = dataset.shape[0]
idx = np.random.permutation(population)

# 划分数据为训练集和测试集，比例4：1， 然后转换成 PyTorch tensor
trainset = th.tensor(dataset[idx[: int(population * 0.8)]], dtype=th.float32)
testset = th.tensor(dataset[idx[int(population * 0.8):]], dtype=th.float32)

# %% 建立模型
# TODO: 实现模型中的方法
from model import UnivariateLinearRegression

from utils import show_data_model, show_contour_model, show_losses

# 实现 UnivariateLinearRegression 中有关方法
model = UnivariateLinearRegression()

# 取出特征和目标值，特征值做标准化操作
train_feature = model.standardize(trainset[:, 2], stage='train')  # footLen
train_target = trainset[:, 0]  # height

# 可视化数据和初始模型
show_data_model(model, train_feature, train_target)

# 标准化测试数据
test_feature = model.standardize(testset[:, 2], stage='test')
test_target = testset[:, 0]

# %% 训练模型
# TODO: 实现均方误差损失函数
from criterion import meanSquareError

iteration = 100
train_loss = th.empty((iteration,), dtype=th.float32)
test_loss = th.empty((iteration,), dtype=th.float32)
learning_rate = 0.04

# 优化
for i in range(iteration):
    # 计算模型损失并保存
    train_loss[i] = meanSquareError(model.predict(train_feature), train_target)
    test_loss[i] = meanSquareError(model.predict(test_feature), test_target)

    # 优化模型（梯度下降法）
    # 修改学习率， 观察训练过程
    model.update(train_feature, train_target, learning_rate)

    # 可视化训练过程：直接观察和在参数空间观察模型优化过程
    if i % 10 == 0:
        show_data_model(model, train_feature, train_target)
        show_contour_model(model, train_feature, train_target)

# 可视化训练过程
show_losses(iteration, train_loss, test_loss)

# %% 评估模型
# 观察测试集数据和当前模型
show_data_model(model, test_feature, test_target)

# 在自身的数据上测试
my_footLen = th.tensor([262], dtype=th.float32)
predict_height = model.predict(model.standardize(my_footLen, stage='test'))

print(f'脚的长度：{my_footLen.item()}mm，预测的身高：{predict_height.item():.2f}mm')
