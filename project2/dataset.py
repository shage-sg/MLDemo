#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
继承 PyTorch Dataset 类需要实现 __len__ 和 __getitem__ 方法
__init__ 中的 tag 参数的取值有：train、validation、test
"""
# import numpy as np
import torch as th
import pandas as pd
from torch.utils.data import Dataset

th.manual_seed(1)


class WineQualityDataset(Dataset):
    def __init__(self, dataframe, tag, feature_mean=None, feature_std=None):
        self.data = dataframe
        self.feature = self.data.loc[:, :'alcohol'].to_numpy()
        self.label = self.data.loc[:, 'quality':'quality'].to_numpy()

        if tag == 'train':
            self.mean = self.feature.mean(axis=0)
            self.std = self.feature.std(axis=0)
        else:
            self.mean = feature_mean
            self.std = feature_std

        try:
            self.feature = (self.feature - self.mean) / self.std
        except Exception as e:
            print('请指定训练集上的特征均值与标准差，类型是 Numpy 数组', str(e))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # TODO: 获得索引序号 idx 处的特征和标签
        # 数据是numpy的array数据类型，直接使用索引进行读取
        feature = self.feature[idx, :]
        label = self.label[idx]
        return th.tensor(feature, dtype=th.float32), th.tensor(label, dtype=th.float32)


if __name__ == '__main__':
    dataframe = pd.read_csv('./winequality-red.csv', sep=';')
    dataset = WineQualityDataset(dataframe, tag='train')

    feature, label = dataset[0]
    print(feature, label)
    # 结果应该是：
    # tensor([-0.5284,  0.9619, -1.3915, -0.4532, -0.2437, -0.4662, -0.3791,  0.5583,
    # 1.2886, -0.5792, -0.9602]) tensor([5.])
