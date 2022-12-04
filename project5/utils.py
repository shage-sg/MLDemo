#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys


# 计算基尼值。基尼值越小数据集的纯度越高
def gini(s):
    return 1-((pd.value_counts(s)/len(s))**2).sum()



# 计算熵。熵越小数据集的纯度越高
def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float') / len(s)
    for p in freqs:
        if p != 0.0:  # 当p为零时，令熵为零
            res -= p * np.log2(p)
    return res


def confusion_matrix(pred, label):
    TP = np.logical_and(pred == 1, label == 1).astype(np.float32).sum().item()
    FP = np.logical_and(pred == 1, label == 0).astype(np.float32).sum().item()
    FN = np.logical_and(pred == 0, label == 1).astype(np.float32).sum().item()
    TN = np.logical_and(pred == 0, label == 0).astype(np.float32).sum().item()
    return TP, FP, TN, FN


def classification_metric(pred, label):
    TP, FP, TN, FN = confusion_matrix(pred, label)
    sample_size = pred.shape[0]
    accuracy = (TP + TN) / sample_size
    precision = TP / (TP + FP + sys.float_info.epsilon)
    recall = TP / (TP + FN + sys.float_info.epsilon)
    F1 = 2 * precision * recall / (precision + recall + sys.float_info.epsilon)
    return accuracy, precision, recall, F1


if __name__ == '__main__':
    # list, ndarray, Series 均可
    s = pd.Series([0, 1, 1, 2, 2, 2, 0, 1, 2, 1])
    print(entropy(s))
    print(gini(s))
