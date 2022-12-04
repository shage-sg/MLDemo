#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json

# %% 载入数据
D = pd.read_csv('titanic/train.csv', sep=',')

D.info()
D.describe()

# %% 数据预处理
# 用均值填补缺失值
D['Age'].fillna(D['Age'].mean(skipna=True), inplace=True)

# 重命名字段
D = D.rename(columns={'Survived': 'Label'})

# 去除 PassengerId , Name, Ticket, Cabin, Embarked 列
del D['PassengerId']
del D['Name']
del D['Ticket']
del D['Cabin']
del D['Embarked']

# %% 分割数据集
D.sample(frac=1).reset_index(drop=True)  # 置乱
trainset = D.loc[:700, :]
testset = D.loc[701:, :]  # 190个测试样本
testset = testset.reset_index(drop=True)

# %% 构造属性，对于 Age 和 Fare 两个连续取值列，以 25%, 50%, 75% 处数值为区间分割依据
A = {
    'Pclass': {'type': 'descrete', 'value': np.sort(D['Pclass'].unique()).tolist()},
    'Sex': {'type': 'descrete', 'value': np.sort(D['Sex'].unique()).tolist()},
    'Age': {'type': 'continue', 'value': [[-np.inf, 22], [22, 30], [30, 35], [35, np.inf]]},  # 四个区间
    'SibSp': {'type': 'descrete', 'value': np.sort(D['SibSp'].unique()).tolist()},
    'Parch': {'type': 'descrete', 'value': np.sort(D['Parch'].unique()).tolist()},
    'Fare': {'type': 'continue', 'value': [[-np.inf, 7.9], [7.9, 14.5], [14.5, 31], [31, np.inf]]},
}

# %% 模型：决策树

from model import grow_tree, evaluate_tree

tree = {}
grow_tree(tree, trainset, A, method='C4.5')

print(json.dumps(tree, indent=4))

# %% 预测
prediction = np.zeros(len(testset), dtype=np.int32)
for rowid, row in testset.iterrows():
    prediction[rowid] = evaluate_tree(tree, row, A)

testset['prediction'] = pd.Series(prediction)

# %% 评估

from utils import classification_metric

accuracy, precision, recall, F1 = classification_metric(testset['prediction'].to_numpy(), testset['Label'].to_numpy())

print(f'accuracy: {accuracy:.2f}, precision: {accuracy:.2f}, recall: {recall:.2f}, F1: {F1:.2f}')
