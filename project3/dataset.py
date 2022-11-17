#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
import torch as th
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CreditDefaultDataset(Dataset):
    def __init__(self, dataframe, tag, feature_mean=None, feature_std=None):
        self.data = dataframe
        # 删除ID列；性别，学历，婚姻列用one-hot形式表示
        del self.data['ID']
        
        gender = pd.get_dummies(self.data.SEX, prefix='gender') # one-hot scheme
        del self.data['SEX']
        
        self.data.loc[(self.data.EDUCATION==0) | (self.data.EDUCATION==5) | (self.data.EDUCATION==6), 'EDUCATION'] = 4
        education = pd.get_dummies(self.data.EDUCATION, prefix='education')
        del self.data['EDUCATION']
        
        self.data.loc[(self.data.MARRIAGE==0), 'MARRIAGE'] = 3
        marriage = pd.get_dummies(self.data.MARRIAGE, prefix='marriage')
        del self.data['MARRIAGE']
        
        self.data = pd.concat([gender,education,marriage,self.data], axis='columns')
        
        
        self.standardized_columns = self.data.columns.delete(-1)
        
        if tag == 'train':
            self.mean = self.data.loc[:, self.standardized_columns].mean()
            self.std = self.data.loc[:, self.standardized_columns].std()
        else:
            self.mean = feature_mean
            self.std = feature_std
            
        try:
            self.data.loc[:, self.standardized_columns] = (self.data.loc[:, self.standardized_columns] - self.mean) / self.std
        except Exception as e:
            print('请指定训练集上的特征均值与标准差', str(e))
            
        self.feature = self.data.loc[:, :'PAY_AMT6'].to_numpy()
        self.label = self.data.loc[:, 'default payment_next_month':'default payment_next_month'].to_numpy()

    
    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        feature = self.feature[idx, :]
        label = self.label[idx]
        return th.tensor(feature, dtype=th.float32), th.tensor(label, dtype=th.float32)
  
    
if __name__ == '__main__':
    data = pd.read_csv('credit_card_default.csv', sep=',')
    dataset = CreditDefaultDataset(data, tag='train')
    
    # feature, label = dataset[0]
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for feature, label in dataloader:
        print(feature, label)
        break