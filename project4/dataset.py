#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mnist_loader import load_mnist



class MNISTDataset(Dataset):
    def __init__(self, feature, label, tag='train', feature_mean=None, feature_std=None):
        self.feature = feature
        self.label = label
        
        if tag == 'train':
            self.mean = self.feature.mean(axis=0, keepdims=True)
            self.std = self.feature.std(axis=0, keepdims=True)
        else:
            self.mean = feature_mean
            self.std = feature_std
            
        try:
            self.feature = (self.feature - self.mean) / (self.std + sys.float_info.epsilon)
        except Exception as e:
            print('请指定训练集上的特征均值与标准差，类型是 Numpy 数组', str(e))
        
        
    
    def __len__(self):
        return self.feature.shape[0]
    
    
    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.label[idx]
        return th.tensor(feature, dtype=th.float32), th.tensor(label, dtype=th.long)
    


    
if __name__ == '__main__':
    train_image, train_label = load_mnist(dataset="training", path="mnist")
    dataset = MNISTDataset(train_image, train_label, tag='train')
    
    # feature, label = dataset[0]
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for feature, label in dataloader:
        print(feature, label)
        break