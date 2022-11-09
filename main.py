def predict(self, feature):
    # 根据模型的数学定义实现预测值的输出（模型 y = wx + b，x 是特征，y 是预测值）
    # 一元线性回归 weight::偏置 bias:截距
    y = self.weight * feature + self.bias
    return y


def get_grad(self, feature, target):
    # 损失函数为MSE的梯度计算（分别计算 weight 和 bias 的梯度）
    # 计算预测值和真实值之间的误差
    error = self.predict(feature) - target
    # 对于 MSE损失函数分别计算w和b梯度，梯度向量表示成元组 (dw, db)
    # 对w求偏导
    dw = (error * feature).sum() / (error.shape[0])
    # 对b求偏导
    db = error.sum() / (error.shape[0])
    return dw, db
