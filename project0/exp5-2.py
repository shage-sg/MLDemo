# -*- coding: utf-8 -*-
"""
-------------------------------------------------
 File Name: exp5-2
 Blog: https://bulingling.top
 Github: https://github.com/shage-sg
 Author: bobo
 Date: 2022/11/1
-------------------------------------------------
"""
__author__ = 'bobo'

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Dropout, PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.python.keras.layers import Activation


def data_preprocess(filepath, flag):
    seed = 7
    np.random.seed(seed)
    # 载入数据集
    df = pd.read_csv(filepath)
    # 删除不需要的栏位
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    # 处理遗失数据
    df[["Age"]] = df[["Age"]].fillna(value=df[["Age"]].mean())
    df[["Fare"]] = df[["Fare"]].fillna(value=df[["Fare"]].mean())
    df[["Embarked"]] = df[["Embarked"]].fillna(value=df["Embarked"].value_counts().idxmax())
    print(df["Embarked"].value_counts())
    print(df["Embarked"].value_counts().idxmax())
    # 转换分类数据
    df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)
    # Embarked 栏位的One-hot 编码
    enbarked_one_hot = pd.get_dummies(df["Embarked"],
                                      prefix="Embarked")
    df = df.drop("Embarked", axis=1)
    df = df.join(enbarked_one_hot)
    if flag == "train":
        # 将训练集标签的survived 栏位移至最后
        df_survived = df.pop("Survived")
        df["Survived"] = df_survived
    else:
        df_gs = pd.read_csv("./titanic/gender_submission.csv")
        df["Survived"] = df_gs["Survived"]

    df.head().to_html("Ch6_2_2.html")
    # 储存处理后的数据
    df.to_csv(f"titanic_{filepath.split(r'/')[-1]}", index=False)


data_preprocess(filepath="./titanic/train.csv", flag="train")
data_preprocess(filepath="./titanic/test.csv", flag="test")

# 2、分割成特征数据和标签数据
seed = 7
np.random.seed(seed)
# 载入Titanic 的训练和测试数据集
df_train = pd.read_csv("./titanic_train.csv")
df_test = pd.read_csv("./titanic_test.csv")
dataset_train = df_train.values
dataset_test = df_test.values
# 分割成特征数据和标签数据
X_train = dataset_train[:, 0:9]
Y_train = dataset_train[:, 9]
X_test = dataset_test[:, 0:9]
Y_test = dataset_test[:, 9]

# 3、数据标准化
# 特征标准化
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

# 4、定义模型并编译模型
# 定义模型
model = Sequential()
model.add(Dense(20, input_dim=X_train.shape[1]))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
# 编译模型
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
# 5、训练模型、评估模型
# 训练模型
print("Training ...")
history = model.fit(X_train, Y_train, validation_split=0.3, epochs=80, batch_size=32, verbose=1)
# 评估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_train, Y_train, verbose=1)
print("训练数据集的准确度= {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
print("测试数据集的准确度= {:.2f}".format(accuracy))

# 显示图表来分析模型的训练过程
import matplotlib.pyplot as plt

# 显示训练和验证损失
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "b-", label="Training Loss")

plt.plot(epochs, val_loss, "r--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 显示训练和验证准确度
acc = history.history["accuracy"]
epochs = range(1, len(acc) + 1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training Acc")
plt.plot(epochs, val_acc, "r--", label="Validation Acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 6、存储模型
# 存储Keras 模型
print("Saving Model: titanic.h5 ...")
model.save("titanic.h5")

# 7、调用训练好的模型
seed = 7
np.random.seed(seed)
# 载入Titanic 的测试数据集
df_test = pd.read_csv("./titanic_test.csv")
dataset_test = df_test.values
# 分割成特征数据和标签数据
X_test = dataset_test[:, 0:9]
Y_test = dataset_test[:, 9]
# 特征标准化
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)
# 建立Keras 的Sequential 模型
model = Sequential()
model = load_model("titanic.h5")
# 编译模型
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])
# 评估模型
print("\nTesting ...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("测试数据集的准确度= {:.2f}".format(accuracy))
# 计算分类的预测值
print("\nPredicting ...")
Y_pred = model.predict(X_test)
print(Y_pred[:, 0])
print(Y_test.astype(int))
# 显示混淆矩阵
tb = pd.crosstab(Y_test.astype(int), Y_pred[:, 0],
                 rownames=["label"], colnames=["predict"])
print(tb)
tb.to_html("Ch6_2_4.html")
