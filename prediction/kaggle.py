import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 获取并读取数据集
train_data = pd.read_csv("../house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../house-prices-advanced-regression-techniques/test.csv")

torch.set_default_tensor_type(torch.FloatTensor)

# 竖向拼接，去掉 ID 以及 label，便于后续对整体数据进行处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数据预处理(标准化)，将所有的数字特征都进行标准化，并将NA 全部替换成0
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean())/(x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散值转换为二值
all_features = pd.get_dummies(all_features, dummy_na=True)


# 将训练数据(features, labels)转换为Tensor类型
num_train = train_data.shape[0]
train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[num_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)


# 定义训练模型
loss = nn.MSELoss()


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)

    return net


# 对数均方差损失函数
def log_mse(net, features, labels):
    with torch.no_grad():
        clipped = torch.max(net(features), torch.tensor(1.0))
        ret = torch.sqrt(loss(clipped.log(), labels.log()))
    return ret.item()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, batch_size, lr, weight_decay):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
    # 注意添加优化器
    optimizer = torch.optim.Adam(params=net.parameters(), lr = lr, weight_decay=weight_decay)
    # 注意将net 转换为 float 类型
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            # 清空梯度
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_mse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_mse(net, test_features, test_labels))
    return train_ls, test_ls


# K-fold verification
def get_k_fold_data(k, i, X, y):
    assert k > 1
    X_train = None
    y_train = None
    fold_size = X.shape[0] // k

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        if j == i:
            X_valid, y_valid = X[idx, :], y[idx]
        elif X_train is None:
            X_train, y_train  = X[idx, :], y[idx]
        else:
            X_train = torch.cat((X_train, X[idx, :]), dim=0)
            y_train = torch.cat((y_train, y[idx]), dim=0)
    return X_train, y_train, X_valid, y_valid


# 利用 k fold 验证进行训练
def k_fold(k, X_train, y_train, num_epochs, learning_rate, batch_size, weight_decay):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k=k, X=X_train, y=y_train, i=i)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs,
                                  batch_size, learning_rate, weight_decay)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum/k, valid_l_sum/k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, learning_rate=lr, weight_decay=weight_decay, batch_size=batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))


def train_and_predict(train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _  = train(net=net, train_features=train_features, train_labels=train_labels,
                         test_features = None, test_labels=None, lr=lr, weight_decay=weight_decay, batch_size=batch_size, num_epochs=num_epochs)
    print('train rmse %f' % train_ls[-1])
    prediction = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(prediction.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)

train_and_predict(train_features, train_labels,test_features,  test_data, num_epochs, lr, weight_decay, batch_size)
