import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv("../house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../house-prices-advanced-regression-techniques/test.csv")

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_feature = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_feature] = all_features[numeric_feature].apply(
    lambda x: (x - x.mean())/ (x.std()))

all_features[numeric_feature] = all_features[numeric_feature].fillna(0)


all_features = pd.get_dummies(all_features, dummy_na=True)

loss = nn.MSELoss()

num_train = train_data.shape[0]
train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[num_train:].values, dtype = torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

def log_mse(net, X, y):
    with torch.no_grad():
        y_hat = net(X)
        clipped = torch.max(y_hat, torch.tensor(1.0))
        ret = torch.sqrt(loss(clipped.log(), y.log())).item()
    return ret

def train(net, train_features, train_labels, test_features, test_labels, batch_size, num_epochs, learning_rate, weight_decay):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        train_ls.append(log_mse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_mse(net, test_features, test_labels))

    return train_ls, test_ls

def get_k_fold_data(i, k, X, y):
    assert k > 1
    X_train, y_train = None, None
    fold_size = X.shape[0] // k
    X_valid, y_valid = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        X_part = X[idx, :]
        y_part = y[idx]

        if j == i:
            X_valid = X_part
            y_valid = y_part
        elif X_train == None:
            X_train = X_part
            y_train = y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid



def k_fold(k, X_train, y_train, num_fold, learning_rate, batch_size, weight_decay):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(i, k, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, batch_size=batch_size, num_epochs=num_fold, learning_rate=learning_rate, weight_decay=weight_decay)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, learning_rate=lr, weight_decay=weight_decay, batch_size=batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))

def train_and_predict(train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _  = train(net=net, train_features=train_features, train_labels=train_labels,
                         test_features = None, test_labels=None, learning_rate=lr, weight_decay=weight_decay, batch_size=batch_size, num_epochs=num_epochs)
    print('train rmse %f' % train_ls[-1])
    prediction = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(prediction.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)

train_and_predict(train_features, train_labels,test_features,  test_data, num_epochs, lr, weight_decay, batch_size)
