# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:15:04 2019

@author: WangZhao
"""
# Attention: 此Linux版本代码的epoch较大，可将epoch调小后，在windows系统运行
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import gensim
from itertools import chain
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import pandas as pd

# from torch.utils.data import TensorDataset, DataLoader
# Step1 Data Precessing--------------------------------------------------------

def read_data(path):
    '''
    读取数据
    输入path: 数据的相对路径
    输出data: 将数据存储到list
    '''
    data = []
    for line in open(path, 'r'):
        line = line.strip('\n')
        item = line.split('\t')
        data.append([item[0] + item[1], item[2]])
    return data


TrecQA_train_path = './TrecQA_train.txt'
TrecQA_dev_path = './TrecQA_dev.txt'
TrecQA_test_path = './TrecQA_test.txt'

train_data = read_data(TrecQA_train_path)
dev_data = read_data(TrecQA_dev_path)
test_data = read_data(TrecQA_test_path)

train_tokenized = []
dev_tokenized = []
test_tokenized = []
for QA, label in train_data:
    train_tokenized.append(QA)
for QA, label in dev_data:
    dev_tokenized.append(QA)
for QA, label in test_data:
    test_tokenized.append(QA)

vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)

wvmodel = gensim.models.KeyedVectors.load_word2vec_format('glove_model2.txt',
                                                          binary=False, encoding='utf-8', unicode_errors='ignore')

word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'


def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=500, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while (len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
dev_features = torch.tensor(pad_samples(encode_samples(dev_tokenized, vocab)))
test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))

train_labels = torch.tensor([eval(score) for _, score in train_data], dtype=torch.long)
dev_labels = torch.tensor([eval(score) for _, score in dev_data], dtype=torch.long)
test_labels = torch.tensor([eval(score) for _, score in test_data], dtype=torch.long)


class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight, **kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.labels = labels
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv2d(1, 1, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 1, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 1, (5, embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.linear = nn.Linear(3, labels)

    def forward(self, inputs):
        inputs = self.embedding(inputs).view(inputs.shape[0], 1, inputs.shape[1], -1)
        x1 = F.relu(self.conv1(inputs))
        x2 = F.relu(self.conv2(inputs))
        x3 = F.relu(self.conv3(inputs))

        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)

        x = torch.cat((x1, x2, x3), -1)
        x = x.view(inputs.shape[0], 1, -1)

        x = self.linear(x)
        x = x.view(-1, self.labels)

        return (x)


num_epochs = 10  # 与Windows的区别  可以将epochs改大些
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.1
device = torch.device('cuda:0')
use_gpu = True

weight = torch.zeros(vocab_size + 1, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
net = textCNN(vocab_size=(vocab_size + 1), embed_size=embed_size,
              seq_len=500, labels=labels, weight=weight)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

train_set = torch.utils.data.TensorDataset(train_features, train_labels)
dev_set = torch.utils.data.TensorDataset(dev_features, dev_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
dev_iter = torch.utils.data.DataLoader(dev_set, batch_size=batch_size,
                                       shuffle=False)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False)

train_loss2, dev_loss2, test_loss2 = [], [], []
for epoch in range(num_epochs):
    start = time.time()
    train_loss, dev_losses = 0, 0,
    train_acc, dev_acc = 0, 0
    n, n2 = 0, 0

    for feature, label in train_iter:
        n += 1
        #         net.train()
        net.zero_grad()
        feature = Variable(feature.cuda())
        label = Variable(label.cuda())
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        #         scheduler.step()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                 dim=1), label.cpu())
        train_loss += loss
    # train_loss2.append((train_loss/n).tolist())
    with torch.no_grad():
        for dev_feature, dev_label in dev_iter:
            n2 += 1
            #             net.eval()
            dev_feature = dev_feature.cuda()
            dev_label = dev_label.cuda()
            dev_score = net(dev_feature)
            dev_loss = loss_function(dev_score, dev_label)
            dev_acc += accuracy_score(torch.argmax(dev_score.cpu().data,
                                                   dim=1), dev_label.cpu())
            dev_losses += dev_loss
    end = time.time()
    runtime = end - start

    train_loss2.append((train_loss / n).tolist())
    dev_loss2.append((dev_losses / n2).tolist())
    #test_loss2.append((test_losses / m).tolist())
    print(
        'epoch: %d, train loss: %.4f, train acc: %.2f, dev loss: %.4f, dev acc: %.2f, time: %.2f' %
        (epoch, train_loss.data / n, train_acc / n, dev_losses.data / n2, dev_acc / n2, runtime))

m = 0
test_losses = 0
test_acc = 0
for test_feature, test_label in test_iter:
    m += 1
    test_feature = test_feature.cuda()
    test_label = test_label.cuda()
    test_score = net(test_feature)
    test_loss = loss_function(test_score, test_label)
    test_acc += accuracy_score(torch.argmax(test_score.cpu().data,dim=1), test_label.cpu())
    test_losses += test_loss
#test_loss = test_losses / m
#test_loss2.append((test_losses / m).tolist())
print('test loss: %.4f, test acc: %.2f' %( test_losses.data / m,test_acc / m))
# 作图
# Figure1: x-axis: epoch number, y-axis: loss values in train set
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(np.arange(len(train_loss2)), train_loss2, 'b')
ax.set_xlabel('Epochs')
ax.set_ylabel('train_loss')
ax.set_title('train_loss vs. Epochs')
fig.savefig('train_loss.png')
# Figure1: x-axis: epoch number, y-axis: loss values in dev set
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(np.arange(len(dev_loss2)), dev_loss2, 'b')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('dev_loss')
ax2.set_title('dev_loss vs. Epochs')
fig2.savefig('dev_loss.png')

losses_df = pd.DataFrame({'train loss': train_loss2, 'dev loss': dev_loss2})
#with open("./losses_df.pkl", 'wb') as f:
#    pickle.dump(losses_df, f)
losses_df.plot()
plt.savefig("./losses.png", dpi=600)
plt.show()