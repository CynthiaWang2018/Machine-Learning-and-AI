import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_fname = 'TrecQA_train.txt'
dev_fname = 'TrecQA_dev.txt'
test_fname = 'TrecQA_test.txt'

def read_data(fname):
    data = pd.read_csv(fname, delimiter='\t', header = None)
    data.columns = ['question', 'answer', 'label']
    return data

train_data = read_data(train_fname)
dev_data = read_data(dev_fname)
test_data = read_data(test_fname)

def read_raw_data(fname):
    with open(fname) as f:
        data = f.read()
    return data

train_raw = read_raw_data(train_fname)
dev_raw = read_raw_data(dev_fname)
test_raw = read_raw_data(test_fname)

def load_glove_model(gloveFile):
    '''
    input: glove file
    output:dict
    '''
    print('Step1: Loading Glove Model')
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print('Step1 has Done', len(model), 'words loaded')
    return model

glove = load_glove_model('glove.6B.100d.txt')
vocabulary = tuple(['#pad']) + tuple(set(train_raw.split() + dev_raw.split() + test_raw.split()))
word2int = {word: idx for idx,word in enumerate(vocabulary)}
int2word = {idx:word for idx, word in enumerate(vocabulary)}

question_seq_len = 16
answer_seq_len = 40
batch_size = 64
num_layers = 2
output_size = 64
hidden_dim = 128
device = torch.device('cuda:1' if torch.cuda.is_available() else 'gpu')

# 将不够一个batch_size的截断
def truncate_for_batch(data, batch_size):
    end_idx = batch_size * (len(data) // batch_size)
    return data[:end_idx]
train_for_batch = truncate_for_batch(train_data, batch_size)
dev_for_batch = truncate_for_batch(dev_data, batch_size)
test_for_batch = truncate_for_batch(test_data, batch_size)

def pad_tokenize(data, word2int, question_seq_len, answer_seq_len):
    '''
    input: data(pd.DataFrame), word2int(dict), question_seq_len(int:16), answer_seq_len(int:40)
    process: the 'question' and 'answer' column of data, use word2int changing to token and then padding
             change label into lint
    output: two tokens(np.ndarray) after padding, label(np.ndarray) after int
    '''
    def tokenize(word2int, series):
        '''
        input: word2int(dict), series(pd.series)
        process: split every sentence, and then change into token using word2int 
        output: token list(list)
        '''
        return [[word2int[word] for word in sentence.strip().split()] for sentence in series]
    
    def pad(token, seq_len):
        '''
        input: token(list), pad len(int)
        process: padding or truncate
        output:token(list) after padding or truncate
        '''
        n_token = len(token)
        if n_token > seq_len:
            return token[:seq_len]
        elif n_token < seq_len:
            return token + [0] * (seq_len - n_token)
        else:
            return token
        
    def pad_token(tokens, seq_len):
        '''
        input:token list of list(list), pad len(int)
        process:paddding or truncate 
        output:tkoen after padding or truncate(np.ndarray)
        '''
        return np.array(list(pad(t, seq_len) for t in tokens))
    
    x1, x2, y = data['question'], data['answer'], data['label']
    tokenize_x1_pad = pad_token(tokenize(word2int, x1), question_seq_len)
    tokenize_x2_pad = pad_token(tokenize(word2int, x2), answer_seq_len)
    tokenize_y_pad = y.map({1.0: 1.0, 0:-1}).values
    #print('Step2: Finish tokenize and padding or truncating')
    return tokenize_x1_pad, tokenize_x2_pad, tokenize_y_pad
train_x1, train_x2, train_y = pad_tokenize(train_for_batch, word2int, question_seq_len, answer_seq_len)
dev_x1, dev_x2, dev_y = pad_tokenize(dev_for_batch, word2int, question_seq_len, answer_seq_len)
test_x1, test_x2, test_y = pad_tokenize(test_for_batch, word2int, question_seq_len, answer_seq_len)
print('Step2: Finish tokenize and padding or truncating')

train_dataset = TensorDataset(torch.from_numpy(train_x1), torch.from_numpy(train_x2), torch.from_numpy(train_y))
dev_dataset = TensorDataset(torch.from_numpy(dev_x1), torch.from_numpy(dev_x2), torch.from_numpy(dev_y))
test_dataset = TensorDataset(torch.from_numpy(test_x1), torch.from_numpy(test_x2), torch.from_numpy(test_y))
train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 6)
dev_loader = DataLoader(dev_dataset, shuffle = False, batch_size = batch_size, num_workers = 6)
test_loader = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 6)

class LSTMLayer(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_size, vocabulary, glove, drop_prob=0.5):
        super().__init__()
        self.embedding_dim = len(list(glove.values())[0])
        weight_matrix = self.get_embedding_matrix(vocabulary, glove)
        self.embedding = self.create_emb_layer(weight_matrix)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = self.tanh(out)
        return out, hidden
    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
               torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
    def get_embedding_matrix(self, vocabulary, glove):
        '''
        input: V(tuple or list); Glove(dict)
        process:initial a new matrix, the ith row is the word embedding corresponding to the Glove dict
                if not found, set random number
        output:the matrix
        '''
        n_vocabulary = len(vocabulary)
        weights_matrix = np.zeros((n_vocabulary, self.embedding_dim))
        words_found = 0
        
        for i, word in enumerate(vocabulary):
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.embedding_dim,))
        print('{} in {} words have found word embedding in Glove'.format(words_found, n_vocabulary),
              '{} words use random initial embedding'.format(n_vocabulary - words_found))
        return weights_matrix
    def create_emb_layer(self, weights_matrix, non_trainable=False):
        '''
        input:weights_matrix(np.ndarray) 
        process:change into OrderDict,use load_state_dict -> nn.embedding
        output:nn.Embedding
        '''
        num_embeddings, embedding_dim = weights_matrix.shape
        from collections import OrderedDict
        weight_ordered_dict = OrderedDict([('weight', torch.from_numpy(weights_matrix))])
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict(weight_ordered_dict)
        if non_trainable:
            emb_layer.requires_grad = False
        return emb_layer
    
model = LSTMLayer(hidden_dim, num_layers, output_size, vocabulary, glove)
model.to(device)
model.train()
cos = nn.CosineSimilarity(dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

train_losses = []
dev_losses = []
train_accuracy_list = []
dev_accuracy_list = []
train_loss = 0
dev_loss = 0
test_loss = 0
train_accuracy = 0
dev_accuracy = 0
test_accuracy = 0
step = 0
print_every = 10
epochs = 10
for epoch in range(epochs):
    hidden = model.init_hidden(batch_size, device)
    for x1, x2, y in iter(train_loader):
        optimizer.zero_grad()
        step += 1
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        hidden = tuple([each.data for each in hidden])
        ques_out, hidden = model.forward(x1, hidden)
        ans_out, hidden = model.forward(x2, hidden)
        cos_out = cos(ques_out, ans_out)
        loss = criterion(cos_out, y.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc = torch.mean(((cos_out < 0) == (y < 0)).float())# add > 0
        train_accuracy += train_acc
        if step % print_every == 0:
            model.eval()
            with torch.no_grad():
                len_dev_loader = len(dev_loader)
                for x1, x2, y in iter(dev_loader):
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    y = y.to(device)
                    hidden_dev = model.init_hidden(batch_size, device)
                    ques_out_dev, hidden_dev = model.forward(x1, hidden_dev)
                    ans_out_dev, hidden_dev = model.forward(x2, hidden_dev)
                    cos_out_dev = cos(ques_out_dev, ans_out_dev)
                    loss = criterion(cos_out_dev, y.float())
                    dev_acc = torch.mean(((cos_out_dev < 0) == (y < 0)).float())
                    dev_loss += loss.item()
                    dev_accuracy += dev_acc.item()
            train_loss_mean = train_loss / print_every
            train_accuracy_mean = train_accuracy / print_every
            dev_loss_mean = dev_loss / len_dev_loader
            dev_accuracy_mean = dev_accuracy / len_dev_loader
            train_losses.append(train_loss_mean)
            train_accuracy_list.append(train_accuracy_mean)
            dev_losses.append(dev_loss_mean)
            dev_accuracy_list.append(dev_accuracy_mean)
            print('Epoch: {}/{}\tStep:{}'.format(epoch+1, epochs, step),
                 '\tTrain loss is:{:.3f}\t Train accuracy is:{:.3f}'.format(train_loss_mean, train_accuracy_mean),
                 '\tDev loss is:{:.3f}\tDev accuracy is{:.3f}'.format(dev_loss_mean, dev_accuracy_mean))
            train_loss = 0
            train_accuracy = 0
            dev_loss = 0
            dev_accuracy = 0
            model.train()
    scheduler.step()
    print('Step3: train one epoch')
for x1, x2, y in iter(test_loader):
    x1 = x1.to(device)
    x2 = x2.to(device)
    y = y.to(device)
    hidden_test = model.init_hidden(batch_size, device)
    ques_out_test, hidden_test = model.forward(x1, hidden_test)
    ans_out_test, hidden_test = model.forward(x2, hidden_test)
    cos_out_test = cos(ques_out_test, ans_out_test)
    loss = criterion(cos_out_test, y.float())
    test_acc = torch.mean(((cos_out_test < 0) == (y < 0)).float())
    test_loss += loss.item()
    test_accuracy += test_acc.item()
print('Test aver loss:{:.3f}\t'.format(test_loss / len(test_loader)),
     'Test aver acc:{:.3f}'.format(test_accuracy / len(test_loader)))
losses_df = pd.DataFrame({'train loss': train_losses, 'dev_loss': dev_losses})
losses_df.plot()
plt.savefig('./Method2.png', dpi=600)