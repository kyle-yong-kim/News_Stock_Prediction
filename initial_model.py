#all imports
import numpy as np
import time
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import json
import torchtext
import torchtext.data as data
import torchtext.vocab as vocab

import random

# for padding
from torch.nn.utils.rnn import pad_sequence

import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from spacy.lang.en import English

# class customData(Dataset):
#     def __init__(self):
#     def __getitem__(self, index):
#         return
#     def __len__(self):
#         return self.len

def loadJson(fileLoc):
    with open(fileLoc) as f:
        data = json.load(f)
        return data

class TweetBatcher:
    def __init__(self, tweets, batch_size, drop_last=False):
        # store tweets by length
        self.tweets_by_length = {}
        for words, label in tweets:
            # compute the length of the tweet
            wlen = words.shape[0]
            # put the tweet in the correct key inside self.tweet_by_length
            if wlen not in self.tweets_by_length:
                self.tweets_by_length[wlen] = []
            self.tweets_by_length[wlen].append((words, label),)
         
        #  create a DataLoader for each set of tweets of the same length
        self.loaders = {wlen : torch.utils.data.DataLoader(
                                    tweets,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=drop_last) # omit last batch if smaller than batch_size
            for wlen, tweets in self.tweets_by_length.items()}
        
    def __iter__(self): # called by Python to create an iterator
        # make an iterator for every tweet length
        iters = [iter(loader) for loader in self.loaders.values()]
        while iters:
            # pick an iterator (a length)
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                # no more elements in the iterator, remove it
                iters.remove(im)



def mainLoop():
    # first, load the stock json obtained from stock_data_crawler
    train, valid, test = [], [], []
    fileLoc = "C:\\Temp\\newsData_sample.json"
    modelLoc = "./GoogleNews-vectors-negative300.bin.gz"
    wordLimit = 5000
    stockJson = loadJson(fileLoc)
    model = gensim.models.KeyedVectors.load_word2vec_format(modelLoc, binary=True, limit = wordLimit)

    tokenizer = English()

    model_itos = model.index2word
    model_stoi = model.vocab
    errorList = []

    for i, item in enumerate(stockJson):
        # we need to tokenize this sentence
        # item['title']
        try:
            token_list = [word.text for word in tokenizer(item['title'])]

            # model_stoi = {"'m":1, "to":2, "our":8}

            idxs = [model_stoi[word].index for word in token_list if word in model_stoi]
            idxs = torch.tensor(idxs)
            label = torch.tensor(int(item['label'])).long()
            if i % 5 < 3:
                train.append((idxs, label))
            else:
                valid.append((idxs, label))
        except:
            errorList.append(item)

    # model_emb = nn.Embedding.from_pretrained(model.vectors)
    return train, valid, test, model

def get_accuracy(model, data_loader):
    correct, total = 0, 0
    errLog = []
    for tweets, labels in data_loader:
        try:
            output = model(tweets)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.shape[0]
        except:
            errLog.append((tweets, labels))
    return correct / total

train, valid, test, model = mainLoop()

weights = torch.FloatTensor(model.vectors)
embedding = nn.Embedding.from_pretrained(weights)

train_loader = TweetBatcher(train, batch_size=16, drop_last=False)
valid_loader = TweetBatcher(valid, batch_size=16, drop_last=False)

class TweetRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TweetRNN, self).__init__()
        self.emb = embedding

        # 300 since vector is of size 300
        # self.emb = torch.eye(300)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

def train_rnn_network(model, train_loader, valid_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    errLog = []
    for epoch in range(num_epochs):
        for tweets, labels in train_loader:
            try:
                optimizer.zero_grad()
                pred = model(tweets)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            except:
                errLog.append((tweets, labels))
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train_loader))
        valid_acc.append(get_accuracy(model, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))
    # plotting
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

model = TweetRNN(300, 1024, 2)

print(get_accuracy(model, train_loader))

train_rnn_network(model, train_loader, valid_loader, num_epochs = 70, learning_rate=3e-6)

# train_padded = pad_sequence([tweet for tweet, label in train[:30]], batch_first = True)

print("done")