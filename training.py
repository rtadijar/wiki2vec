from tqdm import tqdm

import numpy as np
import numpy.random as random

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

import io

from model import *
from util import load_embeddings


#embeddings = load_embeddings('embeddings/wiki-news-300d-1M.vec')[0]

dataset = torch.load('data/dataset.pt')
#dataset1M = TensorDataset(dataset.tensors[0][0:1000000].clone(), dataset.tensors[1][0:1000000].clone())



idx = torch.randperm(dataset.tensors[0].shape[0])

print(idx)

dataset_shuffled = TensorDataset(
                                    dataset.tensors[0][idx],
                                    dataset.tensors[1][idx]
                                )

dataset_shuffled_small = TensorDataset(dataset_shuffled.tensors[0][0:100].clone(), dataset_shuffled.tensors[1][0:100].clone())


with open('data/dataset_s.pt', 'wb') as of:
    torch.save(dataset_shuffled, of)

with open('data/dataset_s_small.pt', 'wb') as of:
    torch.save(dataset_shuffled, of)


#train_loader = DataLoader(dataset1M, sampler=RandomSampler(dataset1M), pin_memory=True)

device = torch.device('cpu')

title_embedding = torch.load('models/model_init.pt').to(device)
#title_embedding = TitleEmbedding(len(embeddings), 300, 12, 25, 25, embeddings=embeddings)
criterion = NegativeSamplingLoss(dataset1M.tensors[1], title_embedding, 5, device=device).to(device)



"""
class MultipleOptimizer:
    def __init__(self, *op):
      self.optimizers = op

    def zero_grad(self):
      for op in self.optimizers:
        op.zero_grad()

    def step(self):
      for op in self.optimizers:
        op.step()

sparse_params = []
dense_params = []

for name, param in title_embedding.named_parameters():
    if name == 'embeddings.weight':
        sparse_params.append(param)
    else:
        dense_params.append(param)

opt_sparse = torch.optim.SparseAdam(sparse_params, lr=1e-3)
opt_dense = torch.optim.Adam(dense_params, lr=1e-3)

optimizer = MultipleOptimizer(opt_sparse, opt_dense)


num_epochs = 5
batch_size = 32

title_embedding.train()

batch_loss = 0
optimizer.zero_grad()

try:
    for epoch in range(num_epochs):
        
        epoch_loss = 0        
        for i, data in enumerate(train_loader):
            x, y = data[0].to(device), data[1].to(device)
            loss = criterion(x, y)
            loss.backward()
        
            batch_loss += loss.item()
            epoch_loss += loss.item()
            
            if (i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

                print('loss at batch {}: {}'.format((i+1)/batch_size, batch_loss/batch_size))
                batch_loss = 0   
                
        print('loss at end of epoch {}: {}'.format(epoch, epoch_loss/len(train_loader)))
        
        with open('models/model1M_epoch{}.pt'.format(epoch), 'wb') as of:
            torch.save(title_embedding, of)

except KeyboardInterrupt:
    with open('models/model1M_interrupted{}.pt'.format(epoch), 'wb') as of:
        torch.save(title_embedding, of)
"""