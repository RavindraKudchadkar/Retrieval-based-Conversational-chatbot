from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import pickle
import torch.autograd as autograd
import datetime
from load_data import single_turn_train_loader
import sys


class ConfigLSTM():
    def __init__(self):
        self.word_embedding_size = 50
        self.hidden_units = 50
        self.vocab_size = 992
        self.batch_size = 40

class Encoder(nn.Module):
  def __init__(self, config): 
    super(Encoder, self).__init__()
    self.emb_size = config.word_embedding_size
    self.hidden_size = config.hidden_units
    self.vocab_size = config.vocab_size
    self.p_dropout = 0.1
    with open("word_embeddings","rb") as f:
      embedding_weights = pickle.load(f)
    self.embedding=nn.Embedding(self.vocab_size,self.emb_size)
    self.embedding.weight=nn.Parameter(torch.FloatTensor(np.array(embedding_weights).astype(np.float32)),requires_grad=True)
    self.lstm = nn.LSTM(self.emb_size, self.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
    self.dropout_layer = nn.Dropout(self.p_dropout) 

    self.init_weights()
      
  def init_weights(self):
    init.uniform_(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
    init.orthogonal_(self.lstm.weight_hh_l0)
    self.lstm.weight_ih_l0.requires_grad = True
    self.lstm.weight_hh_l0.requires_grad = True
    
          
  def forward(self, inputs):
    embeddings = self.embedding(inputs)
    _, (last_hidden,_) = self.lstm(embeddings) 
    last_hidden = self.dropout_layer(last_hidden[-1])
    return last_hidden


class DualEncoder(nn.Module):
  def __init__(self, encoder):
    super(DualEncoder, self).__init__()
    self.encoder = encoder
    self.hidden_size = self.encoder.hidden_size
    M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
    init.xavier_normal_(M)
    self.M = nn.Parameter(M, requires_grad = True)

  def forward(self, context_tensor, response_tensor):
    context_last_hidden = self.encoder(context_tensor) #dimensions: (batch_size x hidden_size)
    response_last_hidden = self.encoder(response_tensor) #dimensions: (batch_size x hidden_size)
    score=torch.einsum('nh,hh,nh->n',context_last_hidden,self.M,response_last_hidden).unsqueeze(1)
    
    return score