import torch
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import pickle


def train_loader(batch_size):
    with open("context","rb") as f:
      context=pickle.load(f)

    with open("context_lengths","rb") as f:
      context_lengths=pickle.load(f)

    with open("response","rb") as f:
      response=pickle.load(f)

    with open("response_lengths","rb") as f:
      response_lengths=pickle.load(f)

    with open("label","rb") as f:
      label=pickle.load(f)

    with open("vocab","rb") as f:
      vocab=pickle.load(f) 

    context=torch.LongTensor(np.array(context).astype(np.float32))
    context_lengths=torch.LongTensor(np.array(context_lengths).astype(np.float32))
    response_lengths=torch.LongTensor(np.array(response_lengths).astype(np.float32))
    response=torch.LongTensor(np.array(response).astype(np.float32))
    label=torch.FloatTensor(np.array(label).astype(np.float32))
    train=TensorDataset(context,response,label,context_lengths,response_lengths)
    train_loader=DataLoader(train,batch_size=batch_size)    
    return train_loader

def test_loader():
    with open("test_context","rb") as f:
      context=pickle.load(f)

    with open("test_context_lengths","rb") as f:
      context_lengths=pickle.load(f)

    with open("test_response","rb") as f:
      response=pickle.load(f)

    with open("test_response_lengths","rb") as f:
      response_lengths=pickle.load(f)

    with open("test_label","rb") as f:
      label=pickle.load(f)

    context=torch.LongTensor(np.array(context).astype(np.float32))
    context_lengths=torch.LongTensor(np.array(context_lengths).astype(np.float32))
    response=torch.LongTensor(np.array(response).astype(np.float32))
    response_lengths=torch.LongTensor(np.array(response_lengths).astype(np.float32))
    label=torch.FloatTensor(np.array(label).astype(np.float32))

    return context,response,label,context_lengths,response_lengths

def single_turn_train_loader(batch_size):
    with open("single_turn_context","rb") as f:
      context=pickle.load(f)

    with open("response","rb") as f:
      response=pickle.load(f)

    with open("label","rb") as f:
      label=pickle.load(f)


    context=torch.LongTensor(np.array(context).astype(np.float32))
    response=torch.LongTensor(np.array(response).astype(np.float32))
    label=torch.FloatTensor(np.array(label).astype(np.float32))
    train=TensorDataset(context,response,label)
    train_loader=DataLoader(train,batch_size=batch_size)    
    return train_loader

def single_turn_test_loader():
    with open("test_single_turn_context","rb") as f:
      context=pickle.load(f)

    with open("test_response","rb") as f:
      response=pickle.load(f)

    with open("test_label","rb") as f:
      label=pickle.load(f)

    context=torch.LongTensor(np.array(context).astype(np.float32))
    response=torch.LongTensor(np.array(response).astype(np.float32))
    label=torch.FloatTensor(np.array(label).astype(np.float32))

    return context,response,label
