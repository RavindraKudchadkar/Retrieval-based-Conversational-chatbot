from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import pickle


class ConfigSMN():
    def __init__(self):
        self.max_num_utterance = 10
        self.negative_samples = 1
        self.max_sentence_len = 50
        self.word_embedding_size = 50
        self.hidden_units = 50
        self.vocab_size = 992
        self.batch_size = 40

class SMN(nn.Module):
  def __init__(self,config):
    super(SMN, self).__init__()
    self.max_num_utterance=config.max_num_utterance
    self.negative_samples=config.negative_samples
    self.max_sentence_len=config.max_sentence_len
    self.word_embedding_size=config.word_embedding_size
    self.hidden_units=config.hidden_units
    self.vocab_size=config.vocab_size
    self.batch_size=config.batch_size

    with open("word_embeddings","rb") as f:
      embedding_weights = pickle.load(f)
    self.word_embedding=nn.Embedding(self.vocab_size,self.word_embedding_size)
    self.word_embedding.weight=nn.Parameter(torch.FloatTensor(np.array(embedding_weights).astype(np.float32)),requires_grad=True)
    

    self.utterance_gru=nn.GRU(self.word_embedding_size,self.hidden_units,batch_first=True)
    ih_u = (param.data for name, param in self.utterance_gru.named_parameters() if 'weight_ih' in name)
    hh_u = (param.data for name, param in self.utterance_gru.named_parameters() if 'weight_hh' in name)
    for k in ih_u:
        nn.init.orthogonal_(k)
    for k in hh_u:
        nn.init.orthogonal_(k)

    self.response_gru=nn.GRU(self.word_embedding_size,self.hidden_units,batch_first=True)
    ih_r = (param.data for name, param in self.response_gru.named_parameters() if 'weight_ih' in name)
    hh_r = (param.data for name, param in self.response_gru.named_parameters() if 'weight_hh' in name)
    for k in ih_r:
        nn.init.orthogonal_(k)
    for k in hh_r:
        nn.init.orthogonal_(k)    

    self.conv2d=nn.Conv2d(2,8,kernel_size=(3,3))
    self.maxpool2d=nn.MaxPool2d(kernel_size=(3,3),stride=(3,3))
    self.linear=nn.Linear(8*16*16,50)

    conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
    for w in conv2d_weight:
        nn.init.kaiming_normal_(w)

    linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
    for w in linear_weight:
        init.xavier_uniform_(w)    

    self.A=torch.zeros((self.hidden_units,self.hidden_units),requires_grad=True)
    nn.init.xavier_uniform_(self.A)

    self.last_gru=nn.GRU(50,self.hidden_units,batch_first=True)
    ih_l = (param.data for name, param in self.last_gru.named_parameters() if 'weight_ih' in name)
    hh_l = (param.data for name, param in self.last_gru.named_parameters() if 'weight_hh' in name)
    for k in ih_l:
        nn.init.orthogonal_(k)
    for k in hh_l:
        nn.init.orthogonal_(k)

    self.last_linear=nn.Linear(50,1)
    final_linear_weight = (param.data for name, param in self.last_linear.named_parameters() if "weight" in name)
    for w in final_linear_weight:
        init.xavier_uniform_(w)

  def forward(self,utterance,response,utterance_lens,response_lens):
    batch_utterance_embeddings=self.word_embedding(utterance)
    response_embedding=self.word_embedding(response)
    batch_utterance_embeddings=batch_utterance_embeddings.permute(1,0,2,3)
    utterance_lens=utterance_lens.permute(1,0)
    response_gru_emb,_=self.response_gru(response_embedding)
    match_vectors=[]

    for utt_emb in batch_utterance_embeddings:
      mat1=torch.einsum('nij,nkj->nik',utt_emb,response_embedding)
      utt_gru_emb,_=self.utterance_gru(utt_emb)
      mat2=torch.einsum('nij,jk,nkl->nil',utt_gru_emb,self.A,response_gru_emb)
      
      mat=torch.stack([mat1,mat2],dim=1)
      conv_layer=torch.relu(self.conv2d(mat))
      pool_layer=self.maxpool2d(conv_layer)
      flattened=torch.flatten(pool_layer,start_dim=1)
      matching_vector=torch.tanh(self.linear(flattened))
      match_vectors.append(matching_vector)
    
    
    _,hidden_state=self.last_gru(torch.stack(match_vectors,dim=1))
    logits=self.last_linear(torch.squeeze(hidden_state))

    return logits

