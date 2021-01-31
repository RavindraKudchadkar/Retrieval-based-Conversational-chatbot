from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import pickle

class AttentiveModule2(nn.Module):

    def __init__(self, input_size, is_layer_norm=False):
        super(AttentiveModule2, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        dk = torch.Tensor([max(1.0, Q.size(-1))])

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)  
        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att) 
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output

class ConfigIOI():
  def __init__(self):
    self.max_num_utterance = 10
    self.negative_samples = 1
    self.max_sentence_len = 50
    self.word_embedding_size = 50
    self.vocab_size = 992
    self.batch_size = 5
    self.num_layers=6
    self.dropout=0.1
    
class IOI(nn.Module):
  def __init__(self,config):
    super(IOI,self).__init__()
    self.max_turn=config.max_num_utterance
    self.max_len=config.max_sentence_len
    self.num_layers=config.num_layers
    self.dropout=nn.Dropout(config.dropout)
    self.emb_size=config.word_embedding_size
    self.vocab_size=config.vocab_size
    self.hidden_size=config.word_embedding_size
    self.batch_size=config.batch_size

    with open("word_embeddings","rb") as f:
      embedding_weights = pickle.load(f)
    self.embedding=nn.Embedding(self.vocab_size,self.emb_size)
    self.embedding.weight=nn.Parameter(torch.FloatTensor(np.array(embedding_weights).astype(np.float32)),requires_grad=True)
    self.attention=AttentiveModule2(self.emb_size)

    self.linear1=nn.Linear(4*self.emb_size,self.emb_size)
    weight1 = (param.data for name, param in self.linear1.named_parameters() if "weight" in name)
    for w in weight1:
        init.xavier_uniform_(w)

    self.sequence1=nn.Sequential(self.linear1,nn.ReLU(),self.dropout)

    self.conv1=nn.Conv2d(3,32,kernel_size=(3,3),stride=(1,1))
    self.pool1=nn.MaxPool2d(kernel_size=(3,3),ceil_mode=True)
    self.conv2=nn.Conv2d(32,16,kernel_size=(3,3),stride=(1,1))
    self.pool2=nn.MaxPool2d(kernel_size=(3,3),ceil_mode=True)
    nn.init.kaiming_normal_(self.conv1.weight)
    nn.init.kaiming_normal_(self.conv2.weight)

    self.convolution=nn.Sequential(self.conv1,nn.ReLU(),
                                    self.pool1,
                                    self.conv2,nn.ReLU(),
                                    self.pool2)
    self.linear2=nn.Linear(400,self.emb_size)
    nn.init.xavier_uniform_(self.linear2.weight)

    self.gru=nn.GRU(self.emb_size,self.hidden_size,batch_first=True)
    self.last_linear=nn.Linear(self.emb_size,1)

  def forward(self,context,response,context_lens,response_lens):
    context_emb=self.embedding(context)
    response_emb=self.embedding(response)
    context_emb=self.dropout(context_emb)
    response_emb=self.dropout(response_emb)
    expand_response_emb=response_emb.squeeze(1).repeat(1,self.max_turn,1,1).view(-1,self.max_len,self.emb_size)
    parall_context_emb=context_emb.view(-1,self.max_len,self.emb_size)

    logit_arr=[]
    for k in range(self.num_layers):
      inter_feat_collection=[]
      context_self_rep=self.attention(parall_context_emb,parall_context_emb,parall_context_emb)
      response_self_rep=self.attention(expand_response_emb,expand_response_emb,expand_response_emb)
      context_cross_rep=self.attention(parall_context_emb,expand_response_emb,expand_response_emb)
      response_cross_rep=self.attention(expand_response_emb,parall_context_emb,parall_context_emb)

      context_inter_feat_multi = torch.einsum('bij,bjk->bik',parall_context_emb,context_cross_rep)
      response_inter_feat_multi = torch.einsum('bij,bjk->bik',expand_response_emb,response_cross_rep)

      context_concat_rep=torch.stack([parall_context_emb,context_self_rep,context_cross_rep,context_inter_feat_multi],dim=-1).view(context_cross_rep.size()[0],self.max_len,-1)
      response_concat_rep=torch.stack([expand_response_emb,response_self_rep,response_cross_rep,response_inter_feat_multi],dim=-1).view(response_cross_rep.size()[0],self.max_len,-1)

      context_concat_dense_rep=self.sequence1(context_concat_rep)
      response_concat_dense_rep=self.sequence1(response_concat_rep)

      inter_feat=torch.einsum('bij,bjk->bik',parall_context_emb,parall_context_emb)/self.emb_size**(1/2)
      inter_feat_self=torch.einsum('bij,bjk->bik',context_self_rep,response_self_rep)/self.emb_size**(1/2)
      inter_feat_cross=torch.einsum('bij,bjk->bik',context_cross_rep,response_cross_rep)/self.emb_size**(1/2)

      inter_feat_collection=[inter_feat,inter_feat_self,inter_feat_cross]

      parall_context_emb=F.normalize(parall_context_emb+context_concat_dense_rep,dim=2)
      expand_response_emb=F.normalize(expand_response_emb+response_concat_dense_rep,dim=2)

      matching_feat=torch.stack(inter_feat_collection,dim=1)
      
      convolved=self.convolution(matching_feat)
      flattened=torch.flatten(convolved,start_dim=1)
      flattened=self.dropout(flattened)
      flattened=self.linear2(flattened.view(-1,self.max_turn,flattened.size()[-1]))

      _,hidden=self.gru(flattened)

      logits=self.last_linear(torch.squeeze(hidden))
      logit_arr.append(logits.detach().numpy())
  

    return torch.sum(torch.FloatTensor(logit_arr),dim=0)