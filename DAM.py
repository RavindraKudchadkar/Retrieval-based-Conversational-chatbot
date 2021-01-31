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

class ConfigDAM():
  def __init__(self):
    self.max_num_utterance = 10
    self.negative_samples = 1
    self.max_sentence_len = 50
    self.word_embedding_size = 50
    self.vocab_size = 992
    self.batch_size = 40
    self.stack_num=6
    self.dropout=0.1

class DAM(nn.Module):
  def __init__(self,config):
    super(DAM,self).__init__()
    self.max_num_utt=config.max_num_utterance
    self.max_len=config.max_sentence_len
    self.emb_size=config.word_embedding_size
    self.vocab_size=config.vocab_size
    self.stack_num=config.stack_num
    self.batch_size=config.batch_size
    self.dropout=config.dropout

    with open("word_embeddings","rb") as f:
      embedding_weights = pickle.load(f)
    self.embedding=nn.Embedding(self.vocab_size,self.emb_size)
    self.embedding.weight=nn.Parameter(torch.FloatTensor(np.array(embedding_weights).astype(np.float32)),requires_grad=True)

  
    self.SelfAttentiveModule=AttentiveModule2(self.emb_size,)
    self.CrossAttentiveModule=AttentiveModule2(self.emb_size)

    self.conv3d_1=nn.Conv3d(self.max_num_utt,32,kernel_size=(3,3,3))
    self.pool3d_1=nn.MaxPool3d(kernel_size=(3,3,3),stride=(3,3,3),ceil_mode=True)
    self.conv3d_2=nn.Conv3d(32,16,kernel_size=(3,3,3))
    self.pool3d_2=nn.MaxPool3d(kernel_size=(3,3,3),stride=(3,3,3),ceil_mode=True)

    convweight1 = (param.data for name, param in self.conv3d_1.named_parameters() if "weight" in name)
    for w in convweight1:
        nn.init.kaiming_normal_(w)

    convweight2 = (param.data for name, param in self.conv3d_2.named_parameters() if "weight" in name)
    for w in convweight2:
        nn.init.kaiming_normal_(w)    

    self.convolution=nn.Sequential(self.conv3d_1,nn.ReLU(),
                                   self.pool3d_1,
                                   self.conv3d_2,nn.ReLU(),
                                   self.pool3d_2)
    self.linear=nn.Linear(400,1)

  def forward(self,context,response,context_lens,response_lens):
    Hr=self.embedding(response)
    Hr_stack=[Hr]
    for i in range(self.stack_num):
      Hr=self.SelfAttentiveModule(Hr,Hr,Hr)
      Hr_stack.append(Hr)

    context=context.permute(1,0,2)
    context_lens=context_lens.permute(1,0)
    sim_turns=[]
    for utt,utt_len in zip(context,context_lens):
      Hu=self.embedding(utt)
  
      Hu_stack=[Hu]
      for i in range(self.stack_num):
        Hu=self.SelfAttentiveModule(Hu,Hu,Hu)
        Hu_stack.append(Hu)
      
      r_a_t_stack,t_a_r_stack=[],[]

      for i in range(self.stack_num+1):
        t_a_r=self.CrossAttentiveModule(Hu_stack[i],Hr_stack[i],Hr_stack[i])
        r_a_t=self.CrossAttentiveModule(Hr_stack[i],Hu_stack[i],Hu_stack[i])
        t_a_r_stack.append(t_a_r)
        r_a_t_stack.append(r_a_t)
      
      t_a_r_stack.extend(Hu_stack)
      r_a_t_stack.extend(Hr_stack)

      t_a_r=torch.stack(t_a_r_stack,dim=-1)
      r_a_t=torch.stack(r_a_t_stack,dim=-1)

      sim=torch.einsum('biks,bjks->bijs',t_a_r,r_a_t)
      sim_turns.append(sim)

    sim=torch.stack(sim_turns,dim=1)
    sim=self.convolution(sim)
    logits=self.linear(torch.flatten(sim,start_dim=1))

    return logits