from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import torch.autograd as autograd
import datetime
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

class ConfigMSN():
  def __init__(self):
    self.max_num_utterance = 10
    self.negative_samples = 1
    self.max_sentence_len = 50
    self.word_embedding_size = 50
    self.vocab_size = 992
    self.batch_size = 5
    self.num_layers=6
    self.dropout=0.1
    self.alpha=0.5
    self.gamma=0.3

class MSN(nn.Module):
  
    def __init__(self,config):
        super(MSN,self).__init__()
        self.vocab_size=config.vocab_size
        self.emb_size=config.word_embedding_size
        self.hidden_units=config.word_embedding_size

        with open("word_embeddings","rb") as f:
           embedding_weights = pickle.load(f)
        self.word_embedding=nn.Embedding(self.vocab_size,self.emb_size)
        self.word_embedding.weight=nn.Parameter(torch.FloatTensor(np.array(embedding_weights).astype(np.float32)),requires_grad=True)

        self.alpha = config.alpha
        self.gamma = config.gamma
        self.selector_transformer = AttentiveModule2(self.emb_size,config.dropout)
        self.W_word = nn.Parameter(data=torch.Tensor(self.emb_size, self.emb_size, 10))
        self.v = nn.Parameter(data=torch.Tensor(10, 1))
        self.linear_word = nn.Linear(2*50, 1)
        self.linear_score = nn.Linear(in_features=3, out_features=1)

        self.transformer_utt =  AttentiveModule2(self.emb_size,config.dropout)
        self.transformer_res = AttentiveModule2(self.emb_size,config.dropout)
        self.transformer_ur =  AttentiveModule2(self.emb_size,config.dropout)
        self.transformer_ru =  AttentiveModule2(self.emb_size,config.dropout)

        self.A1 = nn.Parameter(data=torch.Tensor(self.emb_size, self.emb_size))
        self.A2 = nn.Parameter(data=torch.Tensor(self.emb_size, self.emb_size))
        self.A3 = nn.Parameter(data=torch.Tensor(self.emb_size, self.emb_size))

        self.cnn_2d_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3,3))
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.cnn_2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.cnn_2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.affine2 = nn.Linear(in_features=3*3*64, out_features=300)

        self.gru_acc = nn.GRU(input_size=300, hidden_size=self.hidden_units, batch_first=True)
        
        self.affine_out = nn.Linear(in_features=self.hidden_units, out_features=1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.init_weights()
        


    def init_weights(self):
        init.uniform_(self.W_word)
        init.uniform_(self.v)
        init.uniform_(self.linear_word.weight)
        init.uniform_(self.linear_score.weight)

        init.xavier_normal_(self.A1)
        init.xavier_normal_(self.A2)
        init.xavier_normal_(self.A3)
        init.xavier_normal_(self.cnn_2d_1.weight)
        init.xavier_normal_(self.cnn_2d_2.weight)
        init.xavier_normal_(self.cnn_2d_3.weight)
        init.xavier_normal_(self.affine2.weight)
        init.xavier_normal_(self.affine_out.weight)
        for weights in [self.gru_acc.weight_hh_l0, self.gru_acc.weight_ih_l0]:
            init.orthogonal_(weights)


    def word_selector(self, key, context):
        
        dk = torch.sqrt(torch.Tensor([self.emb_size]))
        A = torch.tanh(torch.einsum("blrd,ddh,bud->blruh", context, self.W_word, key)/dk)
        # print(A.size())
        A = torch.einsum("blruh,hp->blrup", A, self.v).squeeze(dim=-1)  
        # print(A.size())
        a = torch.cat([A.max(dim=2)[0], A.max(dim=3)[0]], dim=-1) 
        s1 = torch.softmax(self.linear_word(a).squeeze(), dim=-1)
        return s1

    def utterance_selector(self, key, context):

        key = key.mean(dim=1)
        context = context.mean(dim=2)
        # print(context.size())
        s2 = torch.einsum("bud,bd->bu", context, key)/(1e-6 + torch.norm(context, dim=-1)*torch.norm(key, dim=-1, keepdim=True) )
        return s2

    def distance(self, A, B, C, epsilon=1e-6):
        M1 = torch.einsum("bud,dd,brd->bur", [A, B, C])

        A_norm = A.norm(dim=-1)
        C_norm = C.norm(dim=-1)
        M2 = torch.einsum("bud,brd->bur", [A, C]) / (torch.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
        # print(M2.size())
        return M1, M2

    def context_selector(self, context, hop=[1, 2, 3]):
      
        su1, su2, su3, su4 = context.size()
        context_ = context.view(-1, su3, su4)   # (batch_size*max_utterances, max_u_words, embedding_dim)
        context_ = self.selector_transformer(context_, context_, context_)
        context_ = context_.view(su1, su2, su3, su4)

        multi_match_score = []
        for hop_i in hop:
            key = context[:, 10-hop_i:, :, :].mean(dim=1)
            key = self.selector_transformer(key, key, key)

            s1 = self.word_selector(key, context_)
            s2 = self.utterance_selector(key, context_)
            s = self.alpha * s1 + (1 - self.alpha) * s2
            multi_match_score.append(s)

        multi_match_score = torch.stack(multi_match_score, dim=-1)
        match_score = self.linear_score(multi_match_score).squeeze()
        mask = (match_score.sigmoid() >= self.gamma).float()
        match_score = match_score * mask
        context = context * match_score.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return context

    def get_Matching_Map(self, bU_embedding, bR_embedding):
      
        
        M1, M2 = self.distance(bU_embedding, self.A1, bR_embedding)

        Hu = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
        Hr = self.transformer_res(bR_embedding, bR_embedding, bR_embedding)
        
        M3, M4 = self.distance(Hu, self.A2, Hr)

        Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)
        Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
        
        M5, M6 = self.distance(Hur, self.A3, Hru)

        M = torch.stack([M1, M2, M3, M4, M5, M6], dim=1) 
        return M


    def UR_Matching(self, bU_embedding, bR_embedding):
      
        M = self.get_Matching_Map(bU_embedding, bR_embedding)

        Z = self.relu(self.cnn_2d_1(M))
        Z = self.maxpooling1(Z)

        Z = self.relu(self.cnn_2d_2(Z))
        Z =self.maxpooling2(Z)

        Z = self.relu(self.cnn_2d_3(Z))
        Z =self.maxpooling3(Z)

        Z = Z.view(Z.size(0), -1) 

        V = self.tanh(self.affine2(Z))   
        return V

    def forward(self, bU, bR, bUl, bRl):

        
        bU_embedding = self.word_embedding(bU) 
        bR_embedding = self.word_embedding(bR) 
        multi_context = self.context_selector(bU_embedding, hop=[1, 2, 3])

        su1, su2, su3, su4 = multi_context.size()
        multi_context = multi_context.view(-1, su3, su4)   

        sr1, sr2, sr3= bR_embedding.size()   
        bR_embedding = bR_embedding.unsqueeze(dim=1).repeat(1, su2, 1, 1)  
        bR_embedding = bR_embedding.view(-1, sr2, sr3)   

        V = self.UR_Matching(multi_context, bR_embedding)
        V = V.view(su1, su2, -1)  
    
        H, _ = self.gru_acc(V) 
        L = self.dropout(H[:,-1,:])

        output = torch.sigmoid(self.affine_out(L))
        return output.view(-1,1)