import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils import cal_cos_similarity
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=512, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                360 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())
        self.decoder =  nn.Linear(hidden_size, 1)

    def forward(self, x, m_items_1,train=False):
        # feature
        # [N, F]
        x = self.mlp(x).squeeze()
        
        pred_min_p = read(x,m_items_1)
        pred_min_n = un_read(x,m_items_1)
        if train:
            m_items_1 = upload(x,m_items_1)
        
        pred = self.decoder(x)
        
        return pred,pred_min_n,pred_min_p,base,m_items_1
    
def get_score(K, Q):
    score = torch.matmul(Q,torch.t(K) )
    score_query = F.softmax(score, dim=0)
    score_memory = F.softmax(score,dim=1)
    return score_query, score_memory

def get_unscore(K, Q):
    score = -1*torch.matmul(Q,torch.t(K) )
    score_query = F.softmax(score, dim=0)
    score_memory = F.softmax(score,dim=1)
    return score_query, score_memory

def read(Q,K):
    softmax_score_query, softmax_score_memory = get_score(K,Q)
    query_reshape = Q.contiguous()
    concat_memory = torch.matmul(softmax_score_memory.detach(), K) # (b X h X w) X d
    updated_query = query_reshape*concat_memory
    return updated_query,softmax_score_query,softmax_score_memory

def un_read(Q,K):
    softmax_score_query, softmax_score_memory = get_unscore(K,Q)
    query_reshape = Q.contiguous()
    concat_memory = torch.matmul(softmax_score_memory.detach(), K) # (b X h X w) X d
    updated_query = query_reshape*concat_memory
    return updated_query,softmax_score_query,softmax_score_memory


def upload(Q,K):
    softmax_score_query, softmax_score_memory = get_score(K,Q)
    query_reshape = Q.contiguous()
    _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
    _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
    m, d = K.size()
    query_update = torch.zeros((m,d)).cuda()
    random_update = torch.zeros((m,d)).cuda()
    for i in range(m):
        idx = torch.nonzero(gathering_indices.squeeze(1)==i)
        a, _ = idx.size()
        if a != 0:
            query_update[i] = torch.sum(((softmax_score_query[idx,i] / torch.max(softmax_score_query[:,i]))*query_reshape[idx].squeeze(1)), dim=0)
        else:
            query_update[i] = 0 
    updated_memory = F.normalize(query_update.cuda() + K.cuda(), dim=1)
    return updated_memory.detach() 
