import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, d_feat: int, hidden_size: int = 512, num_layers: int = 3, dropout: float = 0.0):
        """
        Multilayer Perceptron with optional dropout between layers.
        
        d_feat: number of input features
        hidden_size: number of neurons in hidden layers
        num_layers: number of layers
        dropout: dropout rate
        """
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_size = 360 if i == 0 else hidden_size  # Input size is 360 for first layer, hidden_size for subsequent layers
            layers.extend([nn.Linear(in_size, hidden_size), nn.ReLU()])
            if i > 0:  # Add dropout for all but the first layer
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, m_items_1: torch.Tensor, train: bool = False):
        """
        Forward pass.
        
        x: input features
        m_items_1: items for memory update
        train: whether model is in training mode
        """
        x = self.mlp(x).squeeze()
        
        pred_min_p, _, _ = read(x, m_items_1)
        pred_min_n, _, _ = un_read(x, m_items_1)
        
        if train:
            m_items_1 = upload(x, m_items_1)
        
        pred = self.decoder(x)
        
        return pred, pred_min_n, pred_min_p, m_items_1

def get_score(K: torch.Tensor, Q: torch.Tensor):
    """
    Calculate the score as the product of Q and K transposed.
    
    K: keys
    Q: queries
    """
    score = torch.matmul(Q, torch.t(K))
    score_query = F.softmax(score, dim=0)
    score_memory = F.softmax(score, dim=1)
    
    return score_query, score_memory

def get_unscore(K: torch.Tensor, Q: torch.Tensor):
    """
    Calculate the negative score as the product of Q and K transposed.
    
    K: keys
    Q: queries
    """
    return get_score(-1 * K, Q)

def read(Q: torch.Tensor, K: torch.Tensor):
    """
    Memory read operation.

    Q: queries
    K: keys
    """
    softmax_score_query, softmax_score_memory = get_score(K, Q)
    query_reshape = Q.contiguous()
    concat_memory = torch.matmul(softmax_score_memory.detach(), K)
    updated_query = query_reshape * concat_memory
    
    return updated_query, softmax_score_query, softmax_score_memory

def un_read(Q: torch.Tensor, K: torch.Tensor):
    """
    Memory un-read operation.

    Q: queries
    K: keys
    """
    softmax_score_query, softmax_score_memory = get_unscore(K, Q)
    query_reshape = Q.contiguous()
    concat_memory = torch.matmul(softmax_score_memory.detach(), K)
    updated_query = query_reshape * concat_memory
    
    return updated_query, softmax_score_query, softmax_score_memory

def upload(Q: torch.Tensor, K: torch.Tensor):
    """
    Memory update operation.

    Q: queries
    K: keys
    """
    softmax_score_query, softmax_score_memory = get_score(K, Q)
    query_reshape = Q.contiguous()
    _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
    _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
    m, d = K.size()
    query_update = torch.zeros((m, d)).cuda()
    for i in range(m):
        idx = torch.nonzero(gathering_indices.squeeze(1) == i)
        if idx.size(0) != 0:
            query_update[i] = torch.sum(((softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i])) * query_reshape[idx].squeeze(1)), dim=0)
        else:
            query_update[i] = 0 
    updated_memory = F.normalize(query_update.cuda() + K.cuda(), dim=1)
    
    return updated_memory.detach()
