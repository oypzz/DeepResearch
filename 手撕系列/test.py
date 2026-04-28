import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self,max_len,d_model,vocab_size,drop_out=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=drop_out)

        pos = torch.arange(self.max_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0,self.d_model,2) * (-math.log(10000) / self.d_model))

        pe = torch.zeros(self.max_len,self.d_model)
        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        x = x + self.pe[:,:,:x.shape(1)]  # 最好是截取到序列长度
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,d_embd,n_heads,dim,max_len,drop_out=0.1,is_mask=False):
        super().__init__()
        assert dim % n_heads == 0

        self.d_embd = d_embd
        self.n_heads = n_heads
        self.n_dim = dim / n_heads
        self.dim = dim
        self.max_len = max_len
        self.is_mask = is_mask
        self.dropout = nn.Dropout(p=drop_out)

        mask = torch.full((1,1,self.max_len,self.max_len),float('-Inf'))
        mask = torch.triu(mask,diagonal=1)
        self.register_buffer('mask',mask)

        self.wq = nn.Linear(self.d_embd,self.dim)
        self.wk = nn.Linear(self.d_embd,self.dim)
        self.wv = nn.Linear(self.d_embd,self.dim)
        self.wo = nn.Linear(self.dim,self.d_embd)

    def forward(self,q,k,v):
        bz,seq_len,_ = q.shape
        # b,n,l,d
        xq = self.wq(q).view(bz,seq_len,self.n_heads,self.n_dim).transpose(1,2)
        xk = self.wq(k).view(bz,seq_len,self.n_heads,self.n_dim).transpose(1,2)
        xv = self.wq(v).view(bz,seq_len,self.n_heads,self.n_dim).transpose(1,2)

        scores = torch.matmul(xq,xk.transpose(2,3)) / math.sqrt(self.n_dim)

        if self.is_mask:
            scores = scores + self.mask[:,:,:seq_len,:seq_len]

        scores = F.softmax(scores,dim=-1)

        scores = torch.matmul(scores,xv) # b,n,l,d
        scores = scores.transpose(1,2).contiguous().view(bz,seq_len,-1)
        output = self.wo(scores)

        return self.dropout(output)


