import math
import torch.nn.functional as F
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,dim,n_heads,n_embd,max_len,drop_out=0.1,is_mask=False):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.head_dim = dim / n_heads
        self.n_heads = n_heads
        self.is_mask = is_mask
        self.attn_dropout = nn.Dropout(p=drop_out)
        self.resid_dropout = nn.Dropout(p=drop_out)

        self.wq = nn.Linear(n_embd,self.n_heads * self.head_dim,bias=False)
        self.wk = nn.Linear(n_embd,self.n_heads * self.head_dim,bias=False)
        self.wv = nn.Linear(n_embd,self.n_heads * self.head_dim,bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim,self.dim,bias=False)


        if is_mask:
            mask = torch.full((1,1,max_len,max_len),float("-inf"))
            mask = torch.triu(mask,diagonal=1)
            self.register_buffer('mask',mask)

    def forward(self,q,k,v):
        bz,seq_len,_ = q.shape

        xq,xk,xv = self.wq(q),self.wk(k),self.wv(v) # b,seq_len,n_embd -> b,seq_len,dim
        xq = xq.view(bz,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        xk = xk.view(bz,seq_len,self.n_heads,self.head_dim).transpose(1,2)
        xv = xv.view(bz,seq_len,self.n_heads,self.head_dim).transpose(1,2)

        scores = torch.matmul(xq,xk.transpose(2,3)) / math.sqrt(self.head_dim)  # 因为是多头 这里用各个头的维度
        if self.is_mask:
            assert hasattr(self,'mask')
            scores = scores + self.mask[:,:,:seq_len,:seq_len] # 截取到序列长度

        scores = self.attn_dropout(F.softmax(scores.float(),dim=-1))
        out_put = torch.matmul(scores,xv)

        # 恢复维度 transpose后直接view会报错
        out_put = out_put.transpose(1,2).contiguous().view(bz,seq_len,-1) # b,n,l,n_head -> b,l,n*n_head
        out_put = self.resid_dropout(self.wo(out_put))
        return  out_put






