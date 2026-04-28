import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,dim,n_heads,n_embd,max_len,drop_out=0.1,is_mask=False):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.nheads = n_heads
        self.head_dim = dim // n_heads # 这样出来是整数
        self.is_mask = is_mask
        self.dropout = nn.Dropout(p=drop_out)

        self.wq = nn.Linear(n_embd,self.nheads * self.head_dim,bias = False)
        self.wk = nn.Linear(n_embd,self.nheads * self.head_dim,bias= False)
        self.wv = nn.Linear(n_embd,self.nheads * self.head_dim,bias= False)
        self.wo = nn.Linear(self.dim,self.nheads * self.head_dim,bias= False)

        if is_mask:
            mask = torch.full((1,1,max_len,max_len),float("-inf"))
            mask = torch.triu(mask,diagonal=1)
            self.register_buffer('mask',mask)

    def forward(self,q,k,v):
        bz,seq_len,_ = q.shape

        xq,xk,xv = self.wq(q),self.wk(k),self.wv(v)
        xq = xq.view(bz,seq_len,self.nheads,self.head_dim).transpose(1,2)
        xk = xk.view(bz,seq_len,self.nheads,self.head_dim).transpose(1,2)
        xv = xv.view(bz,seq_len,self.nheads,self.head_dim).transpose(1,2)

        scores = torch.matmul(xq,xk.transpose(2,3)) / math.sqrt(self.head_dim)
        if self.is_mask:
            assert  hasattr(self,'mask')
            scores += scores + self.mask[:,:,:seq_len,:seq_len]
        weight = self.dropout(F.softmax(scores.float(),dim=-1))
        output = torch.matmul(weight,xv)

        output = output.transpose(1,2).contiguous().view(bz,seq_len,-1)
        output = self.dropout(self.wo(output))
        return output


# 参数设置
batch_size = 2
seq_len = 5
dim = 16          # 模型总维度
n_heads = 4       # 头数（需整除 dim）
max_len = 10      # 最大序列长度（用于预定义因果掩码）
is_mask = True    # 开启因果掩码（自回归）

# 实例化模型
mha = MultiHeadAttention(dim=dim, n_embd=dim,n_heads=n_heads, max_len=max_len, is_mask=is_mask)


q = torch.randn(batch_size, seq_len, dim)
k = torch.randn(batch_size, seq_len, dim)
v = torch.randn(batch_size, seq_len, dim)
# 前向传播
output = mha(q, k, v)

print("输入形状:", q.shape)        # (2, 5, 16)
print("输出形状:", output.shape)   # (2, 5, 16)
print("输出数值示例:\n", output[0, 0, :5])  # 打印第一个样本第一个位置的几个维度值

