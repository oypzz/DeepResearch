import torch
import torch.nn as nn
import math

class Token_Embedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        super(Token_Embedding, self).__init__(vocab_size,d_model)
        # self.token_embeddings = nn.Embedding(vocab_size,d_model)

# super.__init__只需要传入分类需要的参数，父类不需要则不用传
class Positional_Embedding(nn.Module):
    def __init__(self,max_len,d_model,drop_out = 0.1):
        super().__init__()
        self.drop_out = nn.Dropout(p = drop_out)
        position = torch.arange(max_len).unsqueeze(1) #  输入的是序列，要确定的是词元在序列中的位置，增加一个维度变成tensor
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000) / d_model))
        pe = torch.zeros(max_len,d_model)
        pe[:,0::2] = torch.sin(position * div_term) # 偶数用sin 奇数用cos
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe',pe.unsqueeze(0)) #  构造出与x一样的形状

    def forward(self,x):
        x = x + self.pe[:,:x.size(1)] #
        return self.drop_out(x)


if __name__ == '__main__':
    x = torch.rand(4,4,32)
    PE = Positional_Embedding(max_len=5000,d_model=32)
    print(PE(x))




