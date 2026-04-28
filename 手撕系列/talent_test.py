# kmeans取最近质心
import numpy as np
import math

n,d = map(int,input().split())
x = np.array([list(map(float,input().split())) for _ in range(n)])

# 读取质心矩阵
k = int(input())
c = np.array([list(map(float,input().split())) for _ in range(k)])

# 处理形状以便于广播计算  最后得到形状(n,k,d)
# 这样得到的 diff[i,j,:]是长度为d的一维数组，代表第i个点跟第j个质心的差值向量，并在长度d上计算L2范数
dist = np.linalg.norm(x[:,None,:] - c[None,:,:],axis=2)

labels = np.argmin(dist,axis=1)
print("".join(map(str,labels)))


# 行级别的softmax操作
s = np.array([list(map(float,input().split(()))) for _ in range(m)],dtype=np.float64)
s = s - s.max(axis=1,keepdims=True) # 保持维度很重要
exp_s = np.exp(s)
p = exp_s / exp_s.sum(axis=1,keepdims=True) # axis=1按行计算


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ex = math.exp(z)
        return 1.0 / (1.0 + ex)

def logistic_regression(x,y,alpha,lam,max_iter,tol):
     n,d = len(x),len(x[0])  # 变量个数和变量维度
     w = [0.0] * d
     b = 0.0
     eps = 1e-15

     def compute_grad():
         gw = [0.0] * d
         gb = 0.0
         loss = 0.0

         for i in range(n): # 批量计算预测值和真实值的损失
             z = b + sum(w[j] * x[i][j] for j in range(d))
             p = sigmoid(z)
             yi = y[i]
             loss += -((yi * math.log(max(p,eps))) - (1-yi) * math.log(max(1-p,eps)))
             diff = p-yi
             for j in range(d):gw[j] += diff * x[i][j] # w的梯度计算结果前半段
             gb += diff
        # 计算批量均值
         loss /= n
         l2 = lam/(2*n) * sum(w[j]**2 for j in range(d)) # l2正则惩罚
         loss += l2
         # 整理w的梯度
         for j in range(d):
             gw[j] = gw[j]/n + (lam/n)*w[j]
         return loss,gw,gb/n

     prev_loss, _, _ = compute_grad()
     for _ in range(max_iter):
        loss,gw,gb = compute_grad()
        for j in range(d):w[j] -= alpha * gw[j]
        b -= alpha * gb
        new_loss,_,_ = compute_grad()
        if abs(prev_loss - new_loss) < tol: break
        prev_loss = new_loss
     return w,b





