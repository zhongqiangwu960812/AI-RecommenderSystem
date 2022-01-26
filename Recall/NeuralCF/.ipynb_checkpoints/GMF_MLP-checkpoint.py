# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GMF(nn.Module):
    
    def __init__(self, num_users, num_items, latent_dim, regs=[0, 0]):
        super(GMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=latent_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=latent_dim)
        self.linear = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作
        
        # 先把输入转成long类型
        inputs = inputs.long()
        
        # 用户和物品的embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第一列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        
        # 两个隐向量点积
        predict_vec = torch.mul(MF_Embedding_User, MF_Embedding_Item)
        
        # liner
        linear = self.linear(predict_vec)
        output = self.sigmoid(linear)
        
        return output

class MLP(nn.Module):
    
    def __init__(self, num_users, num_items, layers=[20, 64, 32, 16], regs=[0, 0]):
        super(MLP, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0]//2)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0]//2)
        
        # 全连接网络
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        self.linear = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs):
        # 这个inputs是一个批次的数据， 所以后面的操作切记写成inputs[0], [1]这种， 这是针对某个样本了， 我们都是对列进行的操作
        # 先把输入转成long类型
        inputs = inputs.long()
        
        # MF的前向传播  用户和物品的embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[:, 0])  # 这里踩了个坑， 千万不要写成[0]， 我们这里是第一列
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[:, 1])
        
        
        # 两个隐向量堆叠起来
        x = torch.cat([MF_Embedding_User, MF_Embedding_Item], dim=-1)
        
        # l全连接网络
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        
        x = self.linear(x)
        output = self.sigmoid(x)
        
        return output
