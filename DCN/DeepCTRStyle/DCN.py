# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K

from utils import DenseFeat, SparseFeat, VarLenSparseFeat

# 构建输入层
# 将输入的数据转换成字典的形式，定义输入层的时候让输入层的name和字典中特征的key一致，就可以使得输入的数据和对应的Input层对应
def build_input_layers(feature_columns):
    """构建Input层字典，并以dense和sparse两类字典的形式返回"""
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1, ), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension, ), name=fc.name, dtype=fc.dtype)
    return dense_input_dict, sparse_input_dict

# 构建embedding层
def build_embedding_layers(feature_columns, input_layer_dict, is_linear):
    # 定义一个embedding层对应的字典
    embedding_layers_dict = dict()
    
    # 将特征中的sparse特征筛选出来
    sparse_features_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    
    # 如果是用于线性部分的embedding层，其维度是1，否则维度是自己定义的embedding维度
    if is_linear:
        for fc in sparse_features_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_'+fc.name)
    else:
        for fc in sparse_features_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_'+fc.name)
    
    return embedding_layers_dict


# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    # 将sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for fc in sparse_feature_columns:
        _input = input_layer_dict[fc.name] # 获取输入层 
        _embed = embedding_layer_dict[fc.name] # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input) # B x dim  将input层输入到embedding层中

        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要
        if flatten:
            embed = Flatten()(embed)
        
        embedding_list.append(embed)
    
    return embedding_list 

def get_dnn_output(dnn_input, hidden_units=[1024, 512, 256], dnn_dropout=0.3, activation='relu'):
    
    # 建立dnn_network
    dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
    dropout = Dropout(dnn_dropout)
    
    # 前向传播
    x = dnn_input
    for dnn in dnn_network:
        x = dropout(dnn(x))
    
    return x

class CrossNet(Layer):
    def __init__(self, layer_nums=3):
        super(CrossNet, self).__init__()
        self.layer_nums = layer_nums
    
    def build(self, input_shape):
        # 计算w的维度，w的维度与输入数据的最后一个维度相同
        self.dim = int(input_shape[-1])
        
        # 注意，在DCN中w不是一个矩阵，而是一个向量，这里根据残差的层数定义一个权重列表
        self.W = [self.add_weight(name='W_' + str(i), shape=(self.dim,)) for i in range(self.layer_nums)]
        self.b = [self.add_weight(name='b_' + str(i), shape=(self.dim,), initializer='zeros') for i in range(self.layer_nums)]
    
    def call(self, inputs):
        
        # 进行特征交叉时的x_0一直没有变，变得是x_1和每一层的权重
        x_0 = inputs   # B*dims
        x_l = x_0
        for i in range(self.layer_nums):
            # 将x_1的第一个维度与w[i]的第0个维度计算点积
            xl_w = tf.tensordot(x_l, self.W[i], axes=(1, 0))  # B,
            xl_w = tf.expand_dims(xl_w, axis=-1)  # 最后一个维度上添加一个维度  B*1
            cross = tf.multiply(x_0, xl_w)   # B*dims  这里会用到python的广播机制
            x_l = cross + self.b[i] + x_l
        
        return x_l

def DCN(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns+dnn_feature_columns)
    
    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入预Input层对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    
    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # 线性层和dnn层统一的embedding层
    embedding_layer_dict = build_embedding_layers(linear_feature_columns+dnn_feature_columns, sparse_input_dict, is_linear=False)
    
    # Cross侧的计算逻辑 -- wide
    # 将linear_feature_columns里面的连续特征筛选出来，并把相应的Input层拼接到一块
    linear_dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), linear_feature_columns)) if linear_feature_columns else []
    linear_dense_feature_columns = [fc.name for fc in linear_dense_feature_columns]
    linear_concat_dense_inputs = Concatenate(axis=1)([dense_input_dict[col] for col in linear_dense_feature_columns])
    
    # 将linear_feature_columns里面的离散特征筛选出来，相应的embedding层拼接到一块
    linear_sparse_kd_embed = concat_embedding_list(linear_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True)
    linear_concat_sparse_kd_embed = Concatenate(axis=1)(linear_sparse_kd_embed)
    
    # Cross层的输入和输出
    linear_input = Concatenate(axis=1)([linear_concat_dense_inputs, linear_concat_sparse_kd_embed])
    cross_output = CrossNet()(linear_input)
    
    
    # DNN侧的计算逻辑 -- Deep
    # 将dnn_feature_columns里面的连续特征筛选出来，并把相应的Input层拼接到一块
    dnn_dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dnn_dense_feature_columns = [fc.name for fc in dnn_dense_feature_columns]
    dnn_concat_dense_inputs = Concatenate(axis=1)([dense_input_dict[col] for col in dnn_dense_feature_columns])
    
    # 将dnn_feature_columns里面的离散特征筛选出来，相应的embedding层拼接到一块
    dnn_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True)
    dnn_concat_sparse_kd_embed = Concatenate(axis=1)(dnn_sparse_kd_embed)
    
    # DNN层的输入和输出
    dnn_input = Concatenate(axis=1)([dnn_concat_dense_inputs, dnn_concat_sparse_kd_embed])
    dnn_output = get_dnn_output(dnn_input)
    
    # 两边的结果stack
    stack_output = Concatenate(axis=1)([cross_output, dnn_output])
    
    # 输出层
    output_layer = Dense(1, activation='sigmoid')(stack_output)
    
    model = Model(input_layers, output_layer)
    
    return model