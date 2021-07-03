# -*- coding: utf-8 -*-
import itertools

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import (Zeros, glorot_normal, glorot_uniform)

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

# 得到线性部分的计算结果, 即线性部分计算的前向传播逻辑
def get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns):
    """
    线性部分的计算，所有特征的Input层，然后经过一个全连接层线性计算结果logits
    即FM线性部分的那块计算w1x1+w2x2+...wnxn + b,只不过，连续特征和离散特征这里的线性计算还不太一样
        连续特征由于是数值，可以直接过全连接，得到线性这边的输出。 
        离散特征需要先embedding得到1维embedding，然后直接把这个1维的embedding相加就得到离散这边的线性输出。
    :param dense_input_dict: A dict. 连续特征构建的输入层字典 形式{'dense_name': Input(shape, name, dtype)}
    :param sparse_input_dict: A dict. 离散特征构建的输入层字典 形式{'sparse_name': Input(shape, name, dtype)}
    :param linear_feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是linear数据的特征封装版
    """
    # 把所有的dense特征合并起来,经过一个神经元的全连接，做的计算  w1x1 + w2x2 + w3x3....wnxn
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = Dense(1)(concat_dense_inputs) 
    
    # 获取linear部分sparse特征的embedding层，这里使用embedding的原因：
    # 对于linear部分直接将特征进行OneHot然后通过一个全连接层，当维度特别大的时候，计算比较慢
    # 使用embedding层的好处就是可以通过查表的方式获取到非零元素对应的权重，然后将这些权重相加，提升效率
    linear_embedding_layers = build_embedding_layers(linear_feature_columns, sparse_input_dict, is_linear=True)
    
    # 将一维的embedding拼接，注意这里需要一个Flatten层， 使维度对应
    sparse_1d_embed = []
    for fc in linear_feature_columns:
        # 离散特征要进行embedding
        if isinstance(fc, SparseFeat):
            # 找到对应Input层，然后后面接上embedding层
            feat_input = sparse_input_dict[fc.name]
            embed = Flatten()(linear_embedding_layers[fc.name](feat_input))
            sparse_1d_embed.append(embed)
    
    # embedding中查询得到的权重就是对应onehot向量中一个位置的权重，所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    # 只不过是这里的输入特征只有0和1，所以直接向非零元素对应的权重相加就等同于进行了全连接操作(非零元素部分乘的是1)
    sparse_logits_output = Add()(sparse_1d_embed)
    
    # 最终将dense特征和sparse特征对应的logits相加，得到最终linear的logits
    linear_part = Add()([dense_logits_output, sparse_logits_output])
    
    return linear_part

class SENETLayer(Layer):
    def __init__(self, reduction_ratio, seed=2021):
        super(SENETLayer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.seed = seed
    
    def build(self, input_shape):
        # input_shape  [None, field_nums, embedding_dim]
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[-1]
        
        # 中间层的神经单元个数 f/r
        reduction_size = max(1, self.field_size // self.reduction_ratio)
        
        # FC layer1和layer2的参数
        self.W_1 = self.add_weight(shape=(
            self.field_size, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.field_size), initializer=glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = tf.keras.layers.Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs [None, field_num, embed_dim]
        
        # Squeeze -> [None, field_num]
        Z = tf.reduce_mean(inputs, axis=-1)
        
        # Excitation
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))   # [None, reduction_size]
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))   # [None, field_num]
        
        # Re-Weight
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))  # [None, field_num, embedding_dim]

        return V

class BilinearInteraction(Layer):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2,embedding_size)``.
    """
    def __init__(self, bilinear_type="interaction", seed=2021, **kwargs):
        super(BilinearInteraction, self).__init__(**kwargs)
        self.bilinear_type = bilinear_type
        self.seed = seed
    
    def build(self, input_shape):
        # input_shape: [None, field_num, embed_num]
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[-1]
        
        if self.bilinear_type == "all":   # 所有embedding矩阵共用一个矩阵W
            self.W = self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each": # 每个field共用一个矩阵W
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(self.field_size-1)]
        elif self.bilinear_type == "interaction":  # 每个交互用一个矩阵W
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(self.field_size), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # inputs: [None, field_nums, embed_dims]
        # 这里把inputs从field_nums处split, 划分成field_nums个embed_dims长向量的列表
        inputs = tf.split(inputs, self.field_size, axis=1)  # [(None, embed_dims), (None, embed_dims), ..] 
        n = len(inputs)  # field_nums个
        
        if self.bilinear_type == "all":
            # inputs[i] (none, embed_dims)    self.W (embed_dims, embed_dims) -> (None, embed_dims)
            vidots = [tf.tensordot(inputs[i], self.W, axes=(-1, 0)) for i in range(n)]   # 点积
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]  # 哈达玛积
        elif self.bilinear_type == "each":
            vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n - 1)]
            # 假设3个域， 则两两组合[(0,1), (0,2), (1,2)]  这里的vidots是第一个维度， inputs是第二个维度 哈达玛积运算
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "interaction":
            # combinations(inputs, 2)  这个得到的是两两向量交互的结果列表
            # 比如 combinations([[1,2], [3,4], [5,6]], 2)
            # 得到 [([1, 2], [3, 4]), ([1, 2], [5, 6]), ([3, 4], [5, 6])]  (v[0], v[1]) 先v[0]与W点积，然后再和v[1]哈达玛积
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError
            
        output = Concatenate(axis=1)(p)
        return output

def fibinet(linear_feature_columns, dnn_feature_columns,  bilinear_type='interaction', reduction_ratio=3, hidden_units=[128, 128]):
    """
    :param linear_feature_columns, dnn_feature_columns: 封装好的wide端和deep端的特征
    :param bilinear_type: 双线性交互类型， 有'all', 'each', 'interaction'三种
    :param reduction_ratio: senet里面reduction ratio
    :param hidden_units: DNN隐藏单元个数
    """
    
    # 构建输出层, 即所有特征对应的Input()层， 用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)
    
    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入预Input层对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    
    # 线性部分的计算逻辑 -- linear
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns)
    
    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # 线性层和dnn层统一的embedding层
    embedding_layer_dict = build_embedding_layers(linear_feature_columns+dnn_feature_columns, sparse_input_dict, is_linear=False)
    
    # DNN侧的计算逻辑 -- Deep
    # 将dnn_feature_columns里面的连续特征筛选出来，并把相应的Input层拼接到一块
    dnn_dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dnn_dense_feature_columns = [fc.name for fc in dnn_dense_feature_columns]
    dnn_concat_dense_inputs = Concatenate(axis=1)([dense_input_dict[col] for col in dnn_dense_feature_columns])
    
    # 将dnn_feature_columns里面的离散特征筛选出来，相应的embedding层拼接到一块,然后过SENet_layer
    dnn_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=False)
    sparse_embedding_list = Concatenate(axis=1)(dnn_sparse_kd_embed)
    
    # SENet layer
    senet_embedding_list = SENETLayer(reduction_ratio)(sparse_embedding_list)
    
    # 双线性交互层
    senet_bilinear_out = BilinearInteraction(bilinear_type=bilinear_type)(senet_embedding_list)
    senet_bilinear_out = Flatten()(senet_bilinear_out)
    raw_bilinear_out = BilinearInteraction(bilinear_type=bilinear_type)(sparse_embedding_list)
    raw_bilinear_out = Flatten()(raw_bilinear_out)
    
    bilinear_out = Concatenate(axis=1)([senet_bilinear_out, raw_bilinear_out])
    
    # DNN层的输入和输出
    dnn_input = Concatenate(axis=1)([bilinear_out, dnn_concat_dense_inputs])
    dnn_out = get_dnn_output(dnn_input, hidden_units=hidden_units)
    dnn_logits = Dense(1)(dnn_out)
                             
    # 最后的输出
    final_logits = Add()([linear_logits, dnn_logits])
    
    # 输出层
    output_layer = Dense(1, activation='sigmoid')(final_logits)
    
    model = Model(input_layers, output_layer)
    
    return model
