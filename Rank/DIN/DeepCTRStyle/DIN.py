# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow.keras.backend as K

from utils import DenseFeat, SparseFeat, VarLenSparseFeat

# 构建输入层
# 将输入的数据转换成字典的形式，定义输入层的时候让输入层的name和字典中特征的key一致，就可以使得输入的数据和对应的Input层对应
def build_input_layers(feature_columns):
    """
    构建Input层字典，并以dense和sparse两类字典的形式返回
    :param feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是数据的特征封装版
    """
    input_layer_dict = {}
    
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_layer_dict[fc.name] = Input(shape=(1,), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_layer_dict[fc.name] = Input(shape=(fc.dimension, ), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_layer_dict[fc.name] = Input(shape=(fc.maxlen, ), name=fc.name, dtype=fc.dtype)

            if fc.length_name:
                input_layer_dict[fc.length_name] = Input((1,), name=fc.length_name, dtype='int32')
    
    return input_layer_dict

# 输入层拼接成列表
def concat_input_list(input_list):
    feature_nums = len(input_list)
    if feature_nums > 1:
        return Concatenate(axis=1)(input_list)
    elif feature_nums == 1:
        return input_list[0]
    else: 
        return None

# 构建embedding层
def build_embedding_layers(feature_columns, input_layers_dict):
    """
    定义一个embedding层对应的字典
    :param features_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是数据的特征封装版
    :input_layers_dict: A dict. 这是离散特征对应的层字典 {'sparse_name': Input(shap, name, dtype)}形式， 这个东西在NFM这没用到，统一形式而已
    """
    embedding_layers_dict = dict()
    
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='emb_' + fc.name)
        elif isinstance(fc, VarLenSparseFeat):  # 这里加1是因为这个需要填充， 预留一个填充字符的embedding
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim, name='emb_' + fc.name, mask_zero=True)


    return embedding_layers_dict

# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    """
    离散特征经过embedding之后得到各自的embedding向量，这里存储在一个列表中
    :feature_columns:A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是数据的特征封装版
    :input_layer_dict:A dict. 这是离散特征对应的层字典 {'sparse_name': Input(shap, name, dtype)}形式
    :embedding_layer_dict: A dict. 离散特征构建的embedding层字典，形式{'sparse_name': Embedding(vocabulary_size, embedding_dim, name)}
    """
    embedding_list = []
    for fc in feature_columns:
        _input = input_layer_dict[fc.name]  # 获取输入层 
        _embed = embedding_layer_dict[fc.name]  # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input)    # B x dim  将input层输入到embedding层中
        
        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要
        if flatten:
            embed = Flatten()(embed)
            
        embedding_list.append(embed)
    
    return embedding_list

def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    embedding_list = []
    
    for fc in feature_columns:
        _input = input_layer_dict[fc]
        _embed = embedding_layer_dict[fc]
        embed = _embed(_input)
        embedding_list.append(embed)

    return embedding_list

# Dice 层
class Dice(Layer):
    """Dice激活函数"""
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
    
    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],), dtype=tf.float32, name='alpha')
    
    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        
        return self.alpha * (1.0-x_p) * x + x_p * x


# Attention Layer
class LocalActivationUnit(Layer):
    def __init__(self, hidden_units=(256, 128, 64), activation='prelu'):
        """
        这里是Attention层的逻辑， 也是本篇论文的很新创新点, 这里只负责根据相关性计算历史行为与候选商品的得分
        :param hidden_units: 全连接层的隐藏层单元个数
        :param activation: 激活单元的类型
        """
        super(LocalActivationUnit, self).__init__()
        self.hidden_units = hidden_units
        self.linear = Dense(1)
        self.dnn = [Dense(unit, activation=PReLU() if activation=='prelu' else Dice()) for unit in hidden_units]
    
    #前向转播逻辑
    def call(self, inputs):
        # query: (None, 1, embed_dim)  keys: (None, max_len, embed_dim)
        query, keys = inputs

        # 获取序列长度
        keys_len, keys_dim = keys.get_shape()[1], keys.get_shape()[2]
        # 每个历史行为都要和当前的商品计算相似关系, 所以当前候选商品的embedding要复制keys_len份
        queries = tf.tile(query, multiples=[1, keys_len, 1])  # (None, max_len * embed_dim)
        queries = tf.reshape(queries, shape=[-1, keys_len, keys_dim]) # (None, max_len, embed_dim)
        
        # 将四种运算的特征拼接起来
        att_input = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1) # (None, max_len, 4*embed_dim)
        
        #将原始向量与外积结果拼接后输入到一个DNN中
        att_out = att_input
        for fc in self.dnn:
            att_out = fc(att_out)   # (None, max_len, att_out)
        
        att_out = self.linear(att_out)  # (None, max_len, 1)
        att_out = tf.squeeze(att_out, -1)  # (None, max_len)
        
        return att_out


# AttentionPoolingLayer层
class AttentionPoolingLayer(Layer):
    """该层基于上面的Attention层的得分，得到Attention的最终输出逻辑"""
    def __init__(self, att_hidden_units=(256, 128, 64)):
        super(AttentionPoolingLayer, self).__init__()
        self.att_hidden_units = att_hidden_units
        self.local_att = LocalActivationUnit(self.att_hidden_units)
    
    # 前向传播逻辑
    def call(self, inputs):
        # query: (None, 1, embed_dim)  keys: (None, max_len, embed_dim)
        query, keys, user_behavior_length= inputs
        # 获取行为序列的embedding的mask矩阵，keras中可通过_keras_mask进行获取
        # 这个里面每一行是[False, False, True, True, ....]的形式， False的长度表示样本填充的那部分
        #key_masks = keys._keras_mask   # (None, max_len)    这个代码目前我这里运行出问题 我先换种方式写
        key_masks = tf.sequence_mask(user_behavior_length, keys.shape[1])  # (None, 1, max_len)  这里注意user_behavior_length是(None,1)
        key_masks = key_masks[:, 0, :]     # 所以上面会多出个1维度来， 这里去掉才行，(None, max_len)
        
        # 获取行为序列中每个商品对应的注意力权重
        attention_score = self.local_att([query, keys])  # (None, max_len)
        
        # 创建一个padding的tensor, 目的是为了标记出行为序列embedding中无效的位置
        paddings = tf.zeros_like(attention_score) # (None, max_len)
        
        # outputs 表示的是padding之后的attention_score
        outputs = tf.where(key_masks, attention_score, paddings) # B x len
        
        # 将注意力分数与序列对应位置加权求和
        outputs = tf.expand_dims(outputs, axis=1)   # (None, 1, max_len)
        outputs = tf.matmul(outputs, keys)   # 三维矩阵相乘， 相乘发生在后两维   (None, 1, max_len) * (None, max_len, embed_dim) = (None, 1, embed_dim)
        outputs = tf.squeeze(outputs, axis=1)  # (None, embedding_dim)
        
        return outputs

# DNN 的输出
def get_dnn_logits(dnn_input, hidden_units=(200, 80), activation='prelu'):
    dnns = [Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]

    dnn_out = dnn_input
    for dnn in dnns:
        dnn_out = dnn(dnn_out)
    
    # 获取logits
    dnn_logits = Dense(1, activation='sigmoid')(dnn_out)

    return dnn_logits


# DIN网络搭建
def DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list):
    """
    这里搭建DIN网络，有了上面的各个模块，这里直接拼起来
    :param feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是数据的特征封装版
    :param behavior_feature_list: A list. 用户的候选行为列表
    :param behavior_seq_feature_list: A list. 用户的历史行为列表
    """
    # 构建Input层并将Input层转成列表作为模型的输入
    input_layer_dict = build_input_layers(feature_columns)
    input_layers = list(input_layer_dict.values())
    user_behavior_length = input_layer_dict["seq_length"]
    
    # 筛选出特征中的sparse和Dense特征， 后面要单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))
    
    # 获取Dense Input
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])
    
    # 将所有的dense特征拼接
    dnn_dense_input = concat_input_list(dnn_dense_input)   # (None, dense_fea_nums)
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)

    # 离散的这些特特征embedding之后，然后拼接，然后直接作为全连接层Dense的输入，所以需要进行Flatten
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=True)
    
    # 将所有的sparse特征embedding特征拼接
    dnn_sparse_input = concat_input_list(dnn_sparse_embed_input)   # (None, sparse_fea_nums*embed_dim)
    
    # 获取当前行为特征的embedding， 这里有可能有多个行为产生了行为列表，所以需要列表将其放在一起
    query_embed_list = embedding_lookup(behavior_feature_list, input_layer_dict, embedding_layer_dict)
    
    # 获取历史行为的embedding， 这里有可能有多个行为产生了行为列表，所以需要列表将其放在一起
    keys_embed_list = embedding_lookup(behavior_seq_feature_list, input_layer_dict, embedding_layer_dict)
    # 使用注意力机制将历史行为的序列池化，得到用户的兴趣
    dnn_seq_input_list = []
    for i in range(len(keys_embed_list)):
        seq_embed = AttentionPoolingLayer()([query_embed_list[i], keys_embed_list[i], user_behavior_length])  # (None, embed_dim)
        dnn_seq_input_list.append(seq_embed)
    
    # 将多个行为序列的embedding进行拼接
    dnn_seq_input = concat_input_list(dnn_seq_input_list)  # (None, hist_len*embed_dim)
    
    # 将dense特征，sparse特征， 即通过注意力机制加权的序列特征拼接起来
    dnn_input = Concatenate(axis=1)([dnn_dense_input, dnn_sparse_input, dnn_seq_input]) # (None, dense_fea_num+sparse_fea_nums*embed_dim+hist_len*embed_dim)
    
    # 获取最终的DNN的预测值
    dnn_logits = get_dnn_logits(dnn_input, activation='prelu')
    
    model = Model(inputs=input_layers, outputs=dnn_logits)
    
    return model