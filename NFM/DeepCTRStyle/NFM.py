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
    dense_input_dict, sparse_input_dict = {}, {}
    
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1, ), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension, ), name=fc.name, dtype=fc.dtype)
    
    return dense_input_dict, sparse_input_dict

# 构建embedding层
def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    """
    定义一个embedding层对应的字典
    :param features_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是数据的特征封装版
    :input_layers_dict: A dict. 这是离散特征对应的层字典 {'sparse_name': Input(shap, name, dtype)}形式， 这个东西在NFM这没用到，统一形式而已
    :is_linear: A bool value. 表示是否是线性部分计算的特征，如果是，则embedding维度为1,因为线性计算就是wixi, 这里的xi是数
                              如果不是，说明是DNN部分的计算，此时embedding维度是自己定的，这部分就是高维稀疏转低维稠密那块
    """
    embedding_layers_dict = dict()
    
    # 将特征中的sparse特征筛选出来，这个离散特征要进行embedding
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    
    # 如果是用于线性部分的embedding层， 其维度是1, 否则，维度是自己定义的embedding维度
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_'+fc.name)
    else:    #这一块就是高维稀疏转低维稠密的embedding操作了
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)
    
    return embedding_layers_dict


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

# 构造特征交叉池化层的运算
class BiInteractionPooling(Layer):
    """构建特征交叉池化层，NFM中的创新点，DNN那边要用到这东西"""
    def __init__(self):
        super(BiInteractionPooling, self).__init__()
    
    # 前向传播逻辑
    def call(self, inputs):
        # 优化后的公式为： 0.5 * （和的平方-平方的和）  =>> (None, embed_dim)
        concated_embeds_value = inputs      # (None, field_num, embed_dim)
        
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=False)) # (None,embed_dim)
        sum_of_square = tf.reduce_sum(concated_embeds_value*concated_embeds_value, axis=1, keepdims=False) # (None, emebd_dim)
        cross_term = 0.5 * (square_of_sum - sum_of_square)  # (None, embed_dim)
        
        return cross_term
    
    def compute_output_shape(self, input_shape):
        return (None, input_shape[2])

# 把特征交叉池化层嵌入到网络中, 即特征交叉池化层的前向传播结果输出逻辑
def get_bi_interaction_pooling_output(sparse_input_dict, dnn_feature_columns, dnn_embedding_layers):
    """
    这里是计算特征池化层的结果部分，接收的输入是DNN部分的离散特征的embedding输入，然后过特征交叉池化层，得到输出
    :param sparse_input_dict:A dict. 离散特征构建的输入层字典 形式{'sparse_name': Input(shape, name, dtype)}
    :param dnn_feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是dnn数据的特征封装版
    :param dnn_embedding_layers: A dict. 离散特征构建的embedding层字典，形式{'sparse_name': Embedding(vocabulary_size, embedding_dim, name)}
    """
    # 只考虑sparse的二阶交叉，将所有的embedding拼接到一起
    # 这里在实际运行的时候，其实只会将那些非零元素对应的embedding拼接到一起
    # 并且将非零元素对应的embedding拼接到一起本质上相当于已经乘了x, 因为x中的值是1(公式中的x)
    sparse_kd_embed = []
    for fc in dnn_feature_columns:
        # 如果是离散特征，进行embedding
        if isinstance(fc, SparseFeat):
            feat_input = sparse_input_dict[fc.name]
            _embed = dnn_embedding_layers[fc.name](feat_input)   # (None, 1, embed_dim)
            sparse_kd_embed.append(_embed)
    
    # 将所有的sparse的embedding拼接起来， 得到(field_num, embed_dim)的矩阵, 然后特征交叉池化层
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    
    pooling_out = BiInteractionPooling()(concat_sparse_kd_embed)
    
    return pooling_out

# DNN 的输出
def get_dnn_logits(pooling_out):
    # dnn层，这里的Dropout参数，Dense中的参数都可以自己设定, 论文中还说使用了BN, 但是个人觉得BN和dropout同时使用
    # 可能会出现一些问题，感兴趣的可以尝试一些，这里就先不加上了
    dnn_out = Dropout(0.5)(Dense(1024, activation='relu')(pooling_out))  
    dnn_out = Dropout(0.3)(Dense(512, activation='relu')(dnn_out))
    dnn_out = Dropout(0.1)(Dense(256, activation='relu')(dnn_out))

    dnn_logits = Dense(1)(dnn_out)

    return dnn_logits

# 上面的模块合并得到就可以得到NFM网络
def NFM(linear_feature_columns, dnn_feature_columns):
    """
    搭建NFM模型，上面已经把所有组块都写好了，这里拼起来就好
    :param linear_feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是linear数据的特征封装版
    :param dnn_feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是DNN数据的特征封装版
    """
    # 构建输入层，即所有特征对应的Input()层， 这里使用字典的形式返回， 方便后续构建模型
    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的对应，是通过模型输入时候的字典数据的key与对应name的Input层
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns+dnn_feature_columns)
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    
    # 线性部分的计算 w1x1 + w2x2 + ..wnxn + b部分，dense特征和sparse两部分的计算结果组成，具体看上面细节
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns)
    
    # DNN部分的计算
    # 首先，在这里构建DNN部分的embedding层，之所以写在这里，是为了灵活的迁移到其他网络上，这里用字典的形式返回
    # embedding层用于构建FM交叉部分以及DNN的输入部分
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)
    
    # 过特征交叉池化层
    pooling_output = get_bi_interaction_pooling_output(sparse_input_dict, dnn_feature_columns, embedding_layers)
    
    # 加个BatchNormalization
    pooling_output = BatchNormalization()(pooling_output)
    
    # dnn部分的计算
    dnn_logits = get_dnn_logits(pooling_output)
    
    # 线性部分和dnn部分的结果相加，最后再过个sigmoid
    output_logits = Add()([linear_logits, dnn_logits])
    output_layers = Activation("sigmoid")(output_logits)
    
    model = Model(inputs=input_layers, outputs=output_layers)
    
    return model