import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.regularizers import l2

from collections import OrderedDict
import itertools

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat


def build_input_layers(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features

def build_embedding_layers(feature_columns):
    embedding_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='emb_' + fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            # 这里加1是因为mask有个默认的embedding，占用了一个位置，比如mask为0， 而0这个位置的embedding是mask专用了
            embedding_layer_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim, name='emb_' + fc.name, mask_zero=True)

    return embedding_layer_dict

def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    # 这个函数是Input层与embedding层给接起来
    embedding_list = []
    
    for fc in feature_columns:
        _input = input_layer_dict[fc]
        _embed = embedding_layer_dict[fc]
        embed = _embed(_input)
        embedding_list.append(embed)

    return embedding_list

def concat_input_list(input_list):
    feature_nums = len(input_list)
    if feature_nums > 1:
        return Concatenate(axis=1)(input_list)
    elif feature_nums == 1:
        return input_list[0]
    else:
        return None

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



class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)
    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)
    # 去掉了mask的属性 比如历史行为序列中
    def call(self, x, mask=None, **kwargs):
        return x
    def compute_mask(self, inputs, mask):
        return None


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)
    
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]
        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]
        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)
        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            fc = self.activation_layers[i](fc)
            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc
        return deep_input


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, str)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")
        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))
        return output

def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)

def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


def lhuc_net(name, nn_inputs, lhuc_inputs, nn_hidden_units=(128, 64, ), lhuc_units=(32, ), 
             dnn_activation='relu', l2_reg_dnn=0, dnn_dropout=0, dnn_use_bn=False, scale_last=True, seed=2021):
    """这个网络是全连接网络搭建的，主要完成lhuc_feature与其他特征的交互， 算是一个特征交互层，不过交互的方式非常新颖
    
        name: 为当前lhuc_net起的名字
        nn_inputs: 与lhuc_feature进行交互的特征输入，比如fm_out， 或者其他特征的embedding拼接等
        lhuc_inputs: lhuc_net的特征输入，在推荐里面，这个其实是能体现用户个性化的一些特征embedding等
        nn_hidden_units: 普通DNN每一层神经单元个数
        lhuc_units: lhuc_net的神经单元个数
        后面就是激活函数， 正则化以及bn的指定参数，不过多解释
    """
    
    # nn_inputs可以是其他特征的embedding拼接向量，或者是其他网络的输出，比如fM的输出向量等
    cur_layer = nn_inputs       
    
    # 这里的nn_hidden_units是一个列表，里面是全连接每一层神经单元个数
    for idx, nn_dim in enumerate(nn_hidden_units):
        # lhuc_feature走一个塔， 这个塔两层， 最终输出的向量维度和nn_inputs的向量维度保持一致， 每个值在0-1之间，代表权重
        # 表示fm_embedding或者其他特征embdding每个维度上的重要性  
        # 这里其实可以用多层 激活函数用relu 
        lhuc_output = DNN(lhuc_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, 
                          seed=seed, name="{}_lhuc_{}".format(name, idx))(lhuc_inputs)
        # 最后这里的输出维度要和交互的embedding保持一致， 激活函数是sigmoid，
        lhuc_scale = Dense(int(cur_layer.shape[1]), activation='sigmoid')(lhuc_output)
        
        # 有了权重之后， lhuc_scale与nn_inputs再过一个塔
        cur_layer = DNN((nn_dim, ), dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, 
                        seed=seed, name="{}_layer_{}".format(name, idx))(cur_layer * lhuc_scale * 2.0)
        
    # 上面这个操作相当于nn_input_embedding过了len(nn_hidden_units)层全连接， 只不过，在过每一层之前，会先lhuc_slot特征通过lhuc_net为
    # nn_input_embedding过完全连接之后的每个维度学习权重，作为每个维度的重要性
    # 如果最后的输出还需要加权，再走一遍上面的操作
    if scale_last:
        lhuc_output = DNN(lhuc_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, 
                          seed=seed, name="{}_lhuc_{}".format(name, len(nn_hidden_units)))(lhuc_inputs)
        lhuc_scale = Dense(int(cur_layer.shape[1]), activation='sigmoid')(lhuc_output)
        
        cur_layer = cur_layer * lhuc_scale * 2.0
    
    return cur_layer


class BilinearInteraction(Layer):
    def __init__(self, bilinear_type="interaction", seed=2022, **kwargs):
        super(BilinearInteraction, self).__init__(**kwargs)
        self.bilinear_type = bilinear_type
        self.seed = seed
    def build(self, input_shape):
        # input_shape: [None, field_num, embed_num]
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[-1]
        
        if self.bilinear_type == 'all':  #所有embedding矩阵共用一个矩阵W
            self.W = self.add_weight(shape=(self.embedding_size, self.embedding_size), 
                                     initializer=glorot_normal(seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each": # 每个field共用一个矩阵W
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(self.field_size-1)]
        elif self.bilinear_type == "interaction":  # 每个交互用一个矩阵W
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(self.field_size), 2)]
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
        
        output = Concatenate(axis=1)(p)
        return output


def SharedBottom(dnn_feature_columns, lhuc_feature_columns, bottom_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64, ), 
                l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=2021, dnn_dropout=0, dnn_activation='relu',
                dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), bilinear_type='interaction'):
    
    num_tasks = len(task_names)
    
    # 异常判断
    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))
    
    # 构建Input层并将Input层转成列表作为模型的输入
    input_layer_dict = build_input_layers(dnn_feature_columns)
    input_layers = list(input_layer_dict.values())
    
    # 筛选出特征中的sparse和Dense特征， 后面要单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))
    
    # 获取Dense Input
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(dnn_feature_columns)
    # 离散的这些特特征embedding之后，然后拼接，然后直接作为全连接层Dense的输入，所以需要进行Flatten
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=False)
    
    # 把连续特征和离散特征合并起来
    bias_input = combined_dnn_input(dnn_sparse_embed_input, dnn_dense_input)
    
    # 下面dnn_sparse_embed_input进行双线性交互
    bilinear_out = BilinearInteraction(bilinear_type=bilinear_type)(Concatenate(axis=1)(dnn_sparse_embed_input))
    
    # lhuc_features_columns
    lhuc_input = concat_embedding_list(lhuc_feature_columns, input_layer_dict, embedding_layer_dict, flatten=True)
    lhuc_input = concat_func(lhuc_input)
    
    # bilinear_out与lhuc_input过lhuc_net
    bilinear_out_flatt = Flatten()(bilinear_out)
    bilinear_lhuc_out = lhuc_net("bilinear_lhuc", bilinear_out_flatt, lhuc_input)
    
    # bias_input与lhuc_input过lhuc_net
    bias_lhuc_out = lhuc_net("bias_lhuc", bias_input, lhuc_input)
    
    # 两个输出拼接就是双线性net的最终输出结果，汇总了原始信息和交叉信息， 且通过lhuc_net对维度加权，在DNN每一层做一个维度筛选
    sb_out = Concatenate(axis=-1)([bilinear_lhuc_out, bias_lhuc_out])

    sb_out = DNN((64, ), dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='sharedlast')(sb_out)
    
    # 每个任务独立的tower
    task_outputs = []
    for task_type, task_name in zip(task_types, task_names):
        # 建立tower
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022, name='tower_'+task_name)(sb_out)
        logit = Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outputs.append(output)
    
    model = Model(inputs=input_layers, outputs=task_outputs)
    return model