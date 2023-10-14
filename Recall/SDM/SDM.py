from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal, Zeros, glorot_normal, TruncatedNormal, Ones

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from tensorflow.python.keras.regularizers import l2
#from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss

if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()

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

def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)

def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

class AttentionSequencePoolingLayer(Layer):
    """
    :param query: [batch_szie, 1, c_q]
    :param keys: [batch_size, T, c_k]
    :param keys_length: [batch_size, 1]
    :return [batch, 1, c_k]
    """
    def __init__(self, dropout_rate=0, scale=True, **kwargs):
        self.dropout_rate = dropout_rate
        self.scale = scale
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.projection_layer = Dense(units=1, activation='tanh')       
        super(AttentionSequencePoolingLayer, self).build(input_shape)
    
    def call(self, inputs, mask=None, **kwargs):
        # queries[None, 1, 64], keys[None, 50, 32], keys_length[None, 1]， 表示真实的会话长度， 后面mask会用到
        queries, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]
        key_mask = tf.sequence_mask(keys_length, hist_len)  # mask 矩阵 (None, 1, 50) 
        
        queries = tf.tile(queries, [1, hist_len, 1])  # [None, 50, 64]   为每个key都配备一个query
        # 后面，把queries与keys拼起来过全连接， 这里是这样的， 本身接下来是要求queryies与keys中每个item相关性，常规操作我们能想到的就是直接内积求得分数
        # 而这里的Attention实现，使用的是LuongAttention， 传统的Attention原来是有两种BahdanauAttention与LuongAttention， 这个在博客上整理下
        # 这里采用的LuongAttention，对其方式是q_k先拼接，然后过DNN的方式
        q_k = tf.concat([queries, keys], axis=-1)  # [None, 50, 96]
        output_scores = self.projection_layer(q_k)  # [None, 50, 1]

        if self.scale:
            output_scores = output_scores / (q_k.get_shape().as_list()[-1] ** 0.5)
        attention_score = tf.transpose(output_scores, [0, 2, 1])
        
        # 加权求和 需要把填充的那部分mask掉
        paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
        attention_score = tf.where(key_mask, attention_score, paddings)
        attention_score = softmax(attention_score)  # [None, 1, 50]
        
        outputs = tf.matmul(attention_score, keys)  # [None, 1, 64]
        return outputs
        
    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][1])

class DynamicMultiRNN(Layer):
    def __init__(self, num_units=None, rnn_type='LSTM', return_sequence=True, num_layers=2, num_residual_layers=1,
                dropout_rate=0.2, forget_bias=1.0, **kwargs):
        self.num_units = num_units
        self.return_sequence = return_sequence
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_residual_layers = num_residual_layers
        self.dropout = dropout_rate
        self.forget_bias = forget_bias
        super(DynamicMultiRNN, self).__init__(**kwargs)
    def build(self, input_shape):
        input_seq_shape = input_shape[0]
        if self.num_units is None:
            self.num_units = input_seq_shape.as_list()[-1]
        # 这里先只用LSTM
        if self.rnn_type == "LSTM":
            try:
                # AttributeError: module 'tensorflow_core._api.v2.nn' has no attribute 'rnn_cell'
                single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
            except AttributeError:
                single_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.num_units, forget_bias=self.forget_bias)
                
        dropout = self.dropout if tf.keras.backend.learning_phase() == 1 else 0  # 训练的时候开启Dropout， 不训练的时候关闭
        try:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0-dropout))
        except AttributeError:
            single_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
        
        cell_list = []
        for i in range(self.num_layers):
            residual = (i >= self.num_layers - self.num_residual_layers)
            if residual:
                # 实现的功能， 把输入concat到输出上一起返回， 就跟ResNet一样
                try: 
                    single_cell_residual = tf.nn.rnn_cell.ResidualWrapper(single_cell)
                except AttributeError:
                    single_cell_residual = tf.compat.v1.nn.rnn_cell.ResidualWrapper(single_cell)
                cell_list.append(single_cell_residual)
            else:
                cell_list.append(single_cell)
        
        if len(cell_list) == 1:
            self.final_cell = cell_list[0]
        else:
            # 构建多隐层RNN
            try:
                self.final_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)
            except AttributeError:
                self.final_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list)
                
        super(DynamicMultiRNN, self).build(input_shape)
    def call(self, input_list):
        rnn_input, sequence_length = input_list
        
        try:
            # AttributeError: module 'tensorflow' has no attribute 'variable_scope'
            with tf.name_scope('rnn'), tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
                # rnn_output是所有时间步的隐藏层状态， hidden_state是最后一个时间步的输出， 如果是LSTM这里是c和h两个
                rnn_output, hidden_state = tf.nn.dynamic_rnn(self.final_cell, inputs=rnn_input, sequence_length=tf.squeeze(sequence_length),
                                                           dtype=tf.float32, scope=self.name)
        except AttributeError:
             with tf.name_scope("rnn"), tf.compat.v1.variable_scope("rnn", reuse=tf.compat.v1.AUTO_REUSE):
                rnn_output, hidden_state = tf.compat.v1.nn.dynamic_rnn(self.final_cell, inputs=rnn_input,
                                                                       sequence_length=tf.squeeze(sequence_length),
                                                                       dtype=tf.float32, scope=self.name)
        if self.return_sequence:
            return rnn_output
        else:
            return tf.expand_dims(hidden_state, axis=1)
    
    def compute_output_shape(self, input_shape):
        rnn_input_shape = input_shape[0]
        if self.return_sequence:
            return rnn_input_shape
        else:
            return (None, 1, rnn_input_shape[2])

class LayerNormalization(Layer):
    def __init__(self, axis=-1, eps=1e-9, center=True,
                 scale=True, **kwargs):
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scale = scale
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.eps)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

class MultiHeadAttention(Layer):
    def __init__(self, num_units=8, head_num=4, scale=True, dropout_rate=0.2, use_layer_norm=True, use_res=True, seed=2020, **kwargs):
        self.num_units = num_units
        self.head_num = head_num
        self.scale = scale
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_res = use_res
        self.seed = seed
        super(MultiHeadAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        # wq_wk_wv 放到一块
        self.W = self.add_weight(name='Q_K_V', shape=[embedding_size, self.num_units*3],
                                 dtype=tf.float32,
                                 initializer=TruncatedNormal(seed=self.seed))
        self.W_output = self.add_weight(name='output_W', shape=[self.num_units, self.num_units],
                                       dtype=tf.float32,
                                        initializer=TruncatedNormal(seed=self.seed))
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.seq_len_max = int(input_shape[0][1])
        super(MultiHeadAttention, self).build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):
        input_info, keys_length = inputs
        hist_len = input_info.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)  # (None, 1, 5)
        key_masks = tf.squeeze(key_masks, axis=1)  # (None, 5)
        
        Q_K_V = tf.tensordot(input_info, self.W, axes=(-1, 0))  # (None, seq_len, embed*3)
        querys, keys, values = tf.split(Q_K_V, 3, -1)
        
        # 计算的时候，分开头计算
        querys = tf.concat(tf.split(querys, self.head_num, axis=2), axis=0)  # (head_num*None, seq_len, embed/head_num)
        keys = tf.concat(tf.split(keys, self.head_num, axis=2), axis=0)   # (head_num*None, seq_len, embed/headnum)
        values = tf.concat(tf.split(values, self.head_num, axis=2), axis=0) # (head_num*None, seq_len, embed/head_num)
        
        # 注意力分数
        att_score = tf.matmul(querys, tf.transpose(keys, [0, 2, 1]))  # (head_num*None, seq_len, seq_len)
        if self.scale:
            att_score = att_score / (keys.get_shape().as_list()[-1] ** 0.5)
        key_masks = tf.tile(key_masks, [self.head_num, 1])  # [head_num*None, seq_len]
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input_info)[1], 1])  # [head_num*None, seq_len, seq_len]
        
        paddings = tf.ones_like(att_score) * (-2 ** 32 + 1)  # [head_num*None, seq_len, seq_len]
        align = tf.where(key_masks, att_score, paddings)  # [head_num*None, seq_len, seq_len]
        align = softmax(align)
        output = tf.matmul(align, values)  # [head_num*None, seq_len, emb/head_num]
        
        output = tf.concat(tf.split(output, self.head_num, axis=0), axis=2)  # [None, seq_len, emb]
        output = tf.tensordot(output, self.W_output, axes=(-1, 0))  # [None, seq_len, emb]
        output = self.dropout(output, training=training)
        if self.use_res:
            output += input_info
        if self.use_layer_norm:
            output = self.layer_norm(output)
        return output
        
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.num_units)

class UserAttention(Layer):
    def __init__(self, num_units=None, activation='tanh', use_res=True, dropout_rate=0, scale=True, seed=2020, **kwargs):
        self.scale = scale
        self.num_units = num_units
        self.activation = activation
        self.seed = seed
        self.dropout_rate = dropout_rate
        self.use_res = use_res
        super(UserAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if self.num_units is None:
            self.num_units = input_shape[0][-1]
        self.dense = Dense(self.num_units, activation=self.activation)
        super(UserAttention, self).build(input_shape)
    
    def call(self, inputs, mask=None, **kwargs):
        # [None, 1, embed]  [None, 5, embed], [None, 1]
        user_query, keys, keys_length = inputs
        hist_len = keys.get_shape()[1]
        key_masks = tf.sequence_mask(keys_length, hist_len)  # [None, 1, 5]
        query = self.dense(user_query)   # [None, 1, num_units]
        
        # 注意力分数
        att_score = tf.matmul(query, tf.transpose(keys, [0, 2, 1]))  # (None, 1, seq_len)
        if self.scale:
            att_score = att_score / (keys.get_shape().as_list()[-1] ** 0.5)
        
        paddings = tf.ones_like(att_score) * (-2 ** 32 + 1)  # [None, 1, seq_len]
        align = tf.where(key_masks, att_score, paddings)  # [None, 1, seq_len]
        align = softmax(align)
        
        output = tf.matmul(align, keys)  # [None, 1, embed]
        return output
    
    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[1][2])
    def compute_mask(self, inputs, mask):
        return mask

class EmbeddingIndex(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)
    def build(self, input_shape):
        super(EmbeddingIndex, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, **kwargs):
        return tf.constant(self.index)

class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.size = input_shape[0][0]   # docs num
        self.zero_bias = self.add_weight(shape=[self.size], initializer=Zeros, dtype=tf.float32, trainable=False, name='bias')
        super(SampledSoftmaxLayer, self).build(input_shape)
    def call(self, inputs_with_label_idx):
        embeddings, inputs, label_idx = inputs_with_label_idx
        # 这里盲猜下这个操作，应该是有了label_idx，就能拿到其embedding，这个是用户行为过多
        # 所以(user_embedding, embedding[label_idx])就是正样本
        # 然后根据num_sampled，再从传入的embeddings字典里面随机取num_sampled个负样本
        # 根据公式log(sigmoid(user_embedding, embedding[label_idx])) + 求和(log(sigmoid(-user_embedding, embedding[负样本])))得到损失
        loss = tf.nn.sampled_softmax_loss(weights=embeddings,   # (numclass, dim)
                                          biases=self.zero_bias,  # (numclass)
                                          labels=label_idx,   # (batch, num_true)
                                          inputs=inputs,  # (batch, dim)
                                          num_sampled=self.num_sampled, # 负采样个数
                                          num_classes=self.size  # 类别数量
                                         )
        return tf.expand_dims(loss, axis=1)

def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)

class PoolingLayer(Layer):
    def __init__(self, mode='mean', supports_masking=False, **kwargs):
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(PoolingLayer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        super(PoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if not isinstance(seq_value_len_list, list):
            seq_value_len_list = [seq_value_len_list]
        if len(seq_value_len_list) == 1:
            return seq_value_len_list[0]
        expand_seq_value_len_list = list(map(lambda x: tf.expand_dims(x, axis=-1), seq_value_len_list))
        a = concat_func(expand_seq_value_len_list)
        if self.mode == "mean":
            hist = reduce_mean(a, axis=-1, )
        if self.mode == "sum":
            hist = reduce_sum(a, axis=-1, )
        if self.mode == "max":
            hist = reduce_max(a, axis=-1, )
        return hist

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(PoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Sdm(user_feature_columns, item_feature_columns, history_feature_list, num_sampled=5, units=32, rnn_layers=2,
        dropout_rate=0.2, rnn_num_res=1, num_head=4, l2_reg_embedding=1e-6, dnn_activation='tanh', seed=1024):
    """
    :param rnn_num_res: rnn的残差层个数 
    :param history_feature_list: short和long sequence field
    """
    # item_feature目前只支持doc_id， 再加别的就不行了，其实这里可以改造下
    if (len(item_feature_columns)) > 1: 
        raise ValueError("SDM only support 1 item feature like doc_id")
    
    # 获取item_feature的一些属性
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_column.vocabulary_size
    
    # 为用户特征创建Input层
    user_input_layer_dict = build_input_layers(user_feature_columns)
    item_input_layer_dict = build_input_layers(item_feature_columns)
    
    # 将Input层转化成列表的形式作为model的输入
    user_input_layers = list(user_input_layer_dict.values())
    item_input_layers = list(item_input_layer_dict.values())
    
    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    if len(dense_feature_columns) != 0:
        raise ValueError("SDM dont support dense feature")  # 目前不支持Dense feature
    varlen_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(user_feature_columns+item_feature_columns)
    
    # 拿到短期会话和长期会话列 之前的命名规则在这里起作用
    sparse_varlen_feature_columns = []
    prefer_history_columns = []
    short_history_columns = []
    
    prefer_fc_names = list(map(lambda x: "prefer_" + x, history_feature_list))
    short_fc_names = list(map(lambda x: "short_" + x, history_feature_list))
    
    for fc in varlen_feature_columns:
        if fc.name in prefer_fc_names:
            prefer_history_columns.append(fc)
        elif fc.name in short_fc_names:
            short_history_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)
    
    # 获取用户的长期行为序列列表 L^u 
    # [<tf.Tensor 'emb_prefer_doc_id_2/Identity:0' shape=(None, 50, 32) dtype=float32>, <tf.Tensor 'emb_prefer_cat1_2/Identity:0' shape=(None, 50, 32) dtype=float32>, <tf.Tensor 'emb_prefer_cat2_2/Identity:0' shape=(None, 50, 32) dtype=float32>]
    prefer_emb_list = embedding_lookup(prefer_fc_names, user_input_layer_dict, embedding_layer_dict)
    # 获取用户的短期序列列表 S^u
    # [<tf.Tensor 'emb_short_doc_id_2/Identity:0' shape=(None, 5, 32) dtype=float32>, <tf.Tensor 'emb_short_cat1_2/Identity:0' shape=(None, 5, 32) dtype=float32>, <tf.Tensor 'emb_short_cat2_2/Identity:0' shape=(None, 5, 32) dtype=float32>]
    short_emb_list = embedding_lookup(short_fc_names, user_input_layer_dict, embedding_layer_dict)
    
    # 用户离散特征的输入层与embedding层拼接 e^u
    user_emb_list = embedding_lookup([col.name for col in sparse_feature_columns], user_input_layer_dict, embedding_layer_dict)
    user_emb = concat_func(user_emb_list)
    user_emb_output = Dense(units, activation=dnn_activation, name='user_emb_output')(user_emb)  # (None, 1, 32)
    
    # 长期序列行为编码
    # 过AttentionSequencePoolingLayer --> Concat --> DNN
    prefer_sess_length = user_input_layer_dict['prefer_sess_length']
    prefer_att_outputs = []
    # 遍历长期行为序列
    for i, prefer_emb in enumerate(prefer_emb_list):
        prefer_attention_output = AttentionSequencePoolingLayer(dropout_rate=0)([user_emb_output, prefer_emb, prefer_sess_length])
        prefer_att_outputs.append(prefer_attention_output)
    prefer_att_concat = concat_func(prefer_att_outputs)   # (None, 1, 64) <== Concat(item_embedding，cat1_embedding,cat2_embedding)
    prefer_output = Dense(units, activation=dnn_activation, name='prefer_output')(prefer_att_concat)
    # print(prefer_output.shape)   # (None, 1, 32)
    
    # 短期行为序列编码
    short_sess_length = user_input_layer_dict['short_sess_length']
    short_emb_concat = concat_func(short_emb_list)   # (None, 5, 64)   这里注意下， 对于短期序列，描述item的side info信息进行了拼接
    short_emb_input = Dense(units, activation=dnn_activation, name='short_emb_input')(short_emb_concat)  # (None, 5, 32)
    # 过rnn 这里的return_sequence=True， 每个时间步都需要输出h
    short_rnn_output = DynamicMultiRNN(num_units=units, return_sequence=True, num_layers=rnn_layers, 
                                       num_residual_layers=rnn_num_res,   # 这里竟然能用到残差
                                       dropout_rate=dropout_rate)([short_emb_input, short_sess_length])
    # print(short_rnn_output) # (None, 5, 32)
    # 过MultiHeadAttention  # (None, 5, 32)
    short_att_output = MultiHeadAttention(num_units=units, head_num=num_head, dropout_rate=dropout_rate)([short_rnn_output, short_sess_length]) # (None, 5, 64)
    # user_attention # (None, 1, 32)
    short_output = UserAttention(num_units=units, activation=dnn_activation, use_res=True, dropout_rate=dropout_rate)([user_emb_output, short_att_output, short_sess_length])
    
    # 门控融合
    gated_input = concat_func([prefer_output, short_output, user_emb_output])
    gate = Dense(units, activation='sigmoid')(gated_input)   # (None, 1, 32)
    
    # temp = tf.multiply(gate, short_output) + tf.multiply(1-gate, prefer_output)  感觉这俩一样？
    gated_output = Lambda(lambda x: tf.multiply(x[0], x[1]) + tf.multiply(1-x[0], x[2]))([gate, short_output, prefer_output])  # [None, 1,32]
    gated_output_reshape = Lambda(lambda x: tf.squeeze(x, 1))(gated_output)  # (None, 32)  这个维度必须要和docembedding层的维度一样，否则后面没法sortmax_loss
    
    # 接下来
    item_embedding_matrix = embedding_layer_dict[item_feature_name]  # 获取doc_id的embedding层
    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_input_layer_dict[item_feature_name]) # 所有doc_id的索引
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))  # 拿到所有item的embedding
    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])  # 这里依然是当可能不止item_id，或许还有brand_id, cat_id等，需要池化
    
    # 这里传入的是整个doc_id的embedding， user_embedding, 以及用户点击的doc_id，然后去进行负采样计算损失操作
    output = SampledSoftmaxLayer(num_sampled)([pooling_item_embedding_weight, gated_output_reshape, item_input_layer_dict[item_feature_name]])
    
    model = Model(inputs=user_input_layers+item_input_layers, outputs=output)
    
    # 下面是等模型训练完了之后，获取用户和item的embedding
    model.__setattr__("user_input", user_input_layers)
    model.__setattr__("user_embedding", gated_output_reshape)  # 用户embedding是取得门控融合的用户向量
    model.__setattr__("item_input", item_input_layers)
    # item_embedding取得pooling_item_embedding_weight, 这个会发现是负采样操作训练的那个embedding矩阵
    model.__setattr__("item_embedding", get_item_embedding(pooling_item_embedding_weight, item_input_layer_dict[item_feature_name]))
    
    return model