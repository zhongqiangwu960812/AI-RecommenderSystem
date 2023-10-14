from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal, Zeros, glorot_normal

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from tensorflow.python.keras.regularizers import l2
#from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss

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

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)
    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)
    def call(self, x, mask=None, **kwargs):
        return x

def squash(inputs):
    vec_squared_norm = reduce_sum(tf.square(inputs), axis=-1, keep_dims=True)  # (None, 2, 1)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * inputs
    return vec_squashed

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

def tile_user_otherfeat(user_other_feature, k_max):
    return tf.tile(tf.expand_dims(user_other_feature, -2), [1, k_max, 1])

class CapsuleLayer(Layer):
    def __init__(self, input_units, out_units, max_len, k_max, iteration_times=3, init_std=1.0, **kwargs):
        self.input_units = input_units
        self.out_units = out_units
        self.max_len = max_len
        self.k_max = k_max
        self.iteration_times = iteration_times
        self.init_std = init_std
        super(CapsuleLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # 路由对数，大小是1，2，50， 即每个路由对数与输入胶囊个数一一对应，同时如果有两组输出胶囊的话， 那么这里就需要2组B
        self.routing_logits = self.add_weight(shape=[1, self.k_max, self.max_len],
                                             initializer=RandomNormal(stddev=self.init_std),
                                             trainable=False, name='B', dtype=tf.float32)
        # 双线性映射矩阵，维度是[输入胶囊维度，输出胶囊维度] 这样才能进行映射
        self.bilinear_mapping_matrix = self.add_weight(shape=[self.input_units, self.out_units],
                                                       initializer=RandomNormal(stddev=self.init_std),
                                                       name="S", dtype=tf.float32)
        super(CapsuleLayer, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        # input (hist_emb, hist_len) ,其中hist_emb是(None, seq_len, emb_dim), hist_len是(none, 1) batch and sel_len
        behavior_embeddings, seq_len = inputs
        batch_size = tf.shape(behavior_embeddings)[0]
        seq_len_tile = tf.tile(seq_len, [1, self.k_max])  # 在第二个维度上复制一份 k_max个输出胶囊嘛 (None, 2)  第一列和第二列都是序列真实长度
        
        for i in range(self.iteration_times):
            mask = tf.sequence_mask(seq_len_tile, self.max_len)  #（None, 2, 50) 第一个维度是样本，第二个维度是胶囊，第三个维度是[True, True, ..False, False, ..]
            pad = tf.ones_like(mask, dtype=tf.float32) * (-2**32+1) # (None, 2, 50)  被mask的位置是非常小的数，这样softmax的时候这个位置正好是0
            routing_logits_with_padding = tf.where(mask, tf.tile(self.routing_logits, [batch_size, 1, 1]), pad) 
            # (None, 2, 50) 沿着batch_size进行复制，每个样本都得有这样一套B，2个输出胶囊， 50个输入胶囊

            weight = tf.nn.softmax(routing_logits_with_padding) # (none, 2, 50)  # softmax得到权重
    
            # (None, seq_len, emb_dim) * (emb_dim, out_emb) = (None, 50, 8) axes=1表示a的最后1个维度，与b进行张量乘法
            behavior_embedding_mapping = tf.tensordot(behavior_embeddings, self.bilinear_mapping_matrix, axes=1)
            
            Z = tf.matmul(weight, behavior_embedding_mapping) # (None, 2, 8)即 上面的B与behavior_embed_map加权求和
            interest_capsules = squash(Z) # (None, 2, 8) 
            
            delta_routing_logits = reduce_sum(
                # (None, 2, 8)  * (None, 8, 50) = (None, 2, 50)
                tf.matmul(interest_capsules, tf.transpose(behavior_embedding_mapping, perm=[0, 2, 1])),
                axis=0, keep_dims=True
            ) # (1, 2, 50)   样本维度这里相加  所有样本一块为聚类做贡献

            self.routing_logits.assign_add(delta_routing_logits)  # 原来的基础上加上这个东西 （1, 2, 50)
        
        interest_capsules = tf.reshape(interest_capsules, [-1, self.k_max, self.out_units])  # (None, 2, 8)
        return interest_capsules
    
    def compute_output_shape(self, input_shape):
        return (None, self.k_max, self.out_units)
    
    # 下面这个如果需要保存模型的时候会用到
    def get_config(self, ):
        config = {'input_units': self.input_units, 'out_units': self.out_units, 'max_len': self.max_len,
                  'k_max': self.k_max, 'iteration_times': self.iteration_times, "init_std": self.init_std}
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

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

class EmbeddingIndex(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)
    def build(self, input_shape):
        super(EmbeddingIndex, self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, **kwargs):
        return tf.constant(self.index)

class LabelAwareAttention(Layer):
    def __init__(self, k_max, pow_p=1, **kwargs):
        self.k_max = k_max
        self.pow_p = pow_p
        super(LabelAwareAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.embedding_size = input_shape[0][-1]
        super(LabelAwareAttention, self).build(input_shape)
    def call(self, inputs):
        keys, query = inputs[0], inputs[1]  # keys (None, 2, 8)  query (None, 1, 8)
        weight = reduce_sum(keys * query, axis=-1, keep_dims=True)  # (None, 2, 1)
        weight = tf.pow(weight, self.pow_p)  # (None, 2, 1)
        
        # k如果需要动态调整，那么这里就根据实际长度mask操作，这样被mask的输出胶囊的权重为0， 发挥不出作用了
        if len(inputs) == 3:
            k_user = tf.cast(tf.maximum(
                1.,
                tf.minimum(
                    tf.cast(self.k_max, dtype="float32"),  # k_max
                    tf.log1p(tf.cast(inputs[2], dtype="float32")) / tf.log(2.)  # hist_len
                )
            ), dtype="int64")
            seq_mask = tf.transpose(tf.sequence_mask(k_user, self.k_max), [0, 2, 1])
            padding = tf.ones_like(seq_mask, dtype=tf.float32) * (-2 ** 32 + 1)  # [x,k_max,1]
            weight = tf.where(seq_mask, weight, padding)
        
        weight = softmax(weight, dim=1, name='weight')
        output = reduce_sum(keys * weight, axis=1)  # (None, 8)
        return output

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

def Mind(user_feature_columns, item_feature_columns, num_sampled=5, k_max=2, p=1.0, dynamic_k=False, user_dnn_hidden_units=(64, 32),
        dnn_activation='relu', dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, output_activation='linear', seed=1024):
    """
        :param k_max: 用户兴趣胶囊的最大个数
    """
    # 目前这里只支持item_feature_columns为1的情况，即只能转入item_id
    if len(item_feature_columns) > 1:
        raise ValueError("Now MIND only support 1 item feature like item_id")
    
    # 获取item相关的配置参数
    item_feature_column = item_feature_columns[0]
    item_feature_name = item_feature_column.name
    item_vocabulary_size = item_feature_column.vocabulary_size
    item_embedding_dim = item_feature_column.embedding_dim
    
    behavior_feature_list = [item_feature_name]
    
    # 为用户特征创建Input层
    user_input_layer_dict = build_input_layers(user_feature_columns)
    item_input_layer_dict = build_input_layers(item_feature_columns)
    # 将Input层转化成列表的形式作为model的输入
    user_input_layers = list(user_input_layer_dict.values())
    item_input_layers = list(item_input_layer_dict.values())
    
    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), user_feature_columns)) if user_feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), user_feature_columns)) if user_feature_columns else []
    varlen_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), user_feature_columns)) if user_feature_columns else []
    
    # 由于这个变长序列里面只有历史点击文章，没有类别啥的，所以这里直接可以用varlen_feature_columns
    # deepctr这里单独把点击文章这个放到了history_feature_columns
    seq_max_len = varlen_feature_columns[0].maxlen
    
    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(user_feature_columns+item_feature_columns)
    
    # 获取当前的行为特征(doc)的embedding，这里面可能又多个类别特征，所以需要pooling下
    query_embed_list = embedding_lookup(behavior_feature_list, item_input_layer_dict, embedding_layer_dict)  # 长度为1 
    # 获取行为序列(doc_id序列, hist_doc_id) 对应的embedding，这里有可能有多个行为产生了行为序列，所以需要使用列表将其放在一起
    keys_embed_list = embedding_lookup([varlen_feature_columns[0].name], user_input_layer_dict, embedding_layer_dict)  # 长度为1
    
    # 用户离散特征的输入层与embedding层拼接
    dnn_input_emb_list = embedding_lookup([col.name for col in sparse_feature_columns], user_input_layer_dict, embedding_layer_dict)
    
    # 获取dense
    dnn_dense_input = []
    for fc in dense_feature_columns:
        if fc.name != 'hist_len':  # 连续特征不要这个
            dnn_dense_input.append(user_input_layer_dict[fc.name])
    
    # 把keys_emb_list和query_emb_listpooling操作， 这是因为可能每个商品不仅有id，还可能用类别，品牌等多个embedding向量，这种需要pooling成一个
    history_emb = PoolingLayer()(NoMask()(keys_embed_list))  # (None, 50, 8)
    target_emb = PoolingLayer()(NoMask()(query_embed_list))   # (None, 1, 8)
    
    hist_len = user_input_layer_dict['hist_len']
    # 胶囊网络
    # (None, 2, 8) 得到了两个兴趣胶囊
    high_capsule = CapsuleLayer(input_units=item_embedding_dim, out_units=item_embedding_dim,
                                max_len=seq_max_len, k_max=k_max)((history_emb, hist_len))
    
    
    # 把用户的其他特征拼接到胶囊网络上来
    if len(dnn_input_emb_list) > 0 or len(dnn_dense_input) > 0:
        user_other_feature = combined_dnn_input(dnn_input_emb_list, dnn_dense_input)
        # (None, 2, 32)   这里会发现其他的用户特征是每个胶囊复制了一份，然后拼接起来
        other_feature_tile = tf.keras.layers.Lambda(tile_user_otherfeat, arguments={'k_max': k_max})(user_other_feature) 
        user_deep_input = Concatenate()([NoMask()(other_feature_tile), high_capsule]) # (None, 2, 40)
    else:
        user_deep_input = high_capsule
        
    # 接下来过一个DNN层，获取最终的用户表示向量 如果是三维输入， 那么最后一个维度与w相乘，所以这里如果不自己写，可以用Dense层的列表也可以
    user_embeddings = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn,
                          dnn_dropout, dnn_use_bn, output_activation=output_activation, seed=seed,
                          name="user_embedding")(user_deep_input)  # (None, 2, 8)
    
    # 接下来，过Label-aware layer
    if dynamic_k:
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p,)((user_embeddings, target_emb, hist_len))
    else:
        user_embedding_final = LabelAwareAttention(k_max=k_max, pow_p=p,)((user_embeddings, target_emb))
    
    # 接下来
    item_embedding_matrix = embedding_layer_dict[item_feature_name]  # 获取doc_id的embedding层
    item_index = EmbeddingIndex(list(range(item_vocabulary_size)))(item_input_layer_dict[item_feature_name]) # 所有doc_id的索引
    item_embedding_weight = NoMask()(item_embedding_matrix(item_index))  # 拿到所有item的embedding
    pooling_item_embedding_weight = PoolingLayer()([item_embedding_weight])  # 这里依然是当可能不止item_id，或许还有brand_id, cat_id等，需要池化
    
    # 这里传入的是整个doc_id的embedding， user_embedding, 以及用户点击的doc_id，然后去进行负采样计算损失操作
    output = SampledSoftmaxLayer(num_sampled)([pooling_item_embedding_weight, user_embedding_final, item_input_layer_dict[item_feature_name]])
    
    model = Model(inputs=user_input_layers+item_input_layers, outputs=output)
    
    # 下面是等模型训练完了之后，获取用户和item的embedding
    model.__setattr__("user_input", user_input_layers)
    model.__setattr__("user_embedding", user_embeddings)
    model.__setattr__("item_input", item_input_layers)
    model.__setattr__("item_embedding", get_item_embedding(pooling_item_embedding_weight, item_input_layer_dict[item_feature_name]))
    
    return model