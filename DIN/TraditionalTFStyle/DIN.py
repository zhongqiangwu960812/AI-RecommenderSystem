import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, BatchNormalization, Dense, Input, PReLU, Embedding, Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

class Attention_layer(Layer):
    """
    自定义Attention层， 这个就是一个全连接神经网络
    """
    def __init__(self, att_hidden_units, activation='sigmoid'):
        super(Attention_layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)
    
    # forward
    def call(self, inputs):
        """
        这里的inputs包含四部分： [item_embed, seq_embed, seq_embed, mask]
        
        item_embed: 这个是候选商品的embedding向量   维度是(None, embedding_dim * behavior_num)   # behavior_num能表示用户行为的特征个数 这里是1， 所以(None, embed_dim)
        seq_embed: 这个是用户历史商品序列的embedding向量， 维度是(None, max_len, embedding_dim * behavior_num)  (None, max_len, embed_dim)
        mask:  维度是(None, max_len)   这个里面每一行是[False, False, True, True, ....]的形式， False的长度表示样本填充的那部分
        """
        q, k, v, key_masks = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])   # (None, max_len*embedding)       # 沿着k.shap[1]的维度复制  毕竟每个历史行为都要和当前的商品计算相似关系
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])      # (None, max_len, emebdding_dim
        
        # q, k, out product should concat
        info = tf.concat([q, k, q-k, q*k], axis=-1)   # (None, max_len, 4*emebdding_dim)
        
        # n层全连接
        for dense in self.att_dense:
            info = dense(info)
        
        outputs = self.att_final_dense(info)      # (None,  max_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)    # (None, max_len)
        
        # mask 把每个行为序列填充的那部分替换成很小的一个值
        paddings = tf.ones_like(outputs) * (-2**32+1)      # (None, max_len)  这个就是之前填充的那个地方， 我们补一个很小的值
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        
        # softmax
        outputs = tf.nn.softmax(logits=outputs) # (None, max_len)
        outputs = tf.expand_dims(outputs, axis=1)   # (None, 1, max_len) 
        
        outputs = tf.matmul(outputs, v)   # 三维矩阵相乘， 相乘发生在后两维   (None, 1, max_len) * (None, max_len, embed_dim) = (None, 1, embed_dim)
        outputs = tf.squeeze(outputs, axis=1)  # (None, embed_dim)
        
        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')
    
    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        
        return self.alpha * (1.0-x_p) * x + x_p * x

class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40), ffn_hidden_units=(80, 40), att_activation='sigmoid', 
                 ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_reg=1e-4):
        """
        DIN:
        feature_columns:列表, [dense_feature_columns,sparse_feature_columns],dense_feature_columns是[{'feat': 'feat_name'}], 而sparse_feature_columns是[{'feat':'feat_name', 'feat_num': 'nunique', 'embed_dim'}]
        behavior_feature_list: 列表. 能表示用户历史行为的特征, 比如商品id， 店铺id ['item', 'cat']
        att_hidden_units: 注意力层的隐藏单元个数.可以是一个列表或者元组，毕竟注意力层也是一个全连接的网络嘛
        ffn_hidden_units:全连接层的隐藏单元个数和层数，可以是一个列表或者元组  (80, 40)  就表示两层神经网络， 第一层隐藏层单元80个， 第二层40个
        att_activation: 激活单元的名称， 字符串
        ffn_activation: 激活单元的名称， 用'prelu'或者'Dice'  
        maxlen: 标量. 用户历史行为序列的最大长度
        dropout: 标量，失活率
        embed_reg: 标量. 正则系数
        """
        super(DIN, self).__init__()      # 初始化网络
        self.maxlen = maxlen
        
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns           # 这里把连续特征和离散特征分别取出来， 因为后面两者的处理方式不同
        
        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)      # 这个other_sparse就是离散特征中去掉了能表示用户行为的特征列
        self.dense_len = len(self.dense_feature_columns)    
        self.behavior_num = len(behavior_feature_list)
        
        # embedding层， 这里分为两部分的embedding， 第一部分是普通的离散特征， 第二部分是能表示用户历史行为的离散特征， 这一块后面要进注意力和当前的商品计算相关性
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'], 
                                              input_length=1, 
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg)
                                             ) for feat in self.sparse_feature_columns if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and catetory id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'], 
                                           input_length=1, 
                                           output_dim=feat['embed_dim'], 
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg)
                                          ) for feat in self.sparse_feature_columns if feat['feat'] in behavior_feature_list]
        
        # 注意力机制
        self.attention_layer = Attention_layer(att_hidden_units, att_activation)
        
        self.bn = BatchNormalization(trainable=True)
        
        # 全连接网络
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation=='prelu' else Dice()) for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)
        
    def call(self, inputs):
        """
        inputs: [dense_input, sparse_input, seq_input, item_input]  ， 第二部分是离散型的特征输入， 第三部分是用户的历史行为， 第四部分是当前商品的输入
    
        dense_input： 连续型的特征输入， 维度是(None, dense_len)
        sparse_input: 离散型的特征输入， 维度是(None, other_sparse_len)
        seq_inputs: 用户的历史行为序列(None, maxlen, behavior_len)
        item_inputs： 当前的候选商品序列 (None, behavior_len)
        """
        
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        
        # attention --->mask, if the element of seq_inputs is equal 0, it must be filled in  这是因为很多序列由于不够长用0进行了填充,并且是前面补的0
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)  类型转换函数， 把seq_input中不等于0的值转成float32
        # 这个函数的作用就是每一行样本中， 不为0的值返回1， 为0的值返回0， 这样就把填充的那部分值都给标记了出来
        
        # 下面把连续型特征和行为无关的离散型特征拼到一块先
        other_info = dense_inputs   # (None, dense_len)
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)      # (None, dense_len+other_sparse_len)
        
        # 下面把候选的商品和用户历史行为商品也各自的拼接起来
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_num)], axis=-1)   # [None, max_len, embed_dim]
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.behavior_num)], axis=-1)  # [None, embed_dim]
        
    
        # 下面进行attention_layer的计算
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed, mask])   # (None, embed_dim) 
        
        # 所有特征拼起来了
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)   # (None, dense_len + other_sparse_len + embed_dim+embed_dim)  
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1) # (None, embed_dim+embed_dim)
        
        info_all = self.bn(info_all)
        
        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)
        
        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all))
        return outputs
    
    def summary(self):
        dense_inputs = Input(shape=(self.dense_len), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.maxlen, self.behavior_num), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_num,), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs], 
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()