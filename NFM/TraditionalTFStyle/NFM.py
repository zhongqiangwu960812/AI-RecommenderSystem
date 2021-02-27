# -*- coding: utf-8 -*-
# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Layer, Dense, Input, BatchNormalization

class DNN(Layer):
    """
    FC network
    """
    def __init__(self, hidden_units, activation='relu', dropout=0.):
        """
        :param hidden_units: A list.  the number of the hidden layer neural units
        :param activation: A string. Activation function of dnn.
        :param dropout: A scalar. Dropout rate
        """
        super(DNN, self).__init__()
        self.dnn_net = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dropout)
    
    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_net:
            x = dnn(x)
        x = self.dropout(x)
        
        return x

class BiInteractionPooling(Layer):
    """
    特征交叉池化层
    """
    def __init__(self):
        super(BiInteractionPooling, self).__init__()
    
    def build(self, input_shape):
        super(BiInteractionPooling, self).build(input_shape)
    
    def call(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(
            concated_embeds_value, axis=1))
        sum_of_square = tf.reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term
    
    def compute_output_shape(self, input_shape):
        return (None, 1, input_shape[-1])

class NFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0., activation='relu', bn_use=True, embed_reg=1e-4):
        """
        NFM framework
        :param features_columns: A list. dense_feaure_columns and sparse_feature_columns info
        :param hidden_units: A list.  the number of the hidden layer neural units
        :param activation: A string. Activation function of dnn.
        :param dropout: A scalar. Dropout rate
        :param bn_use: A Boolean. Use BatchNormalization or not.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(NFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                        input_length=1,
                                        output_dim = feat['embed_dim'],
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg)
                                        )
            for i, feat in enumerate(self.sparse_feature_cols)
        }
        self.bi_interaction = BiInteractionPooling()
        self.bn = BatchNormalization()
        self.bn_use = bn_use
        self.dnn_network = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1)
    
    def call(self, inputs):
        # Inputs layer
        dense_inputs, sparse_inputs = inputs
        # Embedding layer
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])  # (None, field_num, embed_dim)
        
        # 特征交叉池化层
        bi_out = self.bi_interaction(embed)
        # 与连续特征合并
        x = tf.concat([dense_inputs, bi_out], axis=-1)
        
        # BatchNormalization
        x = self.bn(x, training=self.bn_use)
        
        # dnn
        x = self.dnn_network(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        
        return outputs
    
    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_cols), ), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_cols), ), dtype=tf.float32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()
