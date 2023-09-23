from collections import namedtuple
from typing import Dict, List

import tensorflow as tf
from tensorflow import keras

from myUtils import ShapeChecker

EmbeddingConfig = namedtuple('EmbeddingConfig', ['name', 'dim', 'vocab_size'])
# DenseConfig = namedtuple('DenseConfig', ['name', 'func'])



class YouTubeDNN(keras.Model):
    def __init__(self, sparse_feats: List[EmbeddingConfig], doc_embedding: EmbeddingConfig):
        super(YouTubeDNN, self).__init__()
        self._embedding_dict: Dict[str, keras.layers.Embedding] = {}

        for seq_feat in sparse_feats:
            self._embedding_dict[seq_feat.name] = keras.layers.Embedding(
                input_dim=seq_feat.vocab_size,
                output_dim=seq_feat.dim,
            )

        self._dnn_dims = [256, 64, doc_embedding.dim]
        
        self._dnn = [
            keras.layers.Dense(dim, activation=keras.activations.relu)
            for dim in self._dnn_dims
        ]

        self._doc_embedding_config = doc_embedding

        self._doc_embedding = keras.layers.Embedding(
            doc_embedding.vocab_size,
            self._dnn_dims[-1],
        )

        self._final_bias = self.add_weight(
            "final_bias",
            shape=[self._doc_embedding_config.vocab_size],
            initializer="zeros",
            trainable=True,
        )

    # def build(self, shape):
    #     self._final_bias = self.add_weight(
    #         "final_bias",
    #         shape=[self._doc_embedding_config.vocab_size],
    #         initializer="zeros",
    #         trainable=True,
    #     )


    def call(self, inputs: Dict[str, tf.Tensor]):
        shape_checker = ShapeChecker()
        sparse_embs: List[tf.Tensor] = []
        for feat_name in self._embedding_dict.keys():
            if feat_name in inputs:
                sparse_embs.append(
                    self._embedding_dict[feat_name](inputs[feat_name])
                )
        
        seq_input = inputs[self._doc_embedding_config.name + "_seq"]
        tmp_embd = self._doc_embedding(seq_input)
        shape_checker(tmp_embd, "batch T D")
        avg_seq_embd = tf.reduce_mean(tmp_embd, axis=1)
        shape_checker(avg_seq_embd, "batch D")


        tmp_output = keras.layers.concatenate([avg_seq_embd] + sparse_embs, axis=1)
        shape_checker(tmp_output, "batch bottom_width")
        for layer in self._dnn:
            tmp_output = layer(tmp_output)
        shape_checker(tmp_output, "batch D")

        doc_id = inputs[self._doc_embedding_config.name]

        doc_embedding = self._doc_embedding(doc_id)
        # tf.print("doc_embedding ", tf.shape(doc_embedding))
        shape_checker(doc_embedding, "batch D")

        tmp_output = doc_embedding * tmp_output  # B,dim
        tmp_output = tf.reduce_sum(tmp_output, axis=1, keepdims=True)
        shape_checker(tmp_output, "batch 1")

        
        final_bias = tf.gather(self._final_bias, doc_id)
        final_bias = tf.expand_dims(final_bias, axis=1)
        shape_checker(final_bias, "batch 1")
        tmp_output = tmp_output + final_bias
        tmp_output = keras.activations.sigmoid(tmp_output)
        # tf.print("final_bias ", tf.shape(final_bias))
        return tmp_output

    # def summary(self, **kwargs):
    #     pass