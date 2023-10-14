from collections import namedtuple
from typing import Dict, List

import tensorflow as tf
from tensorflow import keras

from myUtils import ShapeChecker

EmbeddingConfig = namedtuple('EmbeddingConfig', ['name', 'dim', 'vocab_size'])
DenseConfig = namedtuple('DenseConfig', ['name', 'dim'])



class YouTubeDNN(keras.Model):
    def __init__(
            self, sparse_feats: List[EmbeddingConfig], doc_embedding: EmbeddingConfig,
            dense_feat: List[DenseConfig], use_seq_feat: bool = True, dnn_dims = [256, 64],
            text_feat = None, text_transformer = None):
        super(YouTubeDNN, self).__init__()
        self._embedding_dict: Dict[str, keras.layers.Embedding] = {}

        for sparse_feat in sparse_feats:
            self._embedding_dict[sparse_feat.name] = keras.layers.Embedding(
                input_dim=sparse_feat.vocab_size,
                output_dim=sparse_feat.dim,
            )

        self._dnn_dims = dnn_dims + [doc_embedding.dim]
        
        self._dnn = [
            keras.layers.Dense(dim, activation=keras.activations.relu)
            for dim in self._dnn_dims
        ]

        self._doc_embedding_config = doc_embedding

        self._doc_embedding = keras.layers.Embedding(
            doc_embedding.vocab_size,
            self._dnn_dims[-1],
        )
        self._dense_feat_config = dense_feat

        self._final_bias = self.add_weight(
            "final_bias",
            shape=[self._doc_embedding_config.vocab_size],
            initializer="zeros",
            trainable=True,
        )

        self._text_feat = text_feat
        self._text_transformer = text_transformer
        self._use_seq_feat = use_seq_feat

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
        for dense_config in self._dense_feat_config:
            if dense_config.name in inputs:
                sparse_embs.append(inputs[dense_config.name])
        if self._text_feat is not None:
            for text_ft in self._text_feat:
                if text_ft in inputs:
                    tmp = self._text_transformer(inputs[text_ft])
                    tmp = tf.stop_gradient(tmp)
                    sparse_embs.append(tmp)
        
        if self._use_seq_feat:
            seq_input = inputs[self._doc_embedding_config.name + "_seq"]
            tmp_embd = self._doc_embedding(seq_input)
            shape_checker(tmp_embd, "batch T D")
            avg_seq_embd = tf.reduce_mean(tmp_embd, axis=1)
            shape_checker(avg_seq_embd, "batch D")
            avg_seq_embd = [avg_seq_embd]
        else:
            avg_seq_embd = []
        tmp_output = keras.layers.concatenate(avg_seq_embd + sparse_embs, axis=1)
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