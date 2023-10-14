import pickle
import random
from datetime import datetime
import collections

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss
from SDM import Sdm

from annoy import AnnoyIndex


"""构造数据集"""
def get_data_set(click_data, seq_short_len=5, seq_prefer_len=50):
    """
    :param: seq_short_len: 短期会话的长度
    :param: seq_prefer_len: 会话的最长长度
    """
    click_data.sort_values("expo_time", inplace=True)
    
    train_set, test_set = [], []
    for user_id, hist_click in tqdm(click_data.groupby('user_id')):
        pos_list = hist_click['article_id'].tolist()
        cat1_list = hist_click['cat_1'].tolist()
        cat2_list = hist_click['cat_2'].tolist()
        
        # 滑动窗口切分数据
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            cat1_hist = cat1_list[:i]
            cat2_hist = cat2_list[:i]
            # 序列长度只够短期的
            if i <= seq_short_len and i != len(pos_list) - 1:
                train_set.append((
                    # 用户id, 用户短期历史行为序列， 用户长期历史行为序列， 当前行为文章， label， 
                    user_id, hist[::-1], [0]*seq_prefer_len, pos_list[i], 1, 
                    # 用户短期历史序列长度， 用户长期历史序列长度， 
                    len(hist[::-1]), 0, 
                    # 用户短期历史序列对应类别1， 用户长期历史行为序列对应类别1
                    cat1_hist[::-1], [0]*seq_prefer_len, 
                    # 历史短期历史序列对应类别2， 用户长期历史行为序列对应类别2 
                    cat2_hist[::-1], [0]*seq_prefer_len
                ))
            # 序列长度够长期的
            elif i != len(pos_list) - 1:
                train_set.append((
                    # 用户id, 用户短期历史行为序列，用户长期历史行为序列， 当前行为文章， label
                    user_id, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1, 
                    # 用户短期行为序列长度，用户长期行为序列长度，
                    seq_short_len, len(hist[::-1])-seq_short_len,
                    # 用户短期历史行为序列对应类别1， 用户长期历史行为序列对应类别1
                    cat1_hist[::-1][:seq_short_len], cat1_hist[::-1][seq_short_len:],
                    # 用户短期历史行为序列对应类别2， 用户长期历史行为序列对应类别2
                    cat2_hist[::-1][:seq_short_len], cat2_hist[::-1][seq_short_len:]             
                ))
            # 测试集保留最长的那一条
            elif i <= seq_short_len and i == len(pos_list) - 1:
                test_set.append((
                    user_id, hist[::-1], [0]*seq_prefer_len, pos_list[i], 1,
                    len(hist[::-1]), 0, 
                    cat1_hist[::-1], [0]*seq_perfer_len, 
                    cat2_hist[::-1], [0]*seq_prefer_len
                ))
            else:
                test_set.append((
                    user_id, hist[::-1][:seq_short_len], hist[::-1][seq_short_len:], pos_list[i], 1,
                    seq_short_len, len(hist[::-1])-seq_short_len, 
                    cat1_hist[::-1][:seq_short_len], cat1_hist[::-1][seq_short_len:],
                    cat2_list[::-1][:seq_short_len], cat2_hist[::-1][seq_short_len:]
                ))
                
    random.shuffle(train_set)
    random.shuffle(test_set)
        
    return train_set, test_set


"""构造模型输入"""
# 构造SDM模型的输入
def gen_model_input(train_set, user_profile, seq_short_len, seq_prefer_len):
    """构造模型输入"""
    # row: [user_id, short_train_seq, perfer_train_seq, item_id, label, short_len, perfer_len, cat_1_short, cat_1_perfer, cat_2_short, cat_2_prefer]
    train_uid = np.array([row[0] for row in train_set])
    short_train_seq = [row[1] for row in train_set]
    prefer_train_seq = [row[2] for row in train_set]
    train_iid = np.array([row[3] for row in train_set])
    train_label = np.array([row[4] for row in train_set])
    train_short_len = np.array([row[5] for row in train_set])
    train_prefer_len = np.array([row[6] for row in train_set])
    short_train_seq_cat1 = np.array([row[7] for row in train_set])
    prefer_train_seq_cat1 = np.array([row[8] for row in train_set])
    short_train_seq_cat2 = np.array([row[9] for row in train_set])
    prefer_train_seq_cat2 = np.array([row[10] for row in train_set])
    
    # padding操作
    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_len, padding='post', truncating='post', value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_len, padding='post', truncating='post', value=0)
    train_short_cat1_pad = pad_sequences(short_train_seq_cat1, maxlen=seq_short_len, padding='post', truncating='post', value=0)
    train_prefer_cat1_pad = pad_sequences(prefer_train_seq_cat1, maxlen=seq_prefer_len, padding='post', truncating='post', value=0)
    train_short_cat2_pad = pad_sequences(short_train_seq_cat2, maxlen=seq_short_len, padding='post', truncating='post', value=0)
    train_prefer_cat2_pad = pad_sequences(prefer_train_seq_cat2, maxlen=seq_prefer_len, padding='post', truncating='post', value=0)
    
    # 形成输入词典
    train_model_input = {
        "user_id": train_uid,
        "doc_id": train_iid,
        "short_doc_id": train_short_item_pad,
        "prefer_doc_id": train_prefer_item_pad,
        "prefer_sess_length": train_prefer_len,
        "short_sess_length": train_short_len,
        "short_cat1": train_short_cat1_pad,
        "prefer_cat1": train_prefer_cat1_pad,
        "short_cat2": train_short_cat2_pad,
        "prefer_cat2": train_prefer_cat2_pad
    }
    
    # 其他的用户特征加入
    for key in ["gender", "age", "city"]:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
    
    return train_model_input, train_label


"""训练SDM模型"""
def train_sdm_model(train_model_input, train_label, embedding_dim, feature_max_idx, SEQ_LEN_short, SEQ_LEN_prefer,batch_size, epochs, verbose, validation_split):
    """构建sdm并完成训练"""
    # 建立模型
    user_feature_columns = [
        SparseFeat('user_id', feature_max_idx['user_id'], 16),
        SparseFeat('gender', feature_max_idx['gender'], 16),
        SparseFeat('age', feature_max_idx['age'], 16),
        SparseFeat('city', feature_max_idx['city'], 16),
        
        VarLenSparseFeat(SparseFeat('short_doc_id', feature_max_idx['article_id'], embedding_dim, embedding_name="doc_id"), SEQ_LEN_short, 'mean', 'short_sess_length'),    
        VarLenSparseFeat(SparseFeat('prefer_doc_id', feature_max_idx['article_id'], embedding_dim, embedding_name='doc_id'), SEQ_LEN_prefer, 'mean', 'prefer_sess_length'),
        VarLenSparseFeat(SparseFeat('short_cat1', feature_max_idx['cat_1'], embedding_dim, embedding_name='cat_1'), SEQ_LEN_short, 'mean', 'short_sess_length'),
        VarLenSparseFeat(SparseFeat('prefer_cat1', feature_max_idx['cat_1'], embedding_dim, embedding_name='cat_1'), SEQ_LEN_prefer, 'mean', 'prefer_sess_length'),
        VarLenSparseFeat(SparseFeat('short_cat2', feature_max_idx['cat_2'], embedding_dim, embedding_name='cat_2'), SEQ_LEN_short, 'mean', 'short_sess_length'),
        VarLenSparseFeat(SparseFeat('prefer_cat2', feature_max_idx['cat_2'], embedding_dim, embedding_name='cat_2'), SEQ_LEN_prefer, 'mean', 'prefer_sess_length'),
    ]

    item_feature_columns = [SparseFeat('doc_id', feature_max_idx['article_id'], embedding_dim)]
    
    # 定义模型
    model = Sdm(user_feature_columns, item_feature_columns, history_feature_list=['doc_id', 'cat1', 'cat2'])
    
    # 模型编译
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    
    # 模型训练，这里可以定义验证集的比例，如果设置为0的话就是全量数据直接进行训练
    history = model.fit(train_model_input, train_label, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split)
    
    return model

"""获取用户embedding和文章embedding"""
def get_embeddings(model, test_model_input, user_idx_2_rawid, doc_idx_2_rawid, save_path='embedding/'):
    doc_model_input = {'doc_id':np.array(list(doc_idx_2_rawid.keys()))}
    
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    doc_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    
    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
    user_embs = user_embedding_model.predict(test_model_input, batch_size=2 ** 12)
    doc_embs = doc_embedding_model.predict(doc_model_input, batch_size=2 ** 12)
    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    
    # 将Embedding转换成字典的形式方便查询
    raw_user_id_emb_dict = {user_idx_2_rawid[k]: \
                                v for k, v in zip(user_idx_2_rawid.keys(), user_embs)}
    raw_doc_id_emb_dict = {doc_idx_2_rawid[k]: \
                                v for k, v in zip(doc_idx_2_rawid.keys(), doc_embs)}
    # 将Embedding保存到本地
    pickle.dump(raw_user_id_emb_dict, open(save_path + 'user_mind_emb.pkl', 'wb'))
    pickle.dump(raw_doc_id_emb_dict, open(save_path + 'doc_mind_emb.pkl', 'wb'))
    
    # 读取
    #user_embs_dict = pickle.load(open('embedding/user_youtube_emb.pkl', 'rb'))
    #doc_embs_dict = pickle.load(open('embedding/doc_youtube_emb.pkl', 'rb'))
    return user_embs, doc_embs

"""最近邻检索得到召回结果"""
def get_sdm_recall_res(user_embs, doc_embs, user_idx_2_rawid, doc_idx_2_rawid, topk):
    """近邻检索，这里用annoy tree"""
    # 把doc_embs构建成索引树
    f = user_embs.shape[1]
    t = AnnoyIndex(f, 'angular')
    for i, v in enumerate(doc_embs):
        t.add_item(i, v)
    t.build(10)
    # 可以保存该索引树 t.save('annoy.ann')
    
    # 每个用户向量， 返回最近的TopK个item
    user_recall_items_dict = collections.defaultdict(dict)
    for i, u in enumerate(user_embs):
        recall_doc_scores = t.get_nns_by_vector(u, topk, include_distances=True)
        # recall_doc_scores是(([doc_idx], [scores]))， 这里需要转成原始doc的id
        raw_doc_scores = list(recall_doc_scores)
        raw_doc_scores[0] = [doc_idx_2_rawid[i] for i in raw_doc_scores[0]]
        # 转换成实际用户id
        try:
            user_recall_items_dict[user_idx_2_rawid[i]] = dict(zip(*raw_doc_scores))
        except:
            continue
    
    # 默认是分数从小到大排的序， 这里要从大到小
    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in user_recall_items_dict.items()}
    
    # 保存一份
    pickle.dump(user_recall_items_dict, open('sdm_u2i_dict.pkl', 'wb'))
    
    return user_recall_items_dict