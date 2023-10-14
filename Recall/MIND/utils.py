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
from MIND import Mind 

from annoy import AnnoyIndex


"""构造数据集"""
def gen_neg_sample_candiate(pos_list, item_ids, doc_clicked_count_dict, negsample, methods='multinomial'):
    
    # w2v的负样本采样， 增加高热item成为负样本的概率
    # 参考https://blog.csdn.net/weixin_42299244/article/details/112734531, 这里用tf相应函数替换
    # tf.random.categoral  非相关数据的降采样
    if methods == 'multinomial':
        # input表示每个item出现的次数
        items, item_counts = [], []
        # 用户未点击的item   这个遍历效率很低  后面看看能不能优化下
#         for item_id in list(set(item_ids)-set(pos_list)):
#             items.append(item_id)
#             item_counts.append(doc_clicked_count_dict[item_id]*1.0)
        
        items = list(doc_clicked_count_dict.keys())
        item_counts = list(doc_clicked_count_dict.values())
        item_freqs = np.array(item_counts) / np.sum(item_counts)
        item_freqs = item_freqs ** 0.75
        item_freqs = item_freqs / np.sum(item_freqs)
        neg_item_index = tf.random.categorical(item_freqs.reshape(1, -1), len(pos_list)*negsample)
        neg_list = np.array([items[i] for i in neg_item_index.numpy().tolist()[0] if items[i] not in pos_list])
        random.shuffle(neg_list)
        
    # 曝光样本随机采
    else:
        candidate_set = list(set(item_ids) - set(pos_list))  #  热度采样
        neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)  # 对于每个正样本，选择n个负样本
    
    return neg_list


def gen_data_set(click_data, doc_clicked_count_dict, negsample, control_users=False):
    """构造youtubeDNN的数据集"""
    # 按照曝光时间排序
    click_data.sort_values("expo_time", inplace=True)
    item_ids = click_data['article_id'].unique()
    
    train_set, test_set = [], []
    for user_id, hist_click in tqdm(click_data.groupby('user_id')):
        # 这里按照expo_date分开，每一天用滑动窗口滑，可能相关性更高些,另外，这样序列不会太长，因为eda发现有点击1111个的
        #for expo_date, hist_click in hist_date_click.groupby('expo_date'):
        # 用户当天的点击历史id
        pos_list = hist_click['article_id'].tolist()
        user_control_flag = True
        
        if control_users:
            user_samples_cou = 0
        
        # 过长的序列截断
        if len(pos_list) > 50:
            pos_list = pos_list[-50:]

        if negsample > 0:
            neg_list = gen_neg_sample_candiate(pos_list, item_ids, doc_clicked_count_dict, negsample, methods='multinomial')
        
        # 只有1个的也截断 去掉，当然我之前做了处理，这里没有这种情况了
        if len(pos_list) < 2:
            continue
        else:
            # 序列至少是2
            for i in range(1, len(pos_list)):
                hist = pos_list[:i]
                # 这里采用打压热门item策略，降低高展item成为正样本的概率
                freq_i = doc_clicked_count_dict[pos_list[i]] / (np.sum(list(doc_clicked_count_dict.values())))
                p_posi = (np.sqrt(freq_i/0.001)+1)*(0.001/freq_i)
                
                # p_posi=0.3  表示该item_i成为正样本的概率是0.3，
                if user_control_flag and i != len(pos_list) - 1:
                    if random.random() > (1-p_posi):
                        row = [user_id, hist[::-1], pos_list[i], hist_click.iloc[0]['city'], hist_click.iloc[0]['age'], hist_click.iloc[0]['gender'], 1, len(hist[::-1])]
                        train_set.append(row)
                        
                        for negi in range(negsample):
                            row = [user_id, hist[::-1], neg_list[i*negsample+negi], hist_click.iloc[0]['city'], hist_click.iloc[0]['age'], hist_click.iloc[0]['gender'], 0, len(hist[::-1])]
                            train_set.append(row)
                        
                        if control_users:
                            user_samples_cou += 1
                            # 每个用户序列最长是50， 即每个用户正样本个数最多是50个, 如果每个用户训练样本数量到了30个，训练集不能加这个用户了
                            if user_samples_cou > 30:  
                                user_samples_cou = False
                
                # 整个序列加入到test_set， 注意，这里一定每个用户只有一个最长序列，相当于测试集数目等于用户个数
                elif i == len(pos_list) - 1:
                    row = [user_id, hist[::-1], pos_list[i], hist_click.iloc[0]['city'], hist_click.iloc[0]['age'], hist_click.iloc[0]['gender'], 0, len(hist[::-1])]
                    test_set.append(row)
    
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    return train_set, test_set   

"""构造模型输入"""
def gen_model_input(train_set, his_seq_max_len):
    """构造模型的输入"""
    # row: [user_id, hist_list, cur_doc_id, city, age, gender, label, hist_len]
    train_uid = np.array([row[0] for row in train_set])
    train_hist_seq = [row[1] for row in train_set]
    train_iid = np.array([row[2] for row in train_set])
    train_u_city = np.array([row[3] for row in train_set])
    train_u_age = np.array([row[4] for row in train_set])
    train_u_gender = np.array([row[5] for row in train_set])
    train_label = np.array([row[6] for row in train_set])
    train_hist_len = np.array([row[7] for row in train_set])
    
    train_seq_pad = pad_sequences(train_hist_seq, maxlen=his_seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {
        "user_id": train_uid,
        "doc_id": train_iid,
        "hist_doc_id": train_seq_pad,
        "hist_len": train_hist_len,
        "u_city": train_u_city,
        "u_age": train_u_age,
        "u_gender": train_u_gender, 
    }
    return train_model_input, train_label


"""训练MIND模型"""
def train_mind_model(train_model_input, train_label, embedding_dim, feature_max_idx, his_seq_maxlen, batch_size, epochs, verbose, validation_split):
    """构建mind并完成训练"""
    # 建立模型
    user_feature_columns = [
            SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
            VarLenSparseFeat(SparseFeat('hist_doc_id', feature_max_idx['article_id'], embedding_dim,
                                                            embedding_name="click_doc_id"), his_seq_maxlen, 'mean', 'hist_len'),    
            DenseFeat('hist_len', 1),
            SparseFeat('u_city', feature_max_idx['city'], embedding_dim),
            SparseFeat('u_age', feature_max_idx['age'], embedding_dim),
            SparseFeat('u_gender', feature_max_idx['gender'], embedding_dim),
        ]
    doc_feature_columns = [
        SparseFeat('doc_id', feature_max_idx['article_id'], embedding_dim)
        # 这里后面也可以把文章的类别画像特征加入
    ]
    
    # 定义模型
    model = MIND(user_feature_columns, doc_feature_columns, num_sampled=5, user_dnn_hidden_units=(64, embedding_dim))
    
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
def get_mind_recall_res(user_embs, doc_embs, user_idx_2_rawid, doc_idx_2_rawid, topk):
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
    pickle.dump(user_recall_items_dict, open('mind_u2i_dict.pkl', 'wb'))
    
    return user_recall_items_dict