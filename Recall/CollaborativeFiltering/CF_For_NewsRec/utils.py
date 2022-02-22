import gc
import numpy as np
import pandas as pd
from tqdm import tqdm


# 减少内存的函数
def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'开始压缩内存...')
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('压缩前: {:.2f} Mb \n压缩后: {:.2f} Mb \n压缩比例: ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    
    return df


# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=100):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(50, topk+1, 50):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            if user in last_click_item_dict:
                # 获取前k个召回的结果
                tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
                if last_click_item_dict[user] in set(tmp_recall_items):
                    hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)