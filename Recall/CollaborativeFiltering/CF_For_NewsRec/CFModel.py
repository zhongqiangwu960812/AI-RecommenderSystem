import numpy as np 
import pandas as pd
from tqdm import tqdm

#from sklearn.uitls import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class UserCF(object):
    def __init__(self, user_click_hist_df, sim_corr_rule=False):
        self.user_click_hist_df = user_click_hist_df
        # 相似性关联规则是否使用，如果用的话，计算用户相似度的时候可以使用简单关联规则
        self.sim_corr_rule = sim_corr_rule
        # 用户活跃度字典
        self.user_activate_degree_dict = self.get_user_activate_degree_dict()
        # 用户相似性矩阵
        self.user_sim_matrix = dict()
        
        # 获取倒排表
        # 每个用户， 获取点击过的文章以及观看时长 {user1: [(item1, duration1), (item2, duration2)..]...}
        self.article_user_duration_dict = self.get_article_user_duration_dict()
        self.user_article_duration_dict = self.get_user_article_duration_dict()
    
    # 计算用户相似度的时候，可以使用简单关联规则，比如用户活跃度， 这里将用户点击次数作为用户活跃度指标
    def get_user_activate_degree_dict(self):
        user_activate_df = self.user_click_hist_df.groupby('user_id')['article_id'].count().reset_index()
        
        # 归一化
        mm = MinMaxScaler()
        user_activate_df['article_id'] = mm.fit_transform(user_activate_df[['article_id']])
        user_activate_degree_dict = dict(zip(user_activate_df['user_id'], user_activate_df['article_id']))
        return user_activate_degree_dict
    
    # 获取文章-用户-观看时长字典，下面计算用户相似矩阵的时候会使用，算是一种倒排
    def get_article_user_duration_dict(self):
        def make_user_duration_pair(df):
            return list(zip(df['user_id'], df['duration']))
        article_user_duration_df = self.user_click_hist_df.groupby('article_id')['user_id', 'duration'] \
                                   .apply(lambda x: make_user_duration_pair(x)) \
                                   .reset_index().rename(columns={0:'user_duration_list'})
        article_user_duration_dict = dict(zip(article_user_duration_df['article_id'], article_user_duration_df['user_duration_list']))
        return article_user_duration_dict
    
    # 获取用户-文章-观看时长字典， 用户推荐的时候会用到
    def get_user_article_duration_dict(self):
        def make_user_duration_pair(df):
            return list(zip(df['article_id'], df['duration']))
        user_article_duration_df = self.user_click_hist_df.groupby('user_id')['article_id', 'duration'] \
                                   .apply(lambda x: make_user_duration_pair(x)) \
                                   .reset_index().rename(columns={0:'user_duration_list'})
        user_article_duration_dict = dict(zip(user_article_duration_df['user_id'], user_article_duration_df['user_duration_list']))
        return user_article_duration_dict
    
    # 根据用户点击计算相似性矩阵
    def usercf_similar_matrix(self):
        
        # 遍历倒排字典， 根据用户行为计算相似权重
        user_interact_cnt = collections.defaultdict(int)
        for item, user_duration_list in tqdm(self.article_user_duration_dict.items()):
            # 遍历看过该item的用户
            for u, u_duration in user_duration_list:
                user_interact_cnt[u] += 1
                self.user_sim_matrix.setdefault(u, {})
                # 同时看过该item的两个用户
                for v, v_duration in user_duration_list:
                    if u == v:
                        continue
                    
                    self.user_sim_matrix[u].setdefault(v, 0)
                    # 如果使用关联规则， 加额外的一些权重，比如用户活跃度，用户观看时长等
                    # 这里的式子都是自己定义的，可以根据实际场景修改
                    if self.sim_corr_rule:
                        activate_weight = 100 * 0.5 * (self.user_activate_degree_dict[u] + self.user_activate_degree_dict[v])   
                        duration_weight = np.log(np.abs(u_duration - v_duration))
                        self.user_sim_matrix[u][v] += activate_weight*duration_weight / math.log(len(user_duration_list) + 1)
                    else:
                         self.user_sim_matrix[u][v] += 1 / math.log(len(user_duration_list) + 1)
        # 权重归一化
        for u, related_users in self.user_sim_matrix.items():
            for v, wij in related_users.items():
                 self.user_sim_matrix[u][v] = wij / math.sqrt(user_interact_cnt[u] * user_interact_cnt[v])
        
        # 保存到本地
        pickle.dump(self.user_sim_matrix, open('usercf_u2u_sim.pkl', 'wb'))
        
    def single_user_rec(self, user_id, sim_user_topk, recall_article_num):
        # 用户历史交互
        user_item_time_list = self.user_article_duration_dict[user_id]
        user_hist_items =set([item[0] for item in user_item_time_list])
         
        article_rank = collections.defaultdict(int)
        for sim_u, wuv in sorted(self.user_sim_matrix[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
            for i, i_duration in self.user_article_duration_dict[sim_u]:
                # 用户之前看过了，跳过去
                if i in user_hist_items:
                    continue
                article_rank.setdefault(i, 0.0)
                
                # 这里可以使用与用户历史点击过的文章的关联规则对当前文章加权计算
                # 比如创建时间， 点击的次序等，由于目前文章画像没有弄过来
                # 这里只加入点击的相对次序， 观看时间
                loc_weight, dur_weight = 1.0, 1.0
                for loc, (j, j_duration) in enumerate(user_item_time_list):
                    loc_weight += 0.9 ** (len(user_item_time_list) - loc)  # 越之前的权重越小
                    dur_weight += np.log(np.abs(i_duration - j_duration))
                    # 创建文章的时间差权重, 这个先不用了
                article_rank[i] += loc_weight * dur_weight
        # 召回数量不够
        if len(article_rank) < recall_article_num:
            print("召回数量不足, 这里随机从看过的文章里面选....")
            articles = set(list(self.user_click_hist_df['article_id'])) - user_hist_items
            random.shuffle(list(articles))
            for i, item in enumerate(articles):
                if item in article_rank:
                    continue
                article_rank[item] = -i - 100  # 随便给负数
                if len(article_rank) == recall_article_num:
                    break
        article_rank = sorted(article_rank.items(), key=lambda x: x[1], reverse=True)[:recall_article_num] 
        return article_rank
              
    def usercf_recommend(self, sim_user_topk=50, recall_arcicle_num=200):
        user_recall_items_dict = collections.defaultdict(dict)
        
        print("计算用户相似性矩阵.....")
        if os.path.exists('usercf_u2u_sim.pkl'):
            self.user_sim_matrix =  pickle.load(open('usercf_u2u_sim.pkl', 'rb'))
        else:
            self.usercf_similar_matrix()
        
        print("用户协同过滤召回....")
        for user in tqdm(self.user_click_hist_df['user_id'].unique()):
            user_recall_items_dict[user] = self.single_user_rec(user, sim_user_topk, recall_arcicle_num)    
                
        pickle.dump(user_recall_items_dict, open('usercf_recall.pkl', 'wb'))
        
        return user_recall_items_dict


class ItemCF(object):
    def __init__(self, user_click_hist_df, sim_corr_rule=False):
        self.user_click_hist_df = user_click_hist_df
        # 相似性关联规则，如果使用，计算物品相似度的时候，考虑一些其他关联规则
        self.sim_corr_rule = sim_corr_rule
        # 文章相似性矩阵
        self.doc_sim_matrix = dict()
        
        # 倒排表
        self.article_user_duration_dict = self.get_article_user_duration_dict()
        self.user_article_duration_dict = self.get_user_article_duration_dict()
    
    # 文章 - 用户 - 点击时长字典， 根据物品相似度产生推荐的时候会用到
    def get_article_user_duration_dict(self):
        def make_user_duration_pair(df):
            return list(zip(df['user_id'], df['duration']))
        article_user_duration_df = self.user_click_hist_df.groupby('article_id')['user_id', 'duration'] \
                                   .apply(lambda x: make_user_duration_pair(x)) \
                                   .reset_index().rename(columns={0:'user_duration_list'})
        article_user_duration_dict = dict(zip(article_user_duration_df['article_id'], article_user_duration_df['user_duration_list']))
        return article_user_duration_dict
    
    # 用户 - 文章 - 点击时长字典， 倒排，计算物品相似度
    def get_user_article_duration_dict(self):
        def make_user_duration_pair(df):
            return list(zip(df['article_id'], df['duration']))
        user_article_duration_df = self.user_click_hist_df.groupby('user_id')['article_id', 'duration'] \
                                   .apply(lambda x: make_user_duration_pair(x)) \
                                   .reset_index().rename(columns={0:'user_duration_list'})
        user_article_duration_dict = dict(zip(user_article_duration_df['user_id'], user_article_duration_df['user_duration_list']))
        return user_article_duration_dict
    
    # 根据文章被点击情况计算相似性矩阵 
    def itemcf_similar_matrix(self):
        # 遍历用户-文章-点击时长字典， 同一用户点击不同物品， 统计共现频次，当然，可以加入一些其他信息
        item_interact_cnt = collections.defaultdict(int)
        for user, item_duration_list in tqdm(self.user_article_duration_dict.items()):
            # 在基于商品的协同过滤优化的时候，可以考虑时间因素
            for i_loc, (i, i_click_duration) in enumerate(item_duration_list):
                item_interact_cnt[i] += 1
                self.doc_sim_matrix.setdefault(i, {})
                # 同时被这个用户看过的doc
                for j_loc, (j, j_click_duration) in enumerate(item_duration_list):
                    if i == j:
                        continue
                    self.doc_sim_matrix[i].setdefault(j, 0)
                    # 如果这里用关联规则， 需要额外加一些权重，比如相对位置， 文章的画像属性等
                    if self.sim_corr_rule:
                        # 考虑文章的正向顺序点击和反向顺序点击
                        loc_alpha = 1.0 if j_loc > i_loc else 0.7
                        # 位置信息权重
                        loc_weight = loc_alpha * (0.9 ** (np.abs(j_loc-i_loc)-1))
                        # 文章的创建时间权重，由于我数据没拼过文章画像来，这个先不做，感兴趣的可以试试
                        # 用户观看时长权重
                        duration_weight = np.log(np.abs(j_click_duration - i_click_duration))
                        self.doc_sim_matrix[i][j] += loc_weight * duration_weight / math.log(len(item_duration_list)+1)
                    else:
                        self.doc_sim_matrix[i][j] += 1 / math.log(len(item_duration_list)+1)
        # 权重归一化
        for i, related_items in self.doc_sim_matrix.items():
            for j, wij in related_items.items():
                self.doc_sim_matrix[i][j] = wij / math.sqrt(item_interact_cnt[i] * item_interact_cnt[j])
        
        # 保存到本地
        pickle.dump(self.doc_sim_matrix, open('itemcf_i2i_sim.pkl', 'wb'))
    
    # 为单个用户产生推荐
    def single_user_rec(self, user_id, sim_item_topk, recall_article_num):
        # 用户点击过的文章
        user_item_time_list = self.user_article_duration_dict[user_id]
        user_hist_items = set([item[0] for item in user_item_time_list])
        
        article_rank = collections.defaultdict(int)
        for loc, (i, duration) in enumerate(user_item_time_list):
            for j, wij in sorted(self.doc_sim_matrix[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items:
                    continue
                # 考虑相似文章和历史点击文章所在位置权重
                loc_weight = (0.9 ** (len(user_item_time_list) - loc))
                # 这里还可以考虑其他权重，比如文章创建时间， 内容差别等
                article_rank.setdefault(j, 0)
                article_rank[j] += loc_weight * wij
        
        # 召回数量不够
        if len(article_rank) < recall_article_num:
            print("召回数量不足, 这里随机从看过的文章里面选....")
            articles = set(list(self.user_click_hist_df['article_id'])) - user_hist_items
            random.shuffle(list(articles))
            for i, item in enumerate(articles):
                if item in article_rank:
                    continue
                article_rank[item] = -i - 100  # 随便给负数
                if len(article_rank) == recall_article_num:
                    break
        article_rank = sorted(article_rank.items(), key=lambda x: x[1], reverse=True)[:recall_article_num] 
        return article_rank
    
    def itemcf_recommend(self, sim_item_topk=100, recall_article_num=200):
        user_recall_items_dict = collections.defaultdict(dict)
        
        print("计算文章相似性矩阵.....")
        if os.path.exists('itemcf_i2i_sim.pkl'):
            self.doc_sim_matrix =  pickle.load(open('itemcf_i2i_sim.pkl', 'rb'))
        else:
            self.itemcf_similar_matrix()
        
        print("文章协同过滤召回....")
        for user in tqdm(self.user_click_hist_df['user_id'].unique()):
            user_recall_items_dict[user] = self.single_user_rec(user, sim_item_topk, recall_article_num)    
                
        pickle.dump(user_recall_items_dict, open('itemcf_recall.pkl', 'wb'))
        
        return user_recall_items_dict