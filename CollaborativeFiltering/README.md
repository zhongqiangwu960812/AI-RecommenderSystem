# CollaborativeFiltering(协同过滤算法)

协同过滤(Collaborative Filtering)算法， 基本思想是根据用户之前的喜好以及其他兴趣相近的用户的选择来给用户推荐物品(基于对用户历史行为数据的挖掘发现用户的喜好偏向， 并预测用户可能喜好的产品进行推荐)， 一般是仅仅基于用户的行为数据（评价、购买、下载等）, 而不依赖于项的任何附加信息（物品自身特征）或者用户的任何附加信息（年龄， 性别等）。目前应用比较广泛的协同过滤算法是基于邻域的方法， 而这种方法主要有下面两种算法：

* 基于用户的协同过滤算法(UserCF): 给用户推荐和他兴趣相似的其他用户喜欢的产品
* 基于物品的协同过滤算法(ItemCF): 给用户推荐和他之前喜欢的物品相似的物品

关于理论的详细介绍， 可以参考博客：[AI上推荐 之 协同过滤](https://blog.csdn.net/wuzhongqiang/article/details/107891787), 这里是UserCF和ItemCF的代码实践部分， 这个文件夹主要是把项亮《推荐系统实践》里面的协同过滤算法(UserCF和ItemCF)实现了一遍， 并做了详细的注释。


# 任务描述：

我们实现一下《推荐系统实践》里面的ItemCF和UserCF算法， 采用的数据集是GroupLens提供的MovieLens的其中一个小数据集ml-latest-small。 该数据及包含700个用户对带有6100个标签的10000部电影的100000条评分。 该数据集是一个评分数据集， 用户可以给电影评5个不同等级的分数(1-5)， 而由于我们主要是研究隐反馈数据中的topN推荐问题， 所以忽略了数据集中的评分记录。 **TopN推荐的任务是预测用户会不会对某部电影评分， 而不是预测用户在准备对某部电影评分的前提下给电影评多少分**。

## 数据集描述
该实验使用的数据集来自:[http://grouplens.org/datasets/movielens/](http://grouplens.org/datasets/movielens/)<br>
数据集简介如下：

* MovieLens 100K Dataset<br>
Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.
* MovieLens 1M Dataset<br>
Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
* MovieLens 10M Dataset<br>
Stable benchmark dataset. 10 million ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users. Released 1/2009.
* MovieLens 20M Dataset<br>
Stable benchmark dataset. 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Released 4/2015.
* MovieLens Latest Datasets<br>
  * **Small: 100,000 ratings and 6,100 tag applications applied to 10,000 movies by 700 users. Last updated 1/2016.**
  * Full: 22,000,000 ratings and 580,000 tag applications applied to 33,000 movies by 240,000 users. Last updated 1/2016.
* MovieLens Tag Genome Dataset<br>
11 million computed tag-movie relevance scores from a pool of 1,100 tags applied to 10,000 movies.

本次实验， 使用的是ml-latest-small。

## 代码的行文逻辑
不管是UserCF还是ItemCF， 行文逻辑都是下面的四个步骤：
1. 导入数据， 读取文件得到"用户-电影"的评分数据， 并且分为训练集和测试集
2. 计算用户(userCF)或者电影(itemcf)之间的相似度
3. 针对目标用户u， 找到其最相似的k个用户/产品， 产生N个推荐
4. 产生推荐之后， 通过准确率、召回率和覆盖率等进行评估。

编码小技巧： 倒排表和字典存储， 由于这种推荐数据非常稀疏， 所以采用了倒排表和字典存储的方式减少时间和空间复杂度，具体的可以参考实际代码。

## 结果的评估
由于UserCF和ItemCF结果评估部分是共性知识点， 所以在这里统一标识。 这里介绍评测指标：

1. 召回率<br>
对用户u推荐N个物品记为$R(u)$, 令用户u在测试集上喜欢的物品集合为$T(u)$， 那么召回率定义为：<br>
$$\operatorname{Recall}=\frac{\sum_{u}|R(u) \cap T(u)|}{\sum_{u}|T(u)|}$$<br>
这个意思就是说， 在用户真实购买或者看过的影片里面， 我模型真正预测出了多少， 这个考察的是模型推荐的一个全面性。 <br>

2. 准确率<br>
准确率定义为：<br>
$$\operatorname{Precision}=\frac{\sum_{u} \mid R(u) \cap T(u)}{\sum_{u}|R(u)|}$$<br>
这个意思再说， 在我推荐的所有物品中， 用户真正看的有多少， 这个考察的是我模型推荐的一个准确性。 <br><br>
为了提高准确率， 模型需要把非常有把握的才对用户进行推荐， 所以这时候就减少了推荐的数量， 而这往往就损失了全面性， 真正预测出来的会非常少，所以实际应用中应该综合考虑两者的平衡。

3. 覆盖率
覆盖率反映了推荐算法发掘长尾的能力， 覆盖率越高， 说明推荐算法越能将长尾中的物品推荐给用户。
<br>$$\text { Coverage }=\frac{\left|\bigcup_{u \in U} R(u)\right|}{|I|}$$<br>
该覆盖率表示最终的推荐列表中包含多大比例的物品。如果所有物品都被给推荐给至少一个用户， 那么覆盖率是100%。

4. 新颖度
用推荐列表中物品的平均流行度度量推荐结果的新颖度。 如果推荐出的物品都很热门， 说明推荐的新颖度较低。  由于物品的流行度分布呈长尾分布， 所以为了流行度的平均值更加稳定， 在计算平均流行度时对每个物品的流行度取对数。


# 文件说明：
1. ml-latest-small/:  这里面是电影评分数据集
2. images/: 存放着ipynb里面需要的图片
3. ItemCF.ipynb, UserCF.ipynb:  非常详细解说ItemCF, UserCF算法的实现
4. ItemCF.py, UserCF.py: 把上面的代码封装成了一个类的形式
5. RecommendExample_GuessScore.ipynb:  博客链接里面猜测用户打分的代码例子， 具体可以参考上面的博客链接







