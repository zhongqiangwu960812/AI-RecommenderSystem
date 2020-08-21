AI-RecommenderSystem
该仓库尝试整理推荐系统领域的一些经典算法模型，主要包括传统的推荐算法模型和深度学习模型， 并尝试用浅显易懂的语言把每个模型或者算法解释清楚！此次整理依然是通过CSDN博客+GitHub的形式进行输出， CSDN主要整理算法的原理或者是经典paper的解读， 而GitHub上主要是模型的复现和实践。

# 1. 传统的推荐算法模型

![传统推荐模型演化关系图](imgs/传统推荐模型演化关系图.png)

## [1.1 协同过滤算法](https://github.com/zhongqiangwu960812/AI-RecommenderSystem/tree/master/CollaborativeFiltering)
协同过滤算法， 虽然这个离我们比较久远， 但它是对业界影响力最大， 应用最广泛的一种经典模型， 从1992年一直延续至今， 尽管现在协同过滤差不多都已经融入到了深度学习，但模型的基本原理依然还是基于经典协同过滤的思路， 或者是在协同过滤的基础上进行演化， 所以这个算法模型依然有种“宝刀未老”的感觉， 掌握和学习非常有必要。<br><br>所谓协同过滤(Collaborative Filtering)算法， 基本思想是**根据用户之前的喜好以及其他兴趣相近的用户的选择来给用户推荐物品**(基于对用户历史行为数据的挖掘发现用户的喜好偏向， 并预测用户可能喜好的产品进行推荐)， **一般是仅仅基于用户的行为数据（评价、购买、下载等）, 而不依赖于项的任何附加信息（物品自身特征）或者用户的任何附加信息（年龄， 性别等）**。目前应用比较广泛的协同过滤算法是基于邻域的方法， 而这种方法主要有下面两种算法：
* **基于用户的协同过滤算法(UserCF)**: 给用户推荐和他兴趣相似的其他用户喜欢的产品
* **基于物品的协同过滤算法(ItemCF)**: 给用户推荐和他之前喜欢的物品相似的物品

所以这一块知识的主要内容是UserCF和ItemCF的工作原理和代码实践， 工作原理部分详情见我写的博客[AI上推荐 之 协同过滤](https://blog.csdn.net/wuzhongqiang/article/details/107891787), 代码实践是把项亮推荐系统实践里面的协同过滤算法(UserCF和ItemCF)实现了一遍， 并进行详细的解释和说明。

# 2. 深度学习的浪潮之巅
![](imgs/深度学习模型演化关系图.png)

# 参考：
* 项亮-《推荐系统实践》
* 王喆-《深度学习推荐系统》
* [https://github.com/BlackSpaceGZY/Recommender-System-with-TF2.0](https://github.com/BlackSpaceGZY/Recommender-System-with-TF2.0)
* [https://github.com/shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR)
