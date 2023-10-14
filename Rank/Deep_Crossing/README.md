# Description:
这里是DeepCrossing的demo， 基于kaggle上的一个比赛数据集, 下载链接：[http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) 数据集介绍：
这是criteo-Display Advertising Challenge比赛的部分数据集， 里面有train.csv和test.csv两个文件：
* train.csv： 训练集由Criteo 7天内的部分流量组成。每一行对应一个由Criteo提供的显示广告。为了减少数据集的大小，正(点击)和负(未点击)的例子都以不同的比例进行了抽样。示例是按时间顺序排列的
* test.csv: 测试集的计算方法与训练集相同，只是针对训练期之后一天的事件

字段说明：
* Label： 目标变量， 0表示未点击， 1表示点击
* l1-l13: 13列的数值特征， 大部分是计数特征
* C1-C26: 26列分类特征， 为了达到匿名的目的， 这些特征的值离散成了32位的数据表示

这个比赛的任务就是：开发预测广告点击率(CTR)的模型。给定一个用户和他正在访问的页面，预测他点击给定广告的概率是多少？比赛的地址链接：[https://www.kaggle.com/c/criteo-display-ad-challenge/overview](https://www.kaggle.com/c/criteo-display-ad-challenge/overview)

## 文件说明：
1. data： 这里面存放的数据文件
2. img图片
3. model/: 保存的模型
4. preprocessed_data: 数据清洗后的文件
5. DataLoadAndPreprocessing.ipynb: 这个jupyter完成了数据的导入， 清洗和保存
6. Deep_CrossingModel.ipynb: 这个jupyter完成了模型的建立， 训练和测试
