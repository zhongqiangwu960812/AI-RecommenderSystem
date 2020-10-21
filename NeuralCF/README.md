该Neural复现的Demon用的数据集是处理过的Movielens的数据集， 关于数据集的具体描述参考[https://github.com/hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)
## 关于数据格式 --- ml-1m
### rating
标签列表为：UserID::MovieID::Rating::Timestamp
* UserIDs：用户ID（1～6040）
* MovieIDs：电影ID（1～3952）
* Ratings：评分（1～5）
* Timestamp：时间戳
### user
标签列表为： UserID::Gender::Age::Occupation::Zip-code
* Gender：性别， "M"代表男， "F"代表女；
* Age：年龄，分为多个区间：1，18， 25， 35， 45， 50；
* Occupation：职业，0～20；
### movies
标签列表为：MovieID::Title::Genres
* Titles：电影名称；
* Genres：电影分类

但是， 这次实验中的数据集，已经处理过， 不是用户的评分， 而是如果评分过的标记为1， 没评分过的标记为0， 根据这个来看用户是否对于某部电影感兴趣。

## 文件说明
1. Data/: 存放数据文件， 这里面会用到rating, test.rating, test.negatives三个数据集
2. Pre_train/:  这里面存放的是GMF和MLP预训练的模型参数
3. Pretrain/: 这个文件夹这里不起作用， 这是tf里面的模型参数
4. ProcessedData: 处理后的数据， 可以直接进行读入构造DataLoader
5. img/: 模型结构
6. GMF_MLP.py: 这个是构建两个分模型， 之所以写成了.py的形式， 是因为预训练测试的时候用的了
7. GMF_Model.ipynb: GMF单模型搭建和训练过程
8. MLP_Model.ipynb: MLP单模型搭建和训练过程
9. NeuralMF_Model.ipynb: NeuralMF模型搭建和训练过程
10. 数据导入与处理.ipynb: 处理数据的
