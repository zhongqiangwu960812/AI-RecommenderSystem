# DIN

Deep Interest Network(DIIN)是2018年阿里巴巴提出来的模型， 该模型基于业务的观察，从实际应用的角度进行改进，相比于之前很多“学术风”的深度模型， 该模型更加具有业务气息。该模型的应用场景是阿里巴巴的电商广告推荐业务， 这样的场景下一般**会有大量的用户历史行为信息**， 这个其实是很关键的，因为DIN模型的创新点或者解决的问题就是使用了注意力机制来对用户的兴趣动态模拟， 而这个模拟过程存在的前提就是用户之前有大量的历史行为了，这样我们在预测某个商品广告用户是否点击的时候，就可以参考他之前购买过或者查看过的商品，这样就能猜测出用户的大致兴趣来，这样我们的推荐才能做的更加到位，所以这个模型的使用场景是**非常注重用户的历史行为特征（历史购买过的商品或者类别信息）**，也希望通过这一点，能够和前面的一些深度学习模型对比一下。

在个性化的电商广告推荐业务场景中，也正式由于用户留下了大量的历史交互行为，才更加看出了之前的深度学习模型(作者统称Embeding&MLP模型)的不足之处。如果学习了前面的各种深度学习模型，就会发现Embeding&MLP模型对于这种推荐任务一般有着差不多的固定处理套路，就是大量稀疏特征先经过embedding层， 转成低维稠密的，然后进行拼接，最后喂入到多层神经网络中去。 

这些模型在这种个性化广告点击预测任务中存在的问题就是**无法表达用户广泛的兴趣**，因为这些模型在得到各个特征的embedding之后，就蛮力拼接了，然后就各种交叉等。这时候根本没有考虑之前用户历史行为商品具体是什么，究竟用户历史行为中的哪个会对当前的点击预测带来积极的作用。 而实际上，对于用户点不点击当前的商品广告，很大程度上是依赖于他的历史行为的，王喆老师举了个例子

>假设广告中的商品是键盘， 如果用户历史点击的商品中有化妆品， 包包，衣服， 洗面奶等商品， 那么大概率上该用户可能是对键盘不感兴趣的， 而如果用户历史行为中的商品有鼠标， 电脑，iPad，手机等， 那么大概率该用户对键盘是感兴趣的， 而如果用户历史商品中有鼠标， 化妆品， T-shirt和洗面奶， 鼠标这个商品embedding对预测“键盘”广告的点击率的重要程度应该大于后面的那三个。

这里也就是说如果是之前的那些深度学习模型，是没法很好的去表达出用户这广泛多样的兴趣的，如果想表达的准确些， 那么就得加大隐向量的维度，让每个特征的信息更加丰富， 那这样带来的问题就是计算量上去了，毕竟真实情景尤其是电商广告推荐的场景，特征维度的规模是非常大的。 并且根据上面的例子， 也**并不是用户所有的历史行为特征都会对某个商品广告点击预测起到作用**。所以对于当前某个商品广告的点击预测任务，没必要考虑之前所有的用户历史行为。 

这样， DIN的动机就出来了，在业务的角度，我们应该自适应的去捕捉用户的兴趣变化，这样才能较为准确的实施广告推荐；而放到模型的角度， 我们应该**考虑到用户的历史行为商品与当前商品广告的一个关联性**，如果用户历史商品中很多与当前商品关联，那么说明该商品可能符合用户的品味，就把该广告推荐给他。而一谈到关联性的话， 我们就容易想到“注意力”的思想了， 所以为了更好的从用户的历史行为中学习到与当前商品广告的关联性，学习到用户的兴趣变化， 作者把注意力引入到了模型，设计了一个"local activation unit"结构，利用候选商品和历史问题商品之间的相关性计算出权重，这个就代表了对于当前商品广告的预测，用户历史行为的各个商品的重要程度大小， 而加入了注意力权重的深度学习网络，就是这次的主角DIN， 下面具体来看下该模型。

## 1.1 DIN模型

在具体分析DIN模型之前， 我们还得先介绍两块小内容，一个是DIN模型的数据集和特征表示， 一个是上面提到的之前深度学习模型的基线模型， 有了这两个， 再看DIN模型，就感觉是水到渠成了。

### 1.1 特征表示

工业上的CTR预测数据集一般都是`multi-group categorial form`的形式，就是类别型特征最为常见，这种数据集一般长这样：

<img src="img/1.png" style="zoom:80%;" />

这里的亮点就是框出来的那个特征，这个包含着丰富的用户兴趣信息。

对于特征编码，作者这里举了个例子：`[weekday=Friday, gender=Female, visited_cate_ids={Bag,Book}, ad_cate_id=Book]`， 这种情况我们知道一般是通过one-hot的形式对其编码， 转成系数的二值特征的形式。但是这里我们会发现一个`visted_cate_ids`， 也就是用户的历史商品列表， 对于某个用户来讲，这个值是个多值型的特征， 而且还要知道这个特征的长度不一样长，也就是用户购买的历史商品个数不一样多，这个显然。这个特征的话，我们一般是用到multi-hot编码，也就是可能不止1个1了，有哪个商品，对应位置就是1， 所以经过编码后的数据长下面这个样子：

![](img/2.png)

这个就是喂入模型的数据格式了，这里还要注意一点 就是上面的特征里面没有任何的交互组合，也就是没有做特征交叉。这个交互信息交给后面的神经网络去学习。

### 1.2 基线模型

这里的base 模型，就是上面提到过的Embedding&MLP的形式， 这个之所以要介绍，就是因为DIN网络的基准也是他，只不过在这个的基础上添加了一个新结构(注意力网络)来学习当前候选广告与用户历史行为特征的相关性，从而动态捕捉用户的兴趣。

基准模型的结构相对比较简单，我们前面也一直用这个基准， 分为三大模块：Embedding layer，Pooling & Concat layer和MLP， 结构如下:

![](img/3.png)


前面的大部分深度模型结构也是遵循着这个范式套路， 简介一下各个模块。

1. **Embedding layer**：这个层的作用是把高维稀疏的输入转成低维稠密向量， 每个离散特征下面都会对应着一个embedding词典， 维度是$D\times K$， 这里的$D$表示的是隐向量的维度， 而$K$表示的是当前离散特征的唯一取值个数,  这里为了好理解，这里举个例子说明，就比如上面的weekday特征：
	
>  假设某个用户的weekday特征就是周五，化成one-hot编码的时候，就是[0,0,0,0,1,0,0]表示，这里如果再假设隐向量维度是D， 那么这个特征对应的embedding词典是一个$D\times7$的一个矩阵(每一列代表一个embedding，7列正好7个embedding向量，对应周一到周日)，那么该用户这个one-hot向量经过embedding层之后会得到一个$D\times1$的向量，也就是周五对应的那个embedding，怎么算的，其实就是$embedding矩阵* [0,0,0,0,1,0,0]^T$ 。其实也就是直接把embedding矩阵中one-hot向量为1的那个位置的embedding向量拿出来。 这样就得到了稀疏特征的稠密向量了。

	其他离散特征也是同理，只不过上面那个multi-hot编码的那个，会得到一个embedding向量的列表，因为他开始的那个multi-hot向量不止有一个是1，这样乘以embedding矩阵，就会得到一个列表了。通过这个层，上面的输入特征都可以拿到相应的稠密embedding向量了。

2. **pooling layer and Concat layer**： pooling层的作用是将用户的历史行为embedding这个最终变成一个定长的向量，因为每个用户历史购买的商品数是不一样的， 也就是每个用户multi-hot中1个个数不一致，这样经过embedding层，得到的用户历史行为embedding的个数不一样多，也就是上面的embedding列表$t_i$不一样长， 那么这样的话，每个用户的历史行为特征拼起来就不一样长了。 而后面如果加全连接网络的话，我们知道，他需要定长的特征输入。 所以往往用一个pooling layer先把用户历史行为embedding变成固定长度(统一长度)，所以有了这个公式：
$$
  e_i=pooling(e_{i1}, e_{i2}, ...e_{ik})
$$
  这里的$e_{ij}$是用户历史行为的那些embedding。$e_i$就变成了定长的向量， 这里的$i$表示第$i$个历史特征组(是历史行为，比如历史的商品id，历史的商品类别id等)， 这里的$k$表示对应历史特种组里面用户购买过的商品数量，也就是历史embedding的数量，看上面图里面的user behaviors系列，就是那个过程了。 Concat layer层的作用就是拼接了，就是把这所有的特征embedding向量，如果再有连续特征的话也算上，从特征维度拼接整合，作为MLP的输入。

3. **MLP**：这个就是普通的全连接，用了学习特征之间的各种交互。

4. **Loss**: 由于这里是点击率预测任务， 二分类的问题，所以这里的损失函数用的负的log对数似然：
$$
L=-\frac{1}{N} \sum_{(\boldsymbol{x}, y) \in \mathcal{S}}(y \log p(\boldsymbol{x})+(1-y) \log (1-p(\boldsymbol{x})))
$$

这就是base 模型的全貌， 这里应该能看出这种模型的问题， 通过上面的图也能看出来， 用户的历史行为特征和当前的候选广告特征在全都拼起来给神经网络之前，是一点交互的过程都没有， 而拼起来之后给神经网络，虽然是有了交互了，但是原来的一些信息，比如，每个历史商品的信息会丢失了一部分，因为这个与当前候选广告商品交互的是池化后的历史特征embedding， 这个embedding是综合了所有的历史商品信息， 这个通过我们前面的分析，对于预测当前广告点击率，并不是所有历史商品都有用，综合所有的商品信息反而会增加一些噪声性的信息，可以联想上面举得那个键盘鼠标的例子，如果加上了各种洗面奶，衣服啥的反而会起到反作用。其次就是这样综合起来，已经没法再看出到底用户历史行为中的哪个商品与当前商品比较相关，也就是丢失了历史行为中各个商品对当前预测的重要性程度。最后一点就是如果所有用户浏览过的历史行为商品，最后都通过embedding和pooling转换成了固定长度的embedding，这样会限制模型学习用户的多样化兴趣。

那么改进这个问题的思路有哪些呢？  第一个就是加大embedding的维度，增加之前各个商品的表达能力，这样即使综合起来，embedding的表达能力也会加强， 能够蕴涵用户的兴趣信息，但是这个在大规模的真实推荐场景计算量超级大，不可取。 另外一个思路就是**在当前候选广告和用户的历史行为之间引入注意力的机制**，这样在预测当前广告是否点击的时候，让模型更关注于与当前广告相关的那些用户历史产品，也就是说**与当前商品更加相关的历史行为更能促进用户的点击行为**。 作者这里又举了之前的一个例子：
> 想象一下，当一个年轻母亲访问电子商务网站时，她发现展示的新手袋很可爱，就点击它。让我们来分析一下点击行为的驱动力。<br><br>展示的广告通过软搜索这位年轻母亲的历史行为，发现她最近曾浏览过类似的商品，如大手提袋和皮包，从而击中了她的相关兴趣


第二个思路就是DIN的改进之处了。DIN通过给定一个候选广告，然后去注意与该广告相关的局部兴趣的表示来模拟此过程。 DIN不会通过使用同一向量来表达所有用户的不同兴趣，而是通过考虑历史行为的相关性来自适应地计算用户兴趣的表示向量（对于给的广告）。 该表示向量随不同广告而变化。下面看一下DIN模型。

### 1.3 DIN模型架构

上面分析完了base模型的不足和改进思路之后，DIN模型的结构就呼之欲出了，首先，它依然是采用了基模型的结构，只不过是在这个的基础上加了一个注意力机制来学习用户兴趣与当前候选广告间的关联程度， 用论文里面的话是，引入了一个新的`local activation unit`， 这个东西用在了用户历史行为特征上面， **能够根据用户历史行为特征和当前广告的相关性给用户历史行为特征embedding进行加权**。我们先看一下它的结构，然后看一下这个加权公式。

<img src="img/4.png" style="zoom:80%;" />

这里改进的地方已经框出来了，这里会发现相比于base model， 这里加了一个local activation unit， 这里面是一个前馈神经网络，输入是用户历史行为商品和当前的候选商品， 输出是它俩之间的相关性， 这个相关性相当于每个历史商品的权重，把这个权重与原来的历史行为embedding相乘求和就得到了用户的兴趣表示$\boldsymbol{v}_{U}(A)$, 这个东西的计算公式如下：
$$
\boldsymbol{v}_{U}(A)=f\left(\boldsymbol{v}_{A}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \ldots, \boldsymbol{e}_{H}\right)=\sum_{j=1}^{H} a\left(\boldsymbol{e}_{j}, \boldsymbol{v}_{A}\right) \boldsymbol{e}_{j}=\sum_{j=1}^{H} \boldsymbol{w}_{j} \boldsymbol{e}_{j}
$$
这里的$\{\boldsymbol{v}_{A}, \boldsymbol{e}_{1}, \boldsymbol{e}_{2}, \ldots, \boldsymbol{e}_{H}\}$是用户$U$的历史行为特征embedding， $v_{A}$表示的是候选广告$A$的embedding向量， $a(e_j, v_A)=w_j$表示的权重或者历史行为商品与当前广告$A$的相关性程度。$a(\cdot)$表示的上面那个前馈神经网络，也就是那个所谓的注意力机制， 当然，看图里的话，输入除了历史行为向量和候选广告向量外，还加了一个它俩的外积操作，作者说这里是有利于模型相关性建模的显性知识。

这里有一点需要特别注意，就是这里的权重加和不是1， 准确的说这里不是权重， 而是直接算的相关性的那种分数作为了权重，也就是平时的那种scores(softmax之前的那个值)，这个是为了保留用户的兴趣强度。

## DIN复现代码

这里DIN数据集换了，因为上面反复强调了应用场景， 这里换成了有着丰富历史行为的亚马逊的数据集， 具体可以在[这里](http://jmcauley.ucsd.edu/data/amazon/)下载， 我们后面应该也给出了。 这个数据比较大，这里依然是进行了采样的过程。 并且这个数据还需要进行一些数据预处理的工作，所以复现这块的逻辑，主要分为数据预处理和模型训练部分。下面简单的来说一下：

关于数据预处理， 这里是按照原论文里面的方式进行处理的。 

原始数据是两个json文件(`meta_Electronics.json`和`reviews_Electronics.json`)， 一个是评论的数据文件reviews， 一个是商品的信息表。首先，我们需要对评论的这个数据json处理，把它转成pd，然后保存成`reviews.pkl`文件，这个的具体处理过程我们放在了GitHub链接DIN里面的data_preprocess.ipynb里面，这个jupyter就是整个数据预处理的过程了。 整体逻辑就是把这两个先读取，然后选择我们需要的几列数据，然后去掉reviews里面没有出现过的商品， 最后保存成pkl的文件。这是运行的第一个文件。

接下来，制作数据集， 按照论文里面描述的，我们首先是根据用户id分组，拿到每个用户购买过的商品id， 这个是正样本数据，也就是用户真正点击过的。 对于每个正样本数据，我们随机生成一个该用户没有买过的商品当做负样本数据。 所以就有了正样本列表和负样本列表。产生数据集的逻辑就是用户正样本数据的索引遍历， 把第i位置的商品当做当前候选广告， 把第i位置之前的当做历史产品序列， 这样分布产生训练集，测试集，验证集。这个从代码中看反而好看一些：

```python
train_data, val_data, test_data = [], [], []
for user_id, hist in tqdm(reviews_df.groupby('user_id')):
    pos_list = hist['item_id'].tolist()    # pos_list就是用户真实购买的商品， 下面针对每个购买的商品， 产生一个用户没有购买过的产品

    def gen_neg():
        neg = pos_list[0]
        while neg in pos_list: 
            neg = random.randint(0, item_count-1)       # 这儿产生一个不在真实用户购买的里面的
        return neg
    
    neg_list = [gen_neg() for i in range(len(pos_list))]
    hist = []   # 历史购买商品
    for i in range(1, len(pos_list)):
        hist.append([pos_list[i-1]])
        if i == len(pos_list) - 1:                   # 最后一个的时候
            test_data.append([hist, [pos_list[i]], 1])
            test_data.append([hist, [neg_list[i]], 0])
        elif i == len(pos_list) - 2:           # 倒数第二个的时候
            val_data.append([hist, [pos_list[i]], 1])
            val_data.append([hist, [neg_list[i]], 0])
        else:
            train_data.append([hist, [pos_list[i]], 1])
            train_data.append([hist, [neg_list[i]], 0])
```

上面这个可以举一个例子，比如某个用户1买的商品有`[10, 8, 5, 4, 9]`， 那么这个算作正样本列表，也就是真正点击过的商品，label会是1， 根据这个列表，生成一个负列表，label都是0， 比如`[11, 17, 12, 14, 13]`， 那么在构建数据集的时候， 可以构造:
> hist_id, target_item , label<br>
> [[10]], [8], 1
> [[10],[8]], [5], 1
> [[10], [8], [5]], [4], 1
> [[10], [8], [5], [4]], [9], 1
> [[10]], [17], 0
> [[10],[8]], [12], 0
> [[10], [8], [5]], [14], 0
> [[10], [8], [5], [4]], [13], 0 <br>
> 只不过后面这两个长的，分别给了验证集和测试集

这样制作完了的数据集。这个代码是data_create.py里面的制作亚马逊数据集里面的核心代码，主要是明白这里的制作数据集的思路。

有了数据集，就可以看模型代码了，这个是DIN.py。模型主要两个大部分， Attention层和DIN全貌。

### Attention层

这里首先是Attention层， 就是图里面那个局部激活单元，这个是一个全连接的神经网络，接收的输入是4部分[item_embed, seq_embed, seq_embed, mask]
* item_embed: 这个是候选商品的embedding向量， 维度是`(None, embedding_dim * behavior_num)`， 由于这里能表示的用户行为特征个数，behavior_num能表示用户行为的特征个数 这里是1
* seq_embed: 这个是用户历史商品序列的embedding向量， 维度是`(None, max_len, embedding_dim * behavior_num)`
* mask:  维度是`(None, max_len)`   这个里面每一行是`[False, False, True, True, ....]`的形式， False的长度表示样本填充的那部分, 填充为0的那些得标识出来，后面计算的时候，填充的那些去掉

```python
class Attention_layer(Layer):
    """
    自定义Attention层， 这个就是一个全连接神经网络
    """
    def __init__(self, att_hidden_units, activation='sigmoid'):
        super(Attention_layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)
    
    # forward
    def call(self, inputs):
        """
        这里的inputs包含四部分： [item_embed, seq_embed, seq_embed, mask]
        
        item_embed: 这个是候选商品的embedding向量   维度是(None, embedding_dim * behavior_num)   # behavior_num能表示用户行为的特征个数 这里是1， 所以(None, embed_dim)
        seq_embed: 这个是用户历史商品序列的embedding向量， 维度是(None, max_len, embedding_dim * behavior_num)  (None, max_len, embed_dim)
        mask:  维度是(None, max_len)   这个里面每一行是[False, False, True, True, ....]的形式， False的长度表示样本填充的那部分
        """
        q, k, v, key_masks = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])   # (None, max_len*embedding)       # 沿着k.shap[1]的维度复制  毕竟每个历史行为都要和当前的商品计算相似关系
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])      # (None, max_len, emebdding_dim
        
        # q, k, out product should concat
        info = tf.concat([q, k, q-k, q*k], axis=-1)   # (None, max_len, 4*emebdding_dim)
        
        # n层全连接
        for dense in self.att_dense:
            info = dense(info)
        
        outputs = self.att_final_dense(info)      # (None,  max_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)    # (None, max_len)
        
        # mask 把每个行为序列填充的那部分替换成很小的一个值
        paddings = tf.ones_like(outputs) * (-2**32+1)      # (None, max_len)  这个就是之前填充的那个地方， 我们补一个很小的值
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        
        # softmax
        outputs = tf.nn.softmax(logits=outputs) # (None, max_len)
        outputs = tf.expand_dims(outputs, axis=1)   # (None, 1, max_len) 
        
        outputs = tf.matmul(outputs, v)   # 三维矩阵相乘， 相乘发生在后两维   (None, 1, max_len) * (None, max_len, embed_dim) = (None, 1, embed_dim)
        outputs = tf.squeeze(outputs, axis=1)  # (None, embed_dim)
        
        return outputs
```

这一块完成的DIN结构图里面绿框里面的整个操作过程。 有两点需要注意，第一个是这里的Dense的激活函数用的sigmoid，没有Prelu或者Dice， 这里是看的大部分网上代码里面这块没有用， 把这两个激活函数用到了后面的MLP里面了。 第二个就是论文里面虽然说不用softmax， 但是这里用了softmax了， 这个看具体的应用场景吧还是。

### 2. Dice激活函数

这是作者那个训练技术上的第二个创新，改得prelu激活函数，自己提出了dice激活函数， 下面我们看看这个函数的具体实现过程：

```python
class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')
    
    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        
        return self.alpha * (1.0-x_p) * x + x_p * x
```

### 3. DIN模型

有了前面的两套， 实现DIN模型就相对简单了，因为我们说DIN模型的原始架构是base model， 然后再这个基础上加了上面的两个新结构模块，所以这里相当于是先有一个base model，然后再前向传播的时候，加入那两个结构就OK了，具体看一下：

```python
class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40), ffn_hidden_units=(80, 40), att_activation='sigmoid', 
                 ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_reg=1e-4):
        """
        DIN:
        feature_columns:列表, [dense_feature_columns,sparse_feature_columns],dense_feature_columns是[{'feat': 'feat_name'}], 而sparse_feature_columns是[{'feat':'feat_name', 'feat_num': 'nunique', 'embed_dim'}]
        behavior_feature_list: 列表. 能表示用户历史行为的特征, 比如商品id， 店铺id ['item', 'cat']
        att_hidden_units: 注意力层的隐藏单元个数.可以是一个列表或者元组，毕竟注意力层也是一个全连接的网络嘛
        ffn_hidden_units:全连接层的隐藏单元个数和层数，可以是一个列表或者元组  (80, 40)  就表示两层神经网络， 第一层隐藏层单元80个， 第二层40个
        att_activation: 激活单元的名称， 字符串
        ffn_activation: 激活单元的名称， 用'prelu'或者'Dice'  
        maxlen: 标量. 用户历史行为序列的最大长度
        dropout: 标量，失活率
        embed_reg: 标量. 正则系数
        """
        super(DIN, self).__init__()      # 初始化网络
        self.maxlen = maxlen
        
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns           # 这里把连续特征和离散特征分别取出来， 因为后面两者的处理方式不同
        
        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)      # 这个other_sparse就是离散特征中去掉了能表示用户行为的特征列
        self.dense_len = len(self.dense_feature_columns)    
        self.behavior_num = len(behavior_feature_list)
        
        # embedding层， 这里分为两部分的embedding， 第一部分是普通的离散特征， 第二部分是能表示用户历史行为的离散特征， 这一块后面要进注意力和当前的商品计算相关性
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'], 
                                              input_length=1, 
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg)
                                             ) for feat in self.sparse_feature_columns if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and catetory id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'], 
                                           input_length=1, 
                                           output_dim=feat['embed_dim'], 
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg)
                                          ) for feat in self.sparse_feature_columns if feat['feat'] in behavior_feature_list]
        
        # 注意力机制
        self.attention_layer = Attention_layer(att_hidden_units, att_activation)
        
        self.bn = BatchNormalization(trainable=True)
        
        # 全连接网络
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation=='prelu' else Dice()) for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)
        
    def call(self, inputs):
        """
        inputs: [dense_input, sparse_input, seq_input, item_input]  ， 第二部分是离散型的特征输入， 第三部分是用户的历史行为， 第四部分是当前商品的输入
    
        dense_input： 连续型的特征输入， 维度是(None, dense_len)
        sparse_input: 离散型的特征输入， 维度是(None, other_sparse_len)
        seq_inputs: 用户的历史行为序列(None, maxlen, behavior_len)
        item_inputs： 当前的候选商品序列 (None, behavior_len)
        """
        
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        
        # attention --->mask, if the element of seq_inputs is equal 0, it must be filled in  这是因为很多序列由于不够长用0进行了填充,并且是前面补的0
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)  类型转换函数， 把seq_input中不等于0的值转成float32
        # 这个函数的作用就是每一行样本中， 不为0的值返回1， 为0的值返回0， 这样就把填充的那部分值都给标记了出来
        
        # 下面把连续型特征和行为无关的离散型特征拼到一块先
        other_info = dense_inputs   # (None, dense_len)
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)      # (None, dense_len+other_sparse_len)
        
        # 下面把候选的商品和用户历史行为商品也各自的拼接起来
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_num)], axis=-1)   # [None, max_len, embed_dim]
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.behavior_num)], axis=-1)  # [None, embed_dim]
        
    
        # 下面进行attention_layer的计算
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed, mask])   # (None, embed_dim) 
        
        # 所有特征拼起来了
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)   # (None, dense_len + other_sparse_len + embed_dim+embed_dim)  
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1) # (None, embed_dim+embed_dim)
        
        info_all = self.bn(info_all)
        
        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)
        
        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all))
        return outputs
```

依然是每一行我都加了注释了， 简单的提几个细节就可以啦，第一个细节就是我们的输入分为四类， 连续特征， 普通离散特征， 用户历史行为的离散特征，和候选商品特征。这四类处理的方式不一样。对于连续特征和普通离散特征来说，这两块这里就直接普通离散经过embedding，和连续特征拼接到了一块，先不管。 而候选商品特征和用户历史行为特征要经过一个Attention layer之后， 把得到的embedding和上面的那两个拼到一块作为了神经网络的输入。第二个细节，就是把Dice和prelu的激活用到了后面的全连接神经网络中。

下面我们看一下这个模型建立的时候，构建的格式(要符合这个格式才行):

```python
def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}
                      ]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DIN(features, behavior_list)
```
这个dense_features是连续特征， sparse_features是稀疏的特征， 都是列表，然后里面是那样字典的格式，这个具体细节看GitHub里面的代码吧。 这里还需要一个behavior_list， 也就是用户的历史行为特征， 这个要单独传入进去。

按照这样的方式，就可以建立DIN模型， 而上面已经把trainx, trainy处理成对应的数据格式了， 这样就能看训练DIN了。 可以到后面的GitHub看具体细节。