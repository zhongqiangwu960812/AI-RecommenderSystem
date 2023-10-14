## Description：

这里主要是实现了协同过滤算法，并将其应用到了真实的新闻推荐召回数据集上，完成召回任务。

* main.ipynb: 代码的主程序入口
* CFModel.py: 这里面是基于用户的协同过滤和基于文章的协同过滤算法
* utils.py: 一些工具函数

实验中，采用的数据集是采样过后的新闻推荐数据集。

实验分别采用了基于用户的协同过滤算法以及优化版本， 基于文章的协同过滤算法及优化版本。

评估方法是HR比例， 结果如下：

* 基于用户的协同过滤：

  * 原始： 

  * ```
     topk:  50  :  hit_num:  2033 hit_rate:  0.10165 user_num :  20000
     topk:  100  :  hit_num:  3033 hit_rate:  0.15165 user_num :  20000
     topk:  150  :  hit_num:  3657 hit_rate:  0.18285 user_num :  20000
     topk:  200  :  hit_num:  4121 hit_rate:  0.20605 user_num :  20000
    ```

  * 关联规则优化

  * ```
     topk:  50  :  hit_num:  2022 hit_rate:  0.1011 user_num :  20000
     topk:  100  :  hit_num:  3016 hit_rate:  0.1508 user_num :  20000
     topk:  150  :  hit_num:  3763 hit_rate:  0.18815 user_num :  20000
     topk:  200  :  hit_num:  4339 hit_rate:  0.21695 user_num :  20000
    ```

* 基于文章的协同过滤：

  * 原始：

  * ```
     topk:  50  :  hit_num:  2055 hit_rate:  0.10275 user_num :  20000
     topk:  100  :  hit_num:  2956 hit_rate:  0.1478 user_num :  20000
     topk:  150  :  hit_num:  3555 hit_rate:  0.17775 user_num :  20000
     topk:  200  :  hit_num:  4040 hit_rate:  0.202 user_num :  20000
    ```

  * 优化：

  * ```
     topk:  50  :  hit_num:  2185 hit_rate:  0.10925 user_num :  20000
     topk:  100  :  hit_num:  3142 hit_rate:  0.1571 user_num :  20000
     topk:  150  :  hit_num:  3828 hit_rate:  0.1914 user_num :  20000
     topk:  200  :  hit_num:  4347 hit_rate:  0.21735 user_num :  20000
    ```

从实验结果上来看，采用关联规则的ItemCF的效果提升比较明显。  协同过滤算法要要比YouTubeDNN的效果好很多。 