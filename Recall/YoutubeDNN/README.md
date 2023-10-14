## Description

这个是YouTubeDNN进行召回的代码，主要包括下面两个文件：

*  utils.py里面是各个工具函数，包括生成数据集， 生成模型数据格式，训练模型函数， 获得用户embedding和item embedding的函数，以及最近邻搜索的函数等。
* YouTubeDNN召回.ipynb： 文件是主文件，主要包括读入数据集，调用上面各个函数完成召回过程，以及生成最后结果

代码逻辑其实比较简单， 详细内容和说明，可以见博客