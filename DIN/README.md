## 文件说明

DIN这块，目前我已经写了两个个版本的模型了， 都整理到这里。

* DeepCTRStyle: 这个是学习的deepctrstyle风格，把各个模块都单独的写好，最后直接拼接成模型的方法。
* TraditionalPytorchStyle： 这个是Pytorch的函数式风格， 把模型写成类的方式，然后在里面写前向传播的逻辑