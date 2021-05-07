from collections import namedtuple

DEFAULT_GROUP_NAME = "default_group"


# 统一输入
# SparseFeat继承了namedtuple, 通过__new__方法中设置的参数，实现对namedtuple中某些字段的初始化
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    # 它的作用是阻止在实例化类时为实例分配dict，默认情况下每个类都会有一个dict, 通过__dict__访问，这个dict维护了这个实例的所有属性
    # 当需要创建大量的实例时，创建大量的__dict__会浪费大量的内存，所以这里使用__slots__()进行限制，当然如果需要某些属性被访问到，需要
    # 在__slots__()中将对应的属性填写进去
    __slots__ = ()

    # new方法是在__init__方法之前运行的，new方法的返回值是类的实例，也就是类中的self
    # new方法中传入的参数是cls,而init的方法传入的参数是self
    # __new__ 负责对象的创建，__init__ 负责对象的初始化
    # 这里使用__new__的原因是，这里最终是想创建一个namedtuple对象，并且避免namedtuple初始化时需要填写所有参数的的情况，使用了一个类来包装
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))  # 如果没有指定embedding_dim的一个默认值，这个默认值是怎么来的？
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    # 要想使用自定义的类作为字典的键，就需要重写类的哈希函数，否则无法将其作为字典的键来使用
    # 由于这个类不需要比较大小所以不必重写__eq__()方法
    def __hash__(self):
        return self.name.__hash__()


# 数值特征，这里需要注意，数值特征不一定只是一维的，也可以是一个多维的
class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


# 长度变化的稀疏特征，其实就是id序列特征
class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    # 由于这里传进来的sparsefeat, 本身就是一个自定义的类型，也有很多有用的信息，例如name, embedding_dim等等
    # 对于VarLenSparseFeat类来说，只不过是一个sparsefeat的序列，需要获取到sparsefeat的相关属性

    # 使用@property装饰器，将一个函数的返回值作为类的属性来使用
    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()