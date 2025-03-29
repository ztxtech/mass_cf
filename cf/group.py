import math

from cf.distribution import distribution, vec2mass
from cf.moment import calculate_high_order_moments_from_mgf, calculate_high_order_moments_from_cf


class FMassDistribution:
    """
    FMassDistribution 类用于管理和计算基于生成函数的质量分布及其高阶矩。

    属性:
        generation_function (callable): 用于生成分布的函数。
        ignore_zero (bool): 是否忽略零元素。
        cache (dict): 缓存已计算的分布和质量，避免重复计算。
    """
    def __init__(self, generation_function, ignore_zero=True):
        """
        初始化 FMassDistribution 类的实例。

        参数:
            generation_function (callable): 用于生成分布的函数。
            ignore_zero (bool, optional): 是否忽略零元素。默认为 True。
        """
        self.generation_function = generation_function
        self.ignore_zero = ignore_zero
        self.cache = {}

    def set_cardinality(self, cardinality):
        """
        根据给定的基数设置并缓存分布向量和质量分布。

        参数:
            cardinality (int): 分布的基数。
        """
        if cardinality in self.cache:
            # 如果基数已在缓存中，直接返回
            return
        else:
            # 若不在缓存中，初始化该基数的缓存项
            self.cache[cardinality] = {}
            # 计算分布向量
            self.cache[cardinality]['vector'] = distribution(cardinality, self.generation_function, self.ignore_zero)
            # 将分布向量转换为质量分布
            self.cache[cardinality]['mass'] = vec2mass(self.cache[cardinality]['vector'])

    def sampling(self, cardinality):
        """
        根据给定的基数进行采样，返回分布向量和质量分布。

        参数:
            cardinality (int): 分布的基数。

        返回:
            tuple: 包含分布向量和质量分布的元组。
        """
        self.set_cardinality(cardinality)
        return self.cache[cardinality]['vector'], self.cache[cardinality]['mass']

    def high_order_moments_from_mgf(self, order, x, f=None):
        """
        从矩生成函数（MGF）计算高阶矩。

        参数:
            order (int): 矩的阶数。
            x (torch.Tensor): 输入的张量。
            f (callable, optional): 可选的变换函数。默认为 None。

        返回:
            float: 计算得到的高阶矩。
        """
        # 计算基数
        cardinality = int(math.log2(len(x)))
        self.set_cardinality(cardinality)
        if order == 0:
            # 零阶矩为 1
            return 1
        else:
            # 获取分布向量
            p = self.cache[cardinality]['vector']
            if f is not None:
                # 若有变换函数，对输入进行变换
                x = f(x)
            # 调用外部函数计算高阶矩
            return calculate_high_order_moments_from_mgf(p, x, order)

    def high_order_moments_from_cf(self, order, x, f=None):
        """
        从特征函数（CF）计算高阶矩。

        参数:
            order (int): 矩的阶数。
            x (torch.Tensor): 输入的张量。
            f (callable, optional): 可选的变换函数。默认为 None。

        返回:
            float: 计算得到的高阶矩。
        """
        # 计算基数
        cardinality = int(math.log2(len(x)))
        self.set_cardinality(cardinality)
        if order == 0:
            # 零阶矩为 1
            return 1
        else:
            # 获取分布向量
            p = self.cache[cardinality]['vector']
            if f is not None:
                # 若有变换函数，对输入进行变换
                x = f(x)
            # 调用外部函数计算高阶矩
            return calculate_high_order_moments_from_cf(p, x, order)
