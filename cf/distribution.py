import torch
from dstz.core.atom import Element
from dstz.core.distribution import Evidence

from cf.generation import axis_generation


def decimal_to_binary_set(decimal_number):
    """
    将十进制数转换为对应的二进制集合表示。

    参数:
    decimal_number (int): 输入的十进制数。

    返回:
    set: 包含对应二进制位为 1 的字符表示的集合。
    """
    # 将十进制数转换为二进制字符串，并去除前缀 '0b'
    binary_str = bin(decimal_number)[2:]
    # 反转二进制字符串，方便后续处理
    binary_str_reversed = binary_str[::-1]
    result_set = set()
    # 遍历反转后的二进制字符串
    for i, bit in enumerate(binary_str_reversed):
        # 如果当前位为 1
        if bit == '1':
            # 将对应的字符添加到结果集合中
            result_set.add(chr(ord('A') + i))
    return result_set


def distribution(cardinality, generation_function, ignore_zero=True):
    """
    根据给定的基数和生成函数生成分布。

    参数:
    cardinality (int): 基数，用于确定分布的维度。
    generation_function (callable): 生成函数，用于生成分布的值。
    ignore_zero (bool, optional): 是否忽略零元素。默认为 True。

    返回:
    torch.Tensor: 生成的分布张量。
    """
    # 生成坐标轴
    x = axis_generation(cardinality)
    # 使用生成函数计算分布值
    gx = generation_function(x)
    # 检查生成函数的输出是否有负数
    if torch.any(gx < 0):
        # 若有负数，抛出断言错误
        assert 'Negative Generation Function'
    # 计算分布值的总和
    sum_factor = gx.sum()
    # 归一化分布值
    distribution = gx / sum_factor
    # 如果忽略零元素
    if ignore_zero:
        # 创建一个零张量
        zero_tensor = torch.tensor([0.0], device=distribution.device)
        # 将零张量和分布张量拼接
        distribution = torch.cat([zero_tensor, distribution])
    return distribution


def vec2mass(distribution):
    """
    将分布向量转换为质量分布。

    参数:
    distribution (torch.Tensor): 输入的分布向量。

    返回:
    Evidence: 质量分布对象。
    """
    # 创建一个证据对象
    mass = Evidence()
    # 遍历分布向量
    for idx, value in enumerate(distribution):
        # 如果当前值大于 0
        if value > 0:
            # 将索引转换为二进制集合表示的焦点元素
            focal_element = Element(decimal_to_binary_set(idx))
            # 将焦点元素及其对应的质量添加到证据对象中
            mass[focal_element] = value.item()  # 转换为 Python 浮点数
    return mass
