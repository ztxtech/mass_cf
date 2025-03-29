import torch


def axis_generation(cardinality, ignore_zero=True):
    """
    根据给定的基数生成坐标轴。

    参数:
    cardinality (int): 基数，用于确定坐标轴的范围。
    ignore_zero (bool, optional): 是否忽略零。默认为 True。

    返回:
    torch.Tensor: 生成的坐标轴张量。
    """
    if ignore_zero:
        # 若忽略零，从 1 开始生成到 2 的 cardinality 次幂的张量
        return torch.arange(1, 2 ** cardinality, dtype=torch.float32)
    else:
        # 若不忽略零，从 0 开始生成到 2 的 cardinality 次幂的张量
        return torch.arange(2 ** cardinality, dtype=torch.float32)


def decimal_to_cardinality(decimal_number):
    """
    计算十进制数对应的二进制表示中 1 的个数。

    参数:
    decimal_number (int): 输入的十进制数。

    返回:
    int: 二进制表示中 1 的个数。
    """
    # 将十进制数转换为二进制字符串并去掉前缀 '0b'
    binary_str = bin(decimal_number)[2:]
    # 统计二进制字符串中 1 的个数
    return binary_str.count('1')


def sin_p1(x):
    """
    对输入的张量每个元素应用正弦函数并加 1。

    参数:
    x (torch.Tensor): 输入的张量。

    返回:
    torch.Tensor: 应用正弦函数并加 1 后的张量。
    """
    return torch.sin(x) + 1


def inf_content(x):
    """
    计算信息含量。

    参数:
    x (torch.Tensor): 输入的张量。

    返回:
    torch.Tensor: 计算得到的信息含量张量。
    """
    # 计算每个元素对应的二进制表示中 1 的个数
    cards = torch.tensor([decimal_to_cardinality(i) for i in range(len(x))], dtype=torch.float32)
    # 计算信息含量
    res = -torch.log2(x / (torch.pow(2, cards) - 1))
    # 将结果中无穷大的值置为 0
    res[torch.isinf(res)] = 0
    # 将结果中 NaN 的值置为 0
    res[torch.isnan(res)] = 0
    return res
