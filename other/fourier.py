import torch


def dft_matrix(N):
    """
    生成离散傅里叶变换（DFT）矩阵。

    参数:
    N (int): 矩阵的维度。

    返回:
    torch.Tensor: 一个 N x N 的 DFT 矩阵。
    """
    # 创建两个张量 i 和 j，分别表示矩阵的行和列索引
    i, j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    # 计算 DFT 矩阵的元素，使用指数函数计算旋转因子
    omega = torch.exp(-2j * torch.pi * i * j / N)
    return omega


def dft(x):
    """
    执行离散傅里叶变换（DFT）。

    参数:
    x (torch.Tensor): 输入的一维张量。

    返回:
    torch.Tensor: 经过 DFT 变换后的一维张量。
    """
    # 获取输入张量的长度
    N = len(x)
    # 调用 dft_matrix 函数生成 DFT 矩阵，并与输入张量相乘
    X = dft_matrix(N) @ x.to(dtype=torch.complex64)
    return X


def idft_matrix(N):
    """
    生成离散傅里叶逆变换（IDFT）矩阵。

    参数:
    N (int): 矩阵的维度。

    返回:
    torch.Tensor: 一个 N x N 的 IDFT 矩阵。
    """
    # 创建两个张量 i 和 j，分别表示矩阵的行和列索引
    i, j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    # 计算 IDFT 矩阵的元素，使用指数函数计算旋转因子
    omega = torch.exp(2j * torch.pi * i * j / N)
    # 对矩阵进行归一化处理
    return omega / N


def idft(X):
    """
    执行离散傅里叶逆变换（IDFT）。

    参数:
    X (torch.Tensor): 输入的一维复数张量。

    返回:
    torch.Tensor: 经过 IDFT 变换后的一维实数张量。
    """
    # 获取输入张量的长度
    N = len(X)
    # 调用 idft_matrix 函数生成 IDFT 矩阵，并与输入张量相乘
    x = idft_matrix(N) @ X
    # 将结果转换为实数张量
    return x.to(dtype=torch.float32)
