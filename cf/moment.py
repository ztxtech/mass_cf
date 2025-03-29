import torch


def discrete_mgf(p, x, t):
    """
    计算离散随机变量的矩母函数

    参数:
    p (torch.Tensor): 概率向量
    x (torch.Tensor): 随机变量取值向量
    t (torch.Tensor): 参数 t

    返回:
    torch.Tensor: 矩母函数的值
    """
    # 计算指数项
    exp_term = torch.exp(t * x)
    # 计算指数项与概率向量的乘积
    product_term = exp_term * p
    # 对乘积项求和得到矩母函数的值
    mgf_value = torch.sum(product_term)
    return mgf_value


def calculate_high_order_moments_from_mgf(p, x, order):
    """
    从矩母函数计算离散随机变量的高阶矩

    参数:
    p (torch.Tensor): 概率向量
    x (torch.Tensor): 随机变量取值向量
    order (int): 矩的阶数

    返回:
    torch.Tensor: 高阶矩的值
    """
    # 初始化参数 t，并开启自动求导
    t = torch.tensor(0.0, requires_grad=True)
    # 计算矩母函数的值
    mgf_value = discrete_mgf(p, x, t)
    # 循环求导 order 次
    for _ in range(order):
        # 计算梯度，并创建新的计算图以支持高阶导数计算
        grads = torch.autograd.grad(mgf_value, t, create_graph=True)[0]
        # 更新矩母函数的值为梯度值
        mgf_value = grads
    # 分离梯度张量，获取高阶矩的值
    high_order_moment = mgf_value.detach()
    return high_order_moment


def discrete_characteristic_function(p, x, t):
    """
    计算离散随机变量的特征函数

    参数:
    p (torch.Tensor): 概率向量
    x (torch.Tensor): 随机变量取值向量
    t (torch.Tensor): 参数 t

    返回:
    torch.Tensor: 特征函数的值
    """
    # 将 t 转换为复数类型
    complex_t = t.to(torch.complex64)
    # 计算指数项
    exp_term = torch.exp(1j * complex_t * x)
    # 计算指数项与概率向量的乘积
    product_term = exp_term * p
    # 对乘积项求和得到特征函数的值
    cf_value = torch.sum(product_term)
    return cf_value


def calculate_high_order_moments_from_cf(p, x, order):
    """
    从特征函数计算离散随机变量的高阶矩

    参数:
    p (torch.Tensor): 概率向量
    x (torch.Tensor): 随机变量取值向量
    order (int): 矩的阶数

    返回:
    torch.Tensor: 高阶矩的值
    """
    # 初始化参数 t，并开启自动求导，设置为 float64 类型
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    # 计算特征函数的值
    cf_value = discrete_characteristic_function(p, x, t)
    # 获取特征函数的实部
    real_part = cf_value.real
    # 获取特征函数的虚部
    imag_part = cf_value.imag
    # 循环求导 order 次
    for _ in range(order):
        # 计算实部的梯度，并创建新的计算图以支持高阶导数计算
        real_grads = torch.autograd.grad(real_part, t, create_graph=True)[0]
        # 计算虚部的梯度，并创建新的计算图以支持高阶导数计算
        imag_grads = torch.autograd.grad(imag_part, t, create_graph=True)[0]
        # 合并实部和虚部的梯度
        grads = real_grads + 1j * imag_grads
        # 更新实部为梯度的实部
        real_part = grads.real
        # 更新虚部为梯度的虚部
        imag_part = grads.imag
    # 合并实部和虚部得到更新后的特征函数值
    cf_value = real_part + 1j * imag_part
    # 计算高阶矩
    high_order_moment = (-1j) ** order * cf_value.detach()
    # 取高阶矩的实部，并转换为 float64 类型
    high_order_moment = high_order_moment.real.to(torch.float64)
    return high_order_moment
