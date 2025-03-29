# 代码功能概述

## 离散矩母函数计算

`discrete_mgf` 函数用于计算离散随机变量的矩母函数 $M_X(t)$，其定义为 $M_X(t)=\mathbb{E}[e^{tX}]=\sum_{x} e^{tx} P(X = x)$
，其中 $x$ 是随机变量 $X$ 的所有可能取值，$P(X = x)$ 是对应的概率。

## 高阶矩计算

`calculate_high_order_moments` 函数通过对矩母函数求 $n$ 阶导数并在 $t = 0$ 处求值，从而得到离散随机变量的 $n$
阶矩。这里利用了矩母函数的性质：$\mathbb{E}[X^n] = \left. \frac{d^n}{dt^n} M_X(t) \right|_{t = 0}$。

# 代码示例

```python
import torch


def discrete_mgf(p, x, t):
    """
    计算离散随机变量的矩母函数
    :param p: 概率向量
    :param x: 随机变量取值向量
    :param t: 参数 t
    :return: 矩母函数的值
    """
    exp_term = torch.exp(t * x)
    product_term = exp_term * p
    mgf_value = torch.sum(product_term)
    return mgf_value


def calculate_high_order_moments(p, x, order):
    """
    计算离散随机变量的高阶矩
    :param p: 概率向量
    :param x: 随机变量取值向量
    :param order: 矩的阶数
    :return: 高阶矩的值
    """
    t = torch.tensor(0.0, requires_grad=True)
    mgf_value = discrete_mgf(p, x, t)
    for _ in range(order):
        # 计算梯度
        grads = torch.autograd.grad(mgf_value, t, create_graph=True)[0]
        mgf_value = grads
    # 在 t = 0 处计算高阶矩
    high_order_moment = mgf_value.detach()
    return high_order_moment
```

# 注意事项

- 矩母函数并非对所有分布都存在，在使用时需要确保所处理的分布的矩母函数存在。
- 对于高阶求导，可能会受到数值稳定性的影响，尤其是在阶数较高时。