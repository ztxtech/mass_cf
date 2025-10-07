# Mass Function 描述的离散随机变量的矩母函数与特征函数计算项目

> 英文版本请参考 [README.md](README.md)

## 项目概述

本项目主要用于计算 Mass Function 描述的离散随机变量的矩母函数（Moment Generating Function, MGF）和特征函数（Characteristic Function, CF），并通过它们计算离散随机变量的高阶矩。同时，项目还包含了傅里叶变换相关的函数，以及用于生成和处理质量分布的类。

## 为什么使用 PyTorch

### 自动求导功能

PyTorch 提供了强大的自动求导系统（Autograd），它能根据用户定义的计算图自动计算梯度。在本项目中，我们需要计算离散随机变量矩母函数的高阶导数，手动求导过程繁琐且容易出错。使用 PyTorch 的自动求导功能，我们可以通过简单的代码实现高阶导数的计算，大大提高了开发效率。

### 动态计算图

PyTorch 的动态计算图特性允许在运行时动态构建计算图。这使得代码更加灵活，便于调试和实现复杂的模型。在计算高阶矩时，我们可以根据不同的阶数动态调整计算过程，而无需提前定义固定的计算图。

### 张量计算

PyTorch 的张量（Tensor）数据结构和高效的张量计算能力，使得我们可以方便地处理大规模数据。在本项目中，我们需要处理概率向量、随机变量取值向量等数据，PyTorch 的张量操作可以高效地完成这些计算任务。

## 为什么利用求导能加速高阶矩的计算

### 理论基础

根据特征函数的性质，离散随机变量的 $n$ 阶矩可以通过对特征函数 $\varphi(t)$ 求 $n$ 阶导数并在 $t = 0$ 处求值得到，即 $\mathbb{E}[X^n] = \frac{1}{i^n} \left. \frac{d^n \varphi(t)}{dt^n} \right|_{t = 0}$。利用这个性质，我们可以将高阶矩的计算转化为求导运算。

### 计算效率

传统的高阶矩计算方法可能需要对每个样本进行多次乘法和求和操作，计算复杂度较高。而利用求导的方法，我们可以通过自动求导系统一次性计算出高阶导数，避免了重复的计算过程，从而提高了计算效率。特别是在处理大规模数据时，这种方法的优势更加明显。

### 代码实现

在代码中，我们通过设置 `create_graph=True` 参数，在计算梯度时创建一个新的计算图，从而实现高阶导数的计算。以下是一个简单的代码示例：

````python
```python
import torch

def discrete_cf(p, x, t):
    """
    计算离散随机变量的特征函数
    :param p: 概率向量
    :param x: 随机变量取值向量
    :param t: 参数 t
    :return: 特征函数的值
    """
    exp_term = torch.exp(1j * t * x)  # 1j 表示虚数单位
    product_term = exp_term * p
    cf_value = torch.sum(product_term)
    return cf_value

def calculate_high_order_moments(p, x, order):
    """
    计算离散随机变量的高阶矩
    :param p: 概率向量
    :param x: 随机变量取值向量
    :param order: 矩的阶数
    :return: 高阶矩的值
    """
    t = torch.tensor(0.0, requires_grad=True)
    cf_value = discrete_cf(p, x, t)
    for _ in range(order):
        # 计算梯度，并创建新的计算图以支持高阶导数计算
        grads = torch.autograd.grad(cf_value, t, create_graph=True)[0]
        cf_value = grads
    # 在 t = 0 处计算高阶矩
    high_order_moment = cf_value.detach() / (1j**order)
    return high_order_moment
````

## 安装依赖

在项目根目录下运行以下命令安装所需的 Python 依赖包：

```bash
pip install -r requirements.txt
```

## 运行项目

在项目根目录下运行以下命令启动主程序：

```bash
python src/main.py
```

## 代码示例

以下是一个简单的代码示例，展示如何使用 `FMassDistribution` 类计算高阶矩：

```python
from cf.generation import sin_p1
from cf.group import FMassDistribution

# 创建 FMassDistribution 实例
fm = FMassDistribution(sin_p1)

# 设置基数
cardinality = 5

# 采样
vector, mass = fm.sampling(cardinality)

# 计算高阶矩
order = 3
cf = fm.high_order_moments_from_cf(order, vector)
print(f"特征函数法计算的 {order} 阶矩: {cf}")
```

## 注意事项

- 请确保在运行项目前已经安装了所需的 Python 依赖包。
- 由于数值稳定性的问题，高阶导数计算可能会产生较大的误差。在处理高精度要求的任务时，需要考虑使用其他方法或进行数值稳定性优化。

## 作者

- Duozi Lin (**Main Contributor & Concept & Methodological Design**)
- Xinyuan Chen (**Concept**)
- Tianxiang Zhan (**Mentor Role**)
