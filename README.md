<div align="center">

  <h2>Central Moments of Belief Information</h2>

  <br>
  
  <p>
      Xingyuan Chen<sup>1,2,3</sup>&nbsp;
      Duozi Lin<sup>4</sup>&nbsp;
      <a href="https://ztxtech.github.io/">Tianxiang Zhan</a><sup>1</sup>&nbsp;
      Yong Deng<sup>1 ★</sup>&nbsp;
  </p>
  
  <p>
      <sup>1</sup> University of Electronic Science and Technology of China &nbsp;&nbsp;
      <sup>2</sup> Kunming University &nbsp;&nbsp; <br>
      <sup>3</sup> Yunnan Province Key Laboratory of Intelligent Logistics Equipment and Systems &nbsp;&nbsp; 
      <sup>4</sup> Zhejiang University &nbsp;&nbsp; 
      <sup>★</sup> Corresponding Author <br>
  </p>
  
  <p align="center">
    <a href="./README_zh.md">中文版本</a>
  </p>
  
</div>

## Project Overview

This project is primarily used to calculate the Moment Generating Function (MGF) and Characteristic Function (CF) of discrete random variables described by a mass function, and compute higher-order moments of discrete random variables through them. Additionally, the project includes functions related to Fourier transforms and classes for generating and processing mass distributions.

## Why Use PyTorch

### Automatic Differentiation

PyTorch provides a powerful automatic differentiation system (Autograd), which can automatically compute gradients based on the computation graph defined by the user. In this project, we need to compute high-order derivatives of the moment generating function of discrete random variables, which is tedious and error-prone when done manually. Using PyTorch's automatic differentiation capability, we can achieve high-order derivative computation with simple code, greatly improving development efficiency.

### Dynamic Computation Graph

PyTorch's dynamic computation graph feature allows the computation graph to be constructed dynamically at runtime. This makes the code more flexible and easier to debug and implement complex models. When calculating high-order moments, we can dynamically adjust the computation process according to different orders without having to predefine a fixed computation graph.

### Tensor Computation

PyTorch's tensor data structure and efficient tensor computation capabilities make it easy to handle large-scale data. In this project, we need to process probability vectors, random variable value vectors, and other data, and PyTorch's tensor operations can efficiently complete these computation tasks.

## Why Derivatives Can Accelerate High-Order Moment Calculation

### Theoretical Foundation

According to the properties of characteristic functions, the $n$-th moment of a discrete random variable can be obtained by taking the $n$-th derivative of the characteristic function $\varphi(t)$ and evaluating it at $t = 0$, i.e., $\mathbb{E}[X^n] = \frac{1}{i^n} \left. \frac{d^n \varphi(t)}{dt^n} \right|_{t = 0}$. Using this property, we can transform the calculation of high-order moments into differentiation operations.

### Computational Efficiency

Traditional methods for calculating high-order moments may require multiple multiplication and summation operations for each sample, resulting in high computational complexity. With the differentiation method, we can compute high-order derivatives directly through the automatic differentiation system, avoiding repeated computation processes and thereby improving computational efficiency. This advantage is particularly evident when dealing with large-scale data.

### Code Implementation

In the code, we set the `create_graph=True` parameter to create a new computation graph when computing gradients, thereby enabling high-order derivative computation. Here is a simple code example:

```python
import torch

def discrete_cf(p, x, t):
    """
    Calculate the characteristic function of a discrete random variable
    :param p: Probability vector
    :param x: Random variable value vector
    :param t: Parameter t
    :return: Value of the characteristic function
    """
    exp_term = torch.exp(1j * t * x)  # 1j represents the imaginary unit
    product_term = exp_term * p
    cf_value = torch.sum(product_term)
    return cf_value

def calculate_high_order_moments(p, x, order):
    """
    Calculate high-order moments of a discrete random variable
    :param p: Probability vector
    :param x: Random variable value vector
    :param order: Order of the moment
    :return: Value of the high-order moment
    """
    t = torch.tensor(0.0, requires_grad=True)
    cf_value = discrete_cf(p, x, t)
    for _ in range(order):
        # Compute gradient and create a new computation graph to support high-order derivative computation
        grads = torch.autograd.grad(cf_value, t, create_graph=True)[0]
        cf_value = grads
    # Calculate high-order moment at t = 0
    high_order_moment = cf_value.detach() / (1j**order)
    return high_order_moment
```

## Installation Dependencies

Run the following command in the project root directory to install the required Python dependency packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the following command in the project root directory to start the main program:

```bash
python src/main.py
```

## Code Example

Here is a simple code example showing how to use the `FMassDistribution` class to compute high-order moments:

```python
from cf.generation import sin_p1
from cf.group import FMassDistribution

# Create FMassDistribution instance
fm = FMassDistribution(sin_p1)

# Set cardinality
cardinality = 5

# Sampling
vector, mass = fm.sampling(cardinality)

# Calculate high-order moments
order = 3
cf = fm.high_order_moments_from_cf(order, vector)
print(f"Characteristic function method calculated {order}-th moment: {cf}")
```

## Notes

- Please ensure that the required Python dependency packages are installed before running the project.
- Due to numerical stability issues, high-order derivative computation may produce large errors. When handling tasks with high precision requirements, other methods or numerical stability optimization should be considered.

## Authors

- Duozi Lin (**Main Contributor & Concept & Methodological Design**)
- Xinyuan Chen (**Concept**)
- Tianxiang Zhan (**Mentor Role**)
