import torch
from dstz.core.atom import Element
from dstz.core.distribution import Evidence

from cf.generation import axis_generation


def decimal_to_binary_set(decimal_number):
    binary_str = bin(decimal_number)[2:]
    binary_str_reversed = binary_str[::-1]
    result_set = set()
    for i, bit in enumerate(binary_str_reversed):
        if bit == '1':
            result_set.add(chr(ord('A') + i))
    return result_set


def distribution(cardinality, generation_function, ignore_zero=True):
    x = axis_generation(cardinality)
    gx = generation_function(x)
    if torch.any(gx < 0):
        assert 'Negative Generation Function'
    sum_factor = gx.sum()
    distribution = gx / sum_factor
    if ignore_zero:
        zero_tensor = torch.tensor([0.0], device=distribution.device)
        distribution = torch.cat([zero_tensor, distribution])
    return distribution


def vec2mass(distribution):
    mass = Evidence()
    for idx, value in enumerate(distribution):
        if value > 0:
            focal_element = Element(decimal_to_binary_set(idx))
            mass[focal_element] = value.item()  # 转换为 Python 浮点数
    return mass
