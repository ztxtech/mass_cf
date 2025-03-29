import torch


def axis_generation(cardinality, ignore_zero=True):
    if ignore_zero:
        return torch.arange(1, 2 ** cardinality, dtype=torch.float32)
    else:
        return torch.arange(2 ** cardinality, dtype=torch.float32)


def decimal_to_cardinality(decimal_number):
    binary_str = bin(decimal_number)[2:]
    return binary_str.count('1')


def sin_p1(x):
    return torch.sin(x) + 1


def inf_content(x):
    cards = torch.tensor([decimal_to_cardinality(i) for i in range(len(x))], dtype=torch.float32)

    res = -torch.log2(x / (torch.pow(2, cards) - 1))
    res[torch.isinf(res)] = 0
    res[torch.isnan(res)] = 0
    return res
