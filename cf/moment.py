import torch


def discrete_mgf(p, x, t):
    exp_term = torch.exp(t * x)
    product_term = exp_term * p
    mgf_value = torch.sum(product_term)
    return mgf_value


def calculate_high_order_moments_from_mgf(p, x, order):
    t = torch.tensor(0.0, requires_grad=True)
    mgf_value = discrete_mgf(p, x, t)
    for _ in range(order):
        grads = torch.autograd.grad(mgf_value, t, create_graph=True)[0]
        mgf_value = grads
    high_order_moment = mgf_value.detach()
    return high_order_moment


def discrete_characteristic_function(p, x, t):
    complex_t = t.to(torch.complex64)  # 转换为复数类型
    exp_term = torch.exp(1j * complex_t * x)
    product_term = exp_term * p
    cf_value = torch.sum(product_term)
    return cf_value


def calculate_high_order_moments_from_cf(p, x, order):
    t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    cf_value = discrete_characteristic_function(p, x, t)
    real_part = cf_value.real
    imag_part = cf_value.imag
    for _ in range(order):
        real_grads = torch.autograd.grad(real_part, t, create_graph=True)[0]
        imag_grads = torch.autograd.grad(imag_part, t, create_graph=True)[0]
        grads = real_grads + 1j * imag_grads
        real_part = grads.real
        imag_part = grads.imag
    cf_value = real_part + 1j * imag_part
    high_order_moment = (-1j) ** order * cf_value.detach()
    high_order_moment = high_order_moment.real.to(torch.float64)
    return high_order_moment
