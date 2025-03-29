import torch


def dft_matrix(N):
    i, j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    omega = torch.exp(-2j * torch.pi * i * j / N)
    return omega


def dft(x):
    N = len(x)
    X = dft_matrix(N) @ x.to(dtype=torch.complex64)
    return X


def idft_matrix(N):
    i, j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')
    omega = torch.exp(2j * torch.pi * i * j / N)
    return omega / N


def idft(X):
    N = len(X)
    x = idft_matrix(N) @ X
    return x.to(dtype=torch.float32)
