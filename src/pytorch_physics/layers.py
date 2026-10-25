import torch

def linear(input, weight, bias=None):
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output += bias
    return output