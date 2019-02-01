import numpy as np
import torch

def print_t(x):
    print(x.cpu().detach().squeeze())

def all_close(a, b):
    return np.allclose(a.cpu().detach().numpy(), b.cpu().detach().numpy())

def check_all_close(a, b):
    if all_close(a, b):
        # print(True)
        pass
    else:
        print((a.cpu()-b.cpu()).abs().max().item())

def check_inf_nan(x, info=''):
    if x is not None:
        n_inf = torch.isinf(x).int().sum().item()
        n_nan = torch.isnan(x).int().sum().item()
        if n_inf or n_nan:
            print(info + '#inf {}, #nan {}, #tot {}'.format(n_inf, n_nan, x.numel()))

