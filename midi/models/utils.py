import torch
import torch.nn as nn
import torch.nn.functional as F
import math;
import numpy as np
from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    elif isinstance(h, (tuple, list)):
        return [repackage_hidden(v) for v in h]
    elif h is None:
        return None
    else:
        return h;


def gaussian_kld(left, right):
    """
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    mu_left, logvar_left = left; mu_right, logvar_right = right
    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (torch.exp(logvar_left) / torch.exp(logvar_right)) +
                        ((mu_left - mu_right)**2.0 / torch.exp(logvar_right)) - 1.0)
    return gauss_klds


def mu_law_encoding(audio, n_channel):
    mu = torch.FloatTensor([n_channel - 1]).to(audio.device)
    one = torch.FloatTensor([1.]).to(audio.device)
    safe_audio_abs = torch.min(torch.abs(audio), one)
    magnitude = torch.log1p(mu * safe_audio_abs) / torch.log1p(mu)
    signal = torch.sign(audio) * magnitude
    data = ((signal + 1) / 2 * mu + 0.5)
    data = data.long()
    return data


def reset_parameters(weight, bias):
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    nn.init.constant_(bias, 0)


