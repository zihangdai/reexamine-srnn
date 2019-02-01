
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureNLL(nn.Module):
    def __init__(self):
        super(GaussianMixtureNLL, self).__init__()

        # const in the log-likelihood
        self.const = -0.5 * math.log(2 * math.pi)

    def forward(self, target, mean, log_std, log_prior=None):
        """
            target    : [... x d_tgt]
            mean      : [... x d_tgt x n_mix]
            log_std   : [... x d_tgt x n_mix]
            log_prior : [... x d_tgt x n_mix] or None
        """

        tgt_ = target.unsqueeze(-1).float()
        mean = mean.float()

        # [... x d_tgt x n_mix]
        log_probs = self.const - log_std \
                  - 0.5 * (((tgt_ - mean) / log_std.exp()) ** 2)

        if log_prior is None: # n_mix = 1
            log_prob = log_probs.squeeze(-1)

        else:
            log_prior = log_prior.float()
            # [... x d_tgt x dim]
            w_log_probs = log_prior + log_probs

            # [... x d_tgt x 1]
            max_w_log_prob = w_log_probs.max(-1, keepdim=True)[0]

            # [... x d_tgt]
            log_prob = torch.logsumexp(w_log_probs - max_w_log_prob, dim=-1) \
                     + max_w_log_prob.squeeze(-1)

        nll = -log_prob
        return nll


class GMCriterion(nn.Module):
    def __init__(self, n_mix, d_tgt, inp_shape):
        super(GMCriterion, self).__init__()

        self.n_mix = n_mix
        self.d_tgt = d_tgt

        if isinstance(inp_shape, (tuple, list)):
            assert len(inp_shape) == 2 and inp_shape[0] == d_tgt
            self.d_inp = inp_shape[1]
            self.einstr = 'dim,...id->...im'

        elif isinstance(inp_shape, int):
            self.d_inp = inp_shape
            self.einstr = 'dim,...d->...im'

        self.mean_weight = nn.Parameter(
            torch.Tensor(self.d_inp, d_tgt, n_mix))
        self.mean_bias = nn.Parameter(
            torch.Tensor(d_tgt, n_mix).zero_())
        nn.init.kaiming_uniform_(self.mean_weight, a=math.sqrt(5))

        self.std_weight = nn.Parameter(
            torch.Tensor(self.d_inp, d_tgt, n_mix))
        self.std_bias = nn.Parameter(
            torch.Tensor(d_tgt, n_mix).zero_())
        nn.init.kaiming_uniform_(self.std_weight, a=math.sqrt(5))

        self.prior_weight = nn.Parameter(
            torch.Tensor(self.d_inp, d_tgt, n_mix))
        self.prior_bias = nn.Parameter(
            torch.Tensor(d_tgt, n_mix).zero_())
        nn.init.kaiming_uniform_(self.prior_weight, a=math.sqrt(5))

        self.neg_log_likelihood = GaussianMixtureNLL()


    def forward(self, inp, target):
        mean = torch.einsum(self.einstr, self.mean_weight, inp) \
             + self.mean_bias
        log_std = torch.einsum(self.einstr, self.std_weight, inp) \
                + self.std_bias
        prior = torch.einsum(self.einstr, self.prior_weight, inp) \
              + self.prior_bias
        log_prior = F.log_softmax(prior, dim=-1)

        nll = self.neg_log_likelihood(target, mean, log_std, log_prior)

        return nll


class BernoulliCriterion(nn.Module):
    def __init__(self, d_tgt, inp_shape):
        super(BernoulliCriterion, self).__init__()

        self.d_tgt = d_tgt

        if isinstance(inp_shape, (tuple, list)):
            assert len(inp_shape) == 2 and inp_shape[0] == d_tgt
            self.d_inp = inp_shape[1]
            self.einstr = 'id,...id->...i'

        elif isinstance(inp_shape, int):
            self.d_inp = inp_shape
            self.einstr = 'id,...d->...i'

        self.logit_weight = nn.Parameter(
            torch.Tensor(d_tgt, self.d_inp))
        self.logit_bias = nn.Parameter(
            torch.Tensor(d_tgt).zero_())
        nn.init.kaiming_uniform_(self.logit_weight, a=math.sqrt(5))

        self.neg_log_likelihood = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, inp, target):
        logit = torch.einsum(self.einstr, self.logit_weight, inp) \
              + self.logit_bias

        nll = self.neg_log_likelihood(logit, target)

        return nll


