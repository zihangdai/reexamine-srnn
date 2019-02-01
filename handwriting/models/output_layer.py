
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

        
class GaussianMixture(nn.Module):
    def __init__(self, n_mix, d_inp, learn_var=True, share_prior=False):
        super(GaussianMixture, self).__init__()
        """
            The current implementation is super simplified, treating each dim 
            of the target as a one-dimensional Gaussian mixture with separate
            mixture weights if `share_prior == False` (default).

            When `share_prior == True`, all `d_tgt` target dims share the same
            mixture weights, which poses some inductive bias.

            However, neither is the optimal case, as corellations between the 
            target dims is essentially ignored. 

            Input:
                inp    : [... x d_inp]
                target : [... x d_tgt]

            Return:
                nll    : [... x d_tgt]
        """
        
        self.n_mix = n_mix
        self.d_tgt = d_tgt = 1
        self.d_inp = d_inp

        self.learn_var = learn_var
        self.share_prior = share_prior

        self.mean = nn.Linear(d_inp, d_tgt * n_mix)
        if learn_var:
            self.var = nn.Linear(d_inp, d_tgt * n_mix, bias=False)
        if n_mix > 1:
            if share_prior:
                self.prior = nn.Linear(d_inp, n_mix)
            else:
                self.prior = nn.Linear(d_inp, d_tgt * n_mix)
        else:
            assert n_mix == 1, '`n_mix` must be positive integers'
    
        self.const = -0.5 * math.log(2 * math.pi)

    def log_likelihood(self, target, mean, log_std, log_prior=None):
        """
            target    : [... x d_tgt]
            mean      : [... x d_tgt x n_mix]
            log_std   : [... x d_tgt x n_mix]
            log_prior : [... x d_tgt x n_mix] or None
        """

        # Gaussian log-likelihood is not safe for half precision due to the 
        # `log_std.exp()` operation, especially in the backward pass.
        # For simplicity, we use float32 for log-likelihood computation.
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

        return log_prob

    def forward(self, inp, target, return_mean=False):
        
        mean = self.mean(inp)
        mean = mean.view(*mean.size()[:-1], self.d_tgt, self.n_mix)
        

        if self.learn_var:
            log_std = self.var(inp)
            log_std = log_std.view(*log_std.size()[:-1], self.d_tgt, self.n_mix)
        else:
            log_std = torch.zeros(1, dtype=inp.dtype, device=inp.device)
        

        if self.n_mix > 1:
            prior = self.prior(inp)
            if self.share_prior:
                prior = prior.view(*prior.size()[:-1], 1, self.n_mix)
            else:
                prior = prior.view(*prior.size()[:-1], self.d_tgt, self.n_mix)
            log_prior = F.log_softmax(prior, dim=-1)
            
        else:
            log_prior = None

        log_prob = self.log_likelihood(target, mean, log_std, log_prior)
        
        nll = - log_prob 

        if return_mean:
            return nll, mean
        else:
            return nll