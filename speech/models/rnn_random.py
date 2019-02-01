import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import GMCriterion, BernoulliCriterion
from models.utils import *
from models import base


class Model(base.RNNBase):
  def __init__(self, d_data, d_emb, d_rnn, n_layer, dropout=0., **kwargs):
    super(Model, self).__init__(d_data, d_emb, d_rnn, n_layer, dropout)

    self.d_leak = kwargs.get('d_leak', d_data // 4)

    perm = torch.randperm(self.d_data)
    self.leak_indices = perm[:self.d_leak]
    self.rest_indices = perm[self.d_leak:]

    self.rest_hid_layer = nn.Sequential(
        nn.Linear(d_rnn + self.d_leak, d_rnn, bias=False),
        nn.LayerNorm(d_rnn),
        nn.ReLU(inplace=True),
        nn.Linear(d_rnn, d_rnn, bias=False),
        nn.LayerNorm(d_rnn),
        nn.ReLU(inplace=True)
    )

    self.drop = nn.Dropout(dropout)

    self.n_mix = kwargs.get('n_mix', 20)
    self.crit_leak = GMCriterion(self.n_mix, self.d_leak, self.d_rnn)
    self.crit_rest = GMCriterion(self.n_mix, d_data-self.d_leak, self.d_rnn)


  def forward(self, x, y, hidden=None, mask=None):
    qlen, bsz, _ = x.size()

    x = self.inp_emb(x)

    if hidden is None:
      output, _ = self.rnn(x)
    else:
      output, hidden = self.rnn(x, hidden)

    # output: [seqlen x bsz x dim]
    y_leak = y[:,:,self.leak_indices]
    y_rest = y[:,:,self.rest_indices]

    # (1): log p(y_leak | output)
    loss_leak = self.crit_leak(self.drop(output), y_leak)

    # (2): log p(y_rest | y_leak, output)
    output_rest = self.rest_hid_layer(torch.cat([output, y_leak], -1))
    loss_rest = self.crit_rest(self.drop(output_rest), y_rest)

    loss = loss_leak.sum(-1) + loss_rest.sum(-1)
    if mask is not None:
      loss = loss * mask

    return loss, repackage_hidden(hidden)

