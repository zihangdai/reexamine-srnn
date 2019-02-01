import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import GMCriterion, BernoulliCriterion
from models.utils import *
from models import base


class Model(base.RNNBase):
  def __init__(self, d_data, d_emb, d_rnn, n_layer, dropout=0., **kwargs):
    super(Model, self).__init__(d_data, d_emb, d_rnn, n_layer, dropout)

    self.crit = BernoulliCriterion(d_data, self.d_rnn)


  def forward(self, x, y, hidden=None, mask=None):
    qlen, bsz, _ = x.size()

    x = self.inp_emb(x)
    if hidden is None:
      output, _ = self.rnn(x)
    else:
      output, hidden = self.rnn(x, hidden)

    output = self.drop(output)

    loss = self.crit(output, y)
    loss = loss.sum(-1)
    if mask is not None:
      loss = loss * mask

    return loss, repackage_hidden(hidden)

