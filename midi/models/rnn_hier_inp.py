import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import GMCriterion, BernoulliCriterion
from models.utils import *
from models import base


class Model(base.RNNBase):
  def __init__(self, d_data, d_emb, d_rnn, n_layer, dropout=0., **kwargs):
    super(Model, self).__init__(d_data, d_emb, d_rnn, n_layer, dropout)

    self.inp_emb_low = nn.Sequential(
      nn.Linear(1, d_emb),
      nn.Dropout(dropout)
    )

    self.n_low_layer = kwargs.get('n_low_layer', 1)
    self.rnn_low = nn.LSTM(d_emb+d_rnn, d_rnn, num_layers=self.n_low_layer,
                 dropout=dropout)
    for i in range(self.n_low_layer):
      params = 'self.rnn_low.weight_hh_l{}'.format(i)
      nn.init.orthogonal_(eval(params).data)

    self.init_h = nn.Parameter(
      torch.Tensor(self.n_low_layer, self.d_rnn).uniform_(-0.01, 0.01))
    self.init_c = nn.Parameter(
      torch.Tensor(self.n_low_layer, self.d_rnn).uniform_(-0.01, 0.01))

    crit_d_tgt = 1
    crit_inp_shape = self.d_rnn

    self.crit = BernoulliCriterion(crit_d_tgt, crit_inp_shape)


  def _init_hid_low(self, bsz):
    h = self.init_h[:,None,:].expand(-1, bsz, -1).contiguous()
    c = self.init_c[:,None,:].expand(-1, bsz, -1).contiguous()
    return (h, c)


  def forward(self, x, y, hidden=None, mask=None):
    qlen, bsz, _ = x.size()

    ##### high-level forward
    x_emb = self.inp_emb(x)
    hidden, _ = self.rnn(x_emb)

    ##### low-level forward
    x_low = y.permute(2, 0, 1).contiguous()
    x_low = x_low.view(self.d_data, qlen * bsz, 1)

    # input to the low-level rnn
    x_low_emb = self.inp_emb_low(x_low)
    extra_inp = hidden.view(1, qlen * bsz, self.d_rnn) \
          .expand(self.d_data, -1, -1)
    extra_inp = self.drop(extra_inp)
    inp_low = torch.cat([x_low_emb, extra_inp], -1)

    # initial state for the low-level
    hid_low = self._init_hid_low(qlen * bsz)

    # low-level rnn forward
    output, _ = self.rnn_low(inp_low, hid_low)

    # output: [d_data x (qlen*bsz) x d_rnn]
    output = torch.cat([hid_low[0][-1:], output[:-1]], 0)
    output = self.drop(output)

    # output: [qlen x bsz x d_data x d_rnn]
    output = output.view(self.d_data, qlen, bsz, self.d_rnn) \
                   .permute(1, 2, 0, 3).contiguous()

    # loss: [qlen x bsz x d_data]
    loss = self.crit(output, y.unsqueeze(-1)).squeeze(-1)
    loss = loss.sum(-1)
    if mask is not None:
      loss = loss * mask

    return loss, None

