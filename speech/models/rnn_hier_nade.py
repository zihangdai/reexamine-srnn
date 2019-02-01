import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import GMCriterion, BernoulliCriterion
from models.utils import *
from models import base


class Model(base.RNNBase):
  def __init__(self, d_data, d_emb, d_rnn, n_layer, dropout=0., **kwargs):
    super(Model, self).__init__(d_data, d_emb, d_rnn, n_layer, dropout)

    self.d_nade = kwargs.get('d_nade', 16)
    self.n_low_layer = kwargs.get('n_low_layer', 1)

    self._create_params()

    self.n_mix = kwargs.get('n_mix', 20)
    self.crit = GMCriterion(self.n_mix, d_data, (d_data, self.d_nade))


  def _create_params(self):
    nade_w_0 = nn.Parameter(torch.Tensor(
      self.d_data,
      self.d_rnn + self.d_data - 1,
      self.d_nade,
      1))

    nn.init.kaiming_uniform_(nade_w_0, a=math.sqrt(5))

    nade_b_0 = nn.Parameter(torch.Tensor(
      self.d_data,
      self.d_nade).zero_())

    self.register_parameter('nade_w_0', nade_w_0)
    self.register_parameter('nade_b_0', nade_b_0)

    mask_0 = torch.tril(torch.ones(
      self.d_data, self.d_rnn + self.d_data - 1), self.d_rnn - 1)
    mask_0 = mask_0[:,:,None,None]
    self.register_buffer('mask_0', mask_0)

    self.norm_mlp = nn.ModuleList()
    self.norm_mlp.append(nn.LayerNorm([self.d_data, self.d_nade]))
    for l in range(1, self.n_low_layer):
      nade_w_l = nn.Parameter(torch.Tensor(
        self.d_data,
        self.d_nade,
        self.d_nade))
      nn.init.kaiming_uniform_(nade_w_l, a=math.sqrt(5))
      nade_b_l = nn.Parameter(torch.Tensor(
        self.d_data,
        self.d_nade).zero_())

      self.register_parameter('nade_w_{}'.format(l), nade_w_l)
      self.register_parameter('nade_b_{}'.format(l), nade_b_l)
      self.norm_mlp.append(nn.LayerNorm([self.d_data, self.d_nade]))


  def nade_forward(self, inp):
    w_0 = self.nade_w_0 * self.mask_0
    b_0 = self.nade_b_0

    out = torch.einsum('...vj, uvij->...ui', inp, w_0) + b_0
    out = self.norm_mlp[0](out)
    out = F.relu(out)
    out = self.drop(out)

    for l in range(1, self.n_low_layer):
      w_l = getattr(self, 'nade_w_{}'.format(l))
      b_l = getattr(self, 'nade_b_{}'.format(l))

      out = torch.einsum('...vj,vij->...vi', out, w_l) + b_l
      out = self.norm_mlp[l](out)
      out = F.relu(out)
      out = self.drop(out)

    return out


  def forward(self, x, y, hidden=None, mask=None):
    qlen, bsz, _ = x.size()

    ##### high-level rnn forward
    x_emb = self.inp_emb(x)

    hidden, _ = self.rnn(x_emb)
    hidden = self.drop(hidden)

    ##### low-level nade forward
    # [qlen x bsz x d_rnn+d_data-1 x 1]
    inp_low = torch.cat([hidden, y[:,:,:-1]], dim=-1).unsqueeze(-1)

    # [qlen x bsz x d_data x d_nade]
    out_low = self.nade_forward(inp_low)

    loss = self.crit(out_low, y)

    loss = loss.sum(-1)
    if mask is not None:
      loss = loss * mask

    return loss, repackage_hidden(hidden)

