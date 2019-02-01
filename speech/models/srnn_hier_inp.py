import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import *
from models.utils import *
from models import base


class Model(base.SRNNBase):
  def __init__(self, d_data, d_emb, d_mlp, d_rnn, d_lat, n_layer, dropout=0.,
               **kwargs):
    super(Model, self).__init__(d_data, d_emb, d_mlp, d_rnn, d_lat, n_layer,
                                dropout)

    ##### low-level modules
    self.inp_emb_low = nn.Sequential(
        nn.Linear(1, d_emb),
        nn.Dropout(dropout)
    )

    self.n_low_layer = kwargs.get('n_low_layer', 1)
    self.rnn_low = nn.LSTM(d_rnn + d_emb, d_rnn, dropout=dropout)
    for i in range(self.n_low_layer):
      params = 'self.rnn_low.weight_hh_l{}'.format(i)
      nn.init.orthogonal_(eval(params).data)

    self.init_h = nn.Parameter(
        torch.Tensor(self.n_low_layer, self.d_rnn).uniform_(-0.01, 0.01))
    self.init_c = nn.Parameter(
        torch.Tensor(self.n_low_layer, self.d_rnn).uniform_(-0.01, 0.01))

    ##### output modules
    crit_d_tgt = self.d_data
    crit_inp_shape = (d_data, self.d_rnn)
    self.crit = GMCriterion(1, crit_d_tgt, crit_inp_shape)


  def _init_hid_low(self, bsz):
    h = self.init_h[:,None,:].expand(-1, bsz, -1).contiguous()
    c = self.init_c[:,None,:].expand(-1, bsz, -1).contiguous()
    return (h, c)


  def forward(self, x, y, hidden=None, mask=None):
    qlen, bsz, _ = x.size()

    ##### high-level computation
    z, output, mu_prior, logvar_prior, mu_post, theta_post = \
        self.srnn_forward(x, y)


    ##### low-level computation
    x_low = y.permute(2, 0, 1).contiguous()
    x_low = x_low.view(self.d_data, qlen * bsz, 1)

    # input to the low-level rnn
    x_low_emb = self.inp_emb_low(x_low)
    extra_inp = output.view(1, qlen * bsz, self.d_rnn) \
                      .expand(self.d_data, -1, -1)
    extra_inp = self.drop(extra_inp)
    inp_low = torch.cat([x_low_emb, extra_inp], -1)

    # initial state for the low-level
    hid_low = self._init_hid_low(qlen * bsz)

    # low-level rnn forward
    out_low, _ = self.rnn_low(inp_low, hid_low)

    # out_low: [d_data x (qlen*bsz) x d_rnn]
    out_low = torch.cat([hid_low[0], out_low[:-1]], 0)
    out_low = self.drop(out_low)

    # out_low: [qlen x bsz x d_data x d_rnn]
    out_low = out_low.view(self.d_data, qlen, bsz, self.d_rnn) \
                     .permute(1, 2, 0, 3).contiguous()

    ##### loss
    # NLL
    nll = self.crit(out_low, y)
    nll = nll.sum(-1)
    if mask is not None:
      nll = nll * mask

    # KL(q||p)
    kld = gaussian_kld([mu_post, theta_post], [mu_prior, logvar_prior])
    kld = kld.sum(-1)
    if mask is not None:
        kld = kld * mask

    return nll, -kld, None

