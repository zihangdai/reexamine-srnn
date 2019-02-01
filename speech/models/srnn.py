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

    self.crit = GMCriterion(1, d_data, self.d_rnn)


  def forward(self, x, y, hidden=None, mask=None):
    ##### srnn forward
    z, output, mu_prior, logvar_prior, mu_post, theta_post = \
      self.srnn_forward(x, y, hidden)

    ##### loss
    # NLL
    nll = self.crit(output, y)
    nll = nll.sum(-1)
    if mask is not None:
      nll = nll * mask

    # KL(q||p)
    kld = gaussian_kld([mu_post, theta_post], [mu_prior, logvar_prior])
    kld = kld.sum(-1)
    if mask is not None:
      kld = kld * mask

    return nll, -kld, repackage_hidden(hidden)


  def eval_with_prior(self, x, y, mask=None):
    qlen, bsz, _ = x.size()
    x = self.inp_emb(x)

    fwd_vecs, _ = self.fwd_rnn(x)

    param_prior = self.prior(fwd_vecs)
    param_prior = torch.clamp(param_prior, -8., 8.)
    mu_prior , logvar_prior = torch.chunk(param_prior, 2, -1)

    z = self.reparameterize(mu_prior, logvar_prior)

    cat_xz = torch.cat([fwd_vecs, z], -1)
    output, _ = self.lat_rnn(cat_xz)

    x = output
    nll = self.crit(x, y)
    nll = nll.sum(-1)
    if mask is not None:
      nll = nll * mask

    return nll, 0, None

