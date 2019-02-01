import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import gaussian_kld, repackage_hidden


class LReLU(nn.Module):
    def __init__(self, c=1./3):
        super(LReLU, self).__init__()
        self.c = c

    def forward(self, x):
        return torch.clamp(F.leaky_relu(x, self.c), -3., 3.)


class RNNBase(nn.Module):
  def __init__(self, d_data, d_emb, d_rnn, n_layer, dropout=0.):
    super(RNNBase, self).__init__()

    self.d_data = d_data
    self.d_emb = d_emb
    self.d_rnn = d_rnn
    self.n_layer = n_layer

    self.inp_emb = nn.Sequential(
      nn.Linear(d_data, d_emb),
      nn.Dropout(dropout)
    )

    self.rnn = nn.LSTM(d_emb, d_rnn, num_layers=n_layer, dropout=dropout)

    self.drop = nn.Dropout(dropout)

    for i in range(n_layer):
      params = 'self.rnn.weight_hh_l{}'.format(i)
      nn.init.orthogonal_(eval(params).data)


  def forward(self, x, y, hidden=None, mask=None):
    raise NotImplemented('Abstract class')


  def init_hidden(self, batch_size):
    h0 = self.init_zero_weight((self.n_layer, batch_size, self.d_rnn))
    c0 = self.init_zero_weight((self.n_layer, batch_size, self.d_rnn))
    hidden = (h0, c0)

    return hidden


  def init_zero_weight(self, shape):
    weight = next(self.parameters())

    return weight.new_zeros(shape)


class SRNNBase(nn.Module):
  def __init__(self, d_data, d_emb, d_mlp, d_rnn, d_lat, n_layer, dropout=0.):
    super(SRNNBase, self).__init__()

    self.d_data = d_data
    self.d_emb = d_emb
    self.d_rnn = d_rnn
    self.d_lat = d_lat
    self.d_mlp = d_mlp

    self.n_layer = n_layer

    self.drop = nn.Dropout(dropout)

    self.inp_emb = nn.Sequential(
        nn.Linear(d_data, d_emb),
        nn.Dropout(dropout)
    )

    self.fwd_rnn = nn.LSTM(d_emb, d_rnn, dropout=dropout, num_layers=n_layer)
    self.bwd_rnn = nn.LSTM(d_emb, d_rnn, dropout=dropout, num_layers=n_layer)
    for i in range(n_layer):
      params = 'self.fwd_rnn.weight_hh_l{}'.format(i)
      nn.init.orthogonal_(eval(params))
      params = 'self.bwd_rnn.weight_hh_l{}'.format(i)
      nn.init.orthogonal_(eval(params))

    self.prior = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_rnn, d_mlp), LReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_mlp, d_lat * 2)
    )
    self.post = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(d_rnn * 2, d_mlp), LReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_mlp, d_lat * 2)
    )

    self.lat_rnn = nn.LSTM(d_rnn + d_lat, d_rnn, dropout=dropout)
    nn.init.orthogonal_(self.lat_rnn.weight_hh_l0)


  def backward_pass(self, y):
    y = self.inp_emb(y)
    y = torch.flip(y, (0,))
    output, hn = self.bwd_rnn(y)
    output = torch.flip(output, (0,))
    return output


  def reparameterize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)


  def srnn_forward(self, x, y, hidden=None):
    bwd_vecs = self.backward_pass(y)
    x = self.inp_emb(x)

    if hidden is None:
      fwd_vecs, _ = self.fwd_rnn(x)
    else:
      fwd_vecs, hidden[0] = self.fwd_rnn(x, hidden[0])

    param_prior = self.prior(fwd_vecs)
    param_prior = torch.clamp(param_prior, -8., 8.)
    mu_prior, logvar_prior = torch.chunk(param_prior, 2, -1)

    param_post = torch.cat([bwd_vecs, fwd_vecs], -1)
    param_post = self.post(param_post)
    param_post = torch.clamp(param_post, -8.,  8.)
    mu_post, theta_post = torch.chunk(param_post, 2, -1)

    z = self.reparameterize(mu_post, theta_post)
    cat_xz = torch.cat([fwd_vecs, z], -1)
    if hidden is None:
      output, _ = self.lat_rnn(cat_xz)
    else:
      output, hidden[1] = self.lat_rnn(cat_xz, hidden[1])
    output = self.drop(output)

    return z, output, mu_prior, logvar_prior, mu_post, theta_post


  def forward(self, x, y, hidden=None, mask=None):
    raise NotImplemented('Abstract class')


  def init_hidden(self, batch_size):
    fwd_z_h0 = self.init_zero_weight((self.n_layer, batch_size, self.d_rnn))
    fwd_z_c0 = self.init_zero_weight((self.n_layer, batch_size, self.d_rnn))
    fwd_h_h0 = self.init_zero_weight((self.n_layer, batch_size, self.d_rnn))
    fwd_h_c0 = self.init_zero_weight((self.n_layer, batch_size, self.d_rnn))
    hidden = [(fwd_z_h0, fwd_z_c0), (fwd_h_h0, fwd_h_c0)]

    return hidden


  def init_zero_weight(self, shape):
    weight = next(self.parameters())
    return weight.new_zeros(shape)

