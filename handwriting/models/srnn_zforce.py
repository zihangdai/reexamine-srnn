import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import *
from models.utils import *

class LReLU(nn.Module):
    def __init__(self, c=1./3):
        super(LReLU, self).__init__()
        self.c = c

    def forward(self, x):
        return torch.clamp(F.leaky_relu(x, self.c), -3., 3.)

class Model(nn.Module):
    def __init__(self, n_mix, d_data, d_emb, d_mlp, d_rnn, d_lat, 
                 dropout, tie_weight=True, tie_projs = [False], **kwargs):
        super(Model, self).__init__()
        self.d_data = d_data
        self.d_emb = d_emb
        self.d_mlp = d_mlp
        self.d_rnn = d_rnn
        self.d_lat = d_lat

        self.inp_emb = nn.Sequential(
            nn.Linear(d_data, d_emb), 
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(d_rnn, 1)
        self.crit0 = nn.BCEWithLogitsLoss(reduction='none')
        self.crit1 = GaussianMixture(n_mix, d_rnn)
        self.crit2 = GaussianMixture(n_mix, d_rnn)

        self.bwd_rnn = nn.LSTM(d_emb, d_rnn)
        self.fwd_rnn = nn.LSTMCell(d_emb+d_mlp, d_rnn)
        self.gen_mod = nn.Linear(d_lat, d_mlp)

        nn.init.orthogonal_(self.bwd_rnn.weight_hh_l0.data)

        self.prior = nn.Sequential(
            nn.Linear(d_rnn, d_mlp), LReLU(), 
            nn.Linear(d_mlp, d_lat * 2)
        )
        self.post = nn.Sequential(
            nn.Linear(d_rnn * 2, d_mlp), LReLU(), 
            nn.Linear(d_mlp, d_lat * 2)
        )


    def init_zero_weight(self, shape):
        weight = next(self.parameters())
        return weight.new_zeros(shape)

    def reparameterize(self, mu, logvar, eps=None):
        std = logvar.mul(0.5).exp_()
        if eps is None:
            eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def backward_pass(self, y):
        y = self.inp_emb(y)
        y_flip = torch.flip(y, (0,))

        brnn_outs_flip, brnn_hid_flip = self.bwd_rnn(y_flip)
        brnn_outs = torch.flip(brnn_outs_flip, (0,))

        return brnn_outs
       
    def forward_pass(self, x, brnn_outs):
        
        x_emb = self.inp_emb(x)

        frnn_hid = [self.init_zero_weight((x.size(1), self.d_rnn)),
                    self.init_zero_weight((x.size(1), self.d_rnn))]
        frnn_out = frnn_hid[0]

        # sample all noise once
        noises = x.new(x.size(0), x.size(1), self.d_lat).float().normal_()
        
        kld = 0.
        frnn_outs = []
        prior_mus, prior_logvars, post_mus, post_logvars = [], [], [], []
        for step in range(x.size(0)):
            prior_param = self.prior(frnn_out)
            prior_param = torch.clamp(prior_param, -8., 8.)
            prior_mu , prior_logvar = torch.chunk(prior_param, 2, -1)

            post_inp = torch.cat([brnn_outs[step], frnn_out], -1)
            post_param = self.post(post_inp)
            post_param = torch.clamp(post_param, -8., 8.)
            post_mu, post_logvar = torch.chunk(post_param, 2, -1)

            # [bsz x d_lat]
            z = self.reparameterize(post_mu, post_logvar, eps=noises[step])

            # forward rnn step
            proj_z = self.gen_mod(z)
            frnn_inp = torch.cat([x_emb[step], proj_z], -1)
            frnn_hid = self.fwd_rnn(frnn_inp, frnn_hid)

            frnn_out = frnn_hid[0]
            frnn_outs.append(frnn_out)

            # # compute KL(q||p)
            # kld += gaussian_kld(
            #     [prior_mu, prior_logvar], [post_mu, post_logvar])

            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar)
            post_mus.append(post_mu)
            post_logvars.append(post_logvar)

        # compute all KL(q||p) once
        kld = gaussian_kld(
            [torch.stack(post_mus), torch.stack(post_logvars)], 
            [torch.stack(prior_mus), torch.stack(prior_logvars)])

        frnn_outs = torch.stack(frnn_outs, 0)
        return frnn_outs, kld
    
    def forward(self, x, y, hidden=None, mask=None):
        
        y0 = y[:,:,0:1]
        y1 = y[:,:,1:2]
        y2 = y[:,:,2:]
        
        qlen, bsz, _ = x.size()
        
            
        brnn_outs = self.backward_pass(y)    
        
        hidden, kld = self.forward_pass(x, brnn_outs)

        p0 = self.classifier(hidden)
        loss0 = self.crit0(p0, y0)
        loss1 = self.crit1(hidden, y1)
        loss2 = self.crit2(hidden, y2)
        loss = loss0 + loss1 + loss2
        loss = loss.squeeze(-1)

        # sum over the seq_len (0) & seg_len (2) and avg over the batch_size (1)
        nll_loss = loss
        kld_loss = kld.sum(2)
        if mask is not None:
            nll_loss = nll_loss * mask
            kld_loss = kld_loss * mask

        return nll_loss, -kld_loss, hidden
    
if __name__ == '__main__':
    d_data, d_emb, d_mlp, d_rnn, d_lat, dropout = 3, 150, 150, 250, 150, 0.
    m = Model(None, d_data, d_emb, d_mlp, d_rnn, d_lat, dropout)
    print('parameter number:', sum([p.nelement() for p in m.parameters()]))
    x = torch.rand(32, 8, d_data)
    y = torch.rand(32, 8, d_data)

    nll_loss, kld_loss, _ = m(x, y, None)

