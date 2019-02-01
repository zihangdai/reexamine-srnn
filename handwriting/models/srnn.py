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
    def __init__(self, n_mix, d_data, d_emb, d_mlp, d_rnn, d_lat, dropout, 
                 tie_weight=True, tie_projs = [False], **kwargs):
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
        
        self.forward_z = nn.LSTM(d_emb, d_rnn)
        self.forward_h = nn.LSTM(d_emb + d_mlp, d_rnn)
        self.bwd_rnn = nn.LSTM(d_emb, d_rnn)
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
    
    def backward_pass(self, y):
        y = self.inp_emb(y)
        y = torch.flip(y, (0,)) 
        output, hn = self.bwd_rnn(y)
        output = torch.flip(output, (0,))
        return output                     
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)  
    
    def forward(self, x, y, hidden=None, mask=None):
        y0 = y[:,:,0:1]
        y1 = y[:,:,1:2]
        y2 = y[:,:,2:]
        
        qlen, bsz, _ = x.size()
            
        backward_vecs = self.backward_pass(y)    
        x = self.inp_emb(x)

        if hidden is None:
            forward_vecs, _ = self.forward_z(x)
        else:
            forward_vecs, hidden[0] = self.forward_z(x, hidden[0])

        z_pri = self.prior(forward_vecs)
        z_pri = torch.clamp(z_pri, -8., 8.)
        z_mu , z_theta = torch.chunk(z_pri, 2, -1)

        z_post = torch.cat([backward_vecs, forward_vecs], -1)
        z_post = self.post(z_post)
        z_post = torch.clamp(z_post, -8.,  8.)
        mu, theta = torch.chunk(z_post, 2, -1)

        z = self.reparameterize(mu, theta)

        proj_z = self.gen_mod(z)
        cat_xz = torch.cat([x, proj_z], -1)
        
        hidden, _ = self.forward_h(cat_xz)

        p0 = self.classifier(hidden)
        loss0 = self.crit0(p0, y0)
        loss1 = self.crit1(hidden, y1)
        loss2 = self.crit2(hidden, y2)
        loss = loss0 + loss1 + loss2
        loss = loss.squeeze(-1)
        if mask is not None:
            loss = loss * mask

        #compute KL(q||p)
        kld = gaussian_kld([mu, theta], [z_mu, z_theta])
        kld = kld.sum(-1)
        if mask is not None:
            kld = kld * mask

        """
            DataParallel requires the `parallel_dim` to exist and the size sum
            along the `parallel_dim` to be the same as the the input.

            So, we return 2D tensors of shape [seqlen x local_bsz]
        """
        
        return loss, -kld, None
    
    def eval_with_prior(self, x, y, mask=None):
        qlen, bsz, _ = x.size()
        #print(x.size())
        x = self.inp_emb(x)
        
        forward_vecs, _ = self.forward_z(x)

        z_pri = self.prior(forward_vecs)
        z_pri = torch.clamp(z_pri, -8., 8.)
        z_mu , z_theta = torch.chunk(z_pri, 2, -1)

        z = self.reparameterize(z_mu, z_theta)

        proj_z = self.gen_mod(z)
        cat_xz = torch.cat([x, proj_z], -1)
        
        output, _ = self.forward_h(cat_xz)
        
        x = output  
        loss = self.crit(x, y)
        loss = loss.sum(-1)
        if mask is not None:
            loss = loss * mask
        
        return loss, 0, None
    
    def init_hidden(self, batch_size):
        fwd_z_h0 = self.init_zero_weight((self.n_layers, batch_size, self.d_rnn))
        fwd_z_c0 = self.init_zero_weight((self.n_layers, batch_size, self.d_rnn))
        fwd_h_h0 = self.init_zero_weight((self.n_layers, batch_size, self.d_rnn))
        fwd_h_c0 = self.init_zero_weight((self.n_layers, batch_size, self.d_rnn))
        hidden = [(fwd_z_h0, fwd_z_c0), (fwd_h_h0, fwd_h_c0)]

        return hidden
        
    def init_zero_weight(self, shape):
        weight = next(self.parameters())
        return weight.new_zeros(shape)

if __name__ == '__main__':
    d_data, d_emb, d_rnn, n_layer  = 3, 150, 200, 1
    m = Model(None, d_data, d_emb, 150, d_rnn, 150, 0.)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    print('parameter number:', sum([p.nelement() for p in m.parameters()]))
    x = torch.rand(20, 32, d_data) * 2 - 1
    y = torch.rand(20, 32, d_data) * 2 - 1
    #hidden = m.init_hidden(32);
    nll_loss, _, _ = m(x, y)
    nll_loss.sum().backward()
