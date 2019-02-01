#the rnn model use previous hidden states

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.output_layer import *

class Model(nn.Module):
    def __init__(self, n_mix, d_data, d_emb, d_rnn, n_layer, 
                 dropout = 0., tie_weight=True, 
                 tie_projs = [False], **kwargs):
        
        super(Model, self).__init__()
        
        self.d_data = d_data
        self.d_emb = d_emb
        self.d_rnn = d_rnn
        self.n_layers = n_layer

        self.inp_emb = nn.Sequential(
            nn.Linear(d_data, d_emb), 
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(d_rnn, 1)
        self.crit0 = nn.BCEWithLogitsLoss(reduction='none')
        self.crit1 = GaussianMixture(n_mix, d_rnn+1)
        self.crit2 = GaussianMixture(n_mix, d_rnn+2)


        self.rnn = nn.LSTM(d_emb, d_rnn, num_layers=n_layer)
        self.rnn_dropout = nn.Dropout(dropout)
        
        for i in range(n_layer):
            params = 'self.rnn.weight_hh_l{}'.format(i)
            nn.init.orthogonal_(eval(params).data)

            
        
    def forward(self, x, y, hidden=None, mask=None):
        qlen, bsz, _ = x.size()
        y0 = y[:,:,0:1]
        y1 = y[:,:,1:2]
        y2 = y[:,:,2:]
        
        x = self.inp_emb(x)
            
        hidden, _ = self.rnn(x)
        hidden = self.rnn_dropout(hidden)
        
        p0 = self.classifier(hidden)
        loss0 = self.crit0(p0, y0)
        hidden1 = torch.cat([hidden, y0], -1)
        loss1 = self.crit1(hidden1, y1)
        hidden2 = torch.cat([hidden1, y1], -1)
        loss2 = self.crit2(hidden2, y2)
        loss = loss0 + loss1 + loss2
        loss = loss.squeeze(-1)
        if mask is not None:
            loss = loss * mask

        return loss, None
    
    def init_hidden(self, batch_size):
        h0 = self.init_zero_weight((self.n_layers, batch_size, self.d_rnn))
        c0 = self.init_zero_weight((self.n_layers, batch_size, self.d_rnn))
        hidden = (h0, c0)

        return hidden

    def init_zero_weight(self, shape):
        weight = next(self.parameters())
        return weight.new_zeros(shape)
    
if __name__ == '__main__':
    d_data, d_emb, d_rnn, n_layer  = 3, 128, 256, 2
    m = Model(1 << 16, d_data, d_emb, d_rnn, n_layer)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    print('parameter number:', sum([p.nelement() for p in m.parameters()]))
    x = torch.rand(32, 32, d_data) * 2 - 1
    y = torch.rand(32, 32, d_data) * 2 - 1
    #hidden = m.init_hidden(32);
    nll_loss, _ = m(x, y)
    nll_loss.sum().backward()
