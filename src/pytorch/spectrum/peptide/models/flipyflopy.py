import torch
from torch import nn
import numpy as np
from massspec_ml.pytorch.base_objects import ModelOutput
from massspec_ml.pytorch.spectrum.spectrum_base_objects import *
try:
    from bayesian_torch.layers import LinearFlipout
except ImportError:
    pass


# Default pytorch initialization behavior is uniform distribution
#  from -(1/sqrt(fan_in)) - +(1/sqrt(fan_in))
Init = lambda P: nn.init.normal_(torch.empty(P), 0.0, 0.03)
class TalkingHeads(SpectrumModule):
    def __init__(self, config,
                 in_ch, # incoming embedding channels
                 dk, # dimension for keys and queries; dkxN
                 dv, # dimension for values; dvxN
                 hv, # heads in the end; project from h
                 hk=None, # heads in first NxN map
                 h=None, # heads in attention/softmax map; project from hk
                 out_dim=None
                 ):
        super().__init__(config)
        self.dk = dk
        self.dv = dv
        self.hv = hv
        self.hk = self.hv if hk==None else hk
        self.h = self.hv if h==None else h
        self.out_dim = in_ch if out_dim==None else out_dim

        init = Init
        drop = self.config.ml.model.FlipyFlopy.get('drop', 0.0)
        if drop != 0.0:
            self.drop = nn.Dropout(drop)
        std = self.config.ml.model.FlipyFlopy.std
        
        self.Wq = nn.Parameter(init((in_ch, self.dk, self.hk)), requires_grad=True)
        self.Wk = nn.Parameter(init((in_ch, self.dk, self.hk)), requires_grad=True)
        self.Wv = nn.Parameter(init((in_ch, self.dv, self.hv)), requires_grad=True)

        self.Wl = nn.Parameter(init((self.hk, self.h)), requires_grad=True)
        self.Ww = nn.Parameter(nn.init.normal_(torch.empty((self.h, self.hv)), 0.0, std), requires_grad=True)
        if self.config.ml.bayesian_network.bayes:
            self.Wo = LinearFlipout(self.dv*self.hv, self.out_dim)
        else:
            self.Wo = nn.Parameter(Init((self.dv*self.hv, self.out_dim)), requires_grad=True)
    
    def forward(self, inp):
        Q = torch.relu(torch.einsum('abc,bde->adce', inp, self.Wq)) # bs, dk, feat_dim, hk
        K = torch.relu(torch.einsum('abc,bde->adce', inp, self.Wk)) # bs, dk, feat_dim, hk
        V = torch.relu(torch.einsum('abc,bde->adce', inp, self.Wv)) # bs, dv, feat_dim, hv
        
        J = torch.einsum('abcd,abed->aced', Q, K) / K.shape[1]**0.5 # bs, feat_dim, feat_dim, hk
        EL = torch.einsum('abcd,de->abce', J, self.Wl) # bs, feat_dim, feat_dim, h
        W = torch.softmax(EL, dim=2)
        U = torch.einsum('abcd,de->abce', W, self.Ww) # bs, feat_dim, feat_dim, hv
        O = torch.einsum('abcd,aecd->abed', U, V) # bs, feat_dim, dv, hv
        O = O.reshape(O.shape[0], O.shape[1], O.shape[2]*O.shape[3]) # bs, feat_dim, dv*hv
        if self.config.ml.bayesian_network.bayes:
            resid = self.Wo(O)[0].transpose(-1,-2)
        else:
            resid = torch.einsum('abc,cd->abd', O, self.Wo).transpose(-1,-2) # bs, dv*hv, feat_dim
        if self.config.ml.model.FlipyFlopy.get('drop', 0.0) != 0.0:
            resid = self.drop(resid)
        return inp + resid

class FFN(SpectrumModule):
    def __init__(self, config,
                 in_ch,
                 units
                 ):
        super().__init__(config)
        self.W1 = nn.Parameter(Init((in_ch, units)), requires_grad=True)
        if self.config.ml.bayesian_network.bayes:
            self.W2 = LinearFlipout(units, in_ch)
        else:
            self.W2 = nn.Parameter(Init((units, in_ch)), requires_grad=True)
        drop = self.config.ml.model.FlipyFlopy.get('drop', 0)
        if drop != 0.0:
            self.drop = nn.Dropout(drop)

    def forward(self, inp):
        out = torch.relu(torch.einsum('abc,bd->adc', inp, self.W1))
        if self.config.ml.bayesian_network.bayes:
            resid = torch.relu(self.W2(out.transpose(-1,-2))[0].transpose(-1,-2))
        else:
            resid = torch.relu(torch.einsum('abc,bd->adc', out, self.W2))
        if self.config.ml.model.FlipyFlopy.get('drop', 0.0) != 0.0:
            resid = self.drop(resid)
        return inp + resid

class TransBlock(SpectrumModule):
    def __init__(self, config,
                 in_ch,
                 dk,
                 dv,
                 hv,
                 hk=None,
                 h=None,
                 units=None
                 ):
        super().__init__(config)
        hk = hv if hk=='None' else hk
        h = hv if h=='None' else h
        self.head = TalkingHeads(config, in_ch, dk, dv, hv, hk, h)
        units = in_ch if units=='None' else units
        self.ffn = FFN(config, in_ch, units)
        self.normH = nn.BatchNorm1d(units)
        self.normF = nn.BatchNorm1d(units)
    
    def forward(self, inp):
        return self.normF(self.ffn(self.normH(self.head(inp))))

class FlipyFlopy(SpectrumModel):
    def __init__(self, config):
        super().__init__(config)
        outdim = self.bins
        #dropout = self.config.ml.model.AIomicsModel.dropout
        #filtstart = self.config.ml.model.AIomicsModel.filtstart
        #bayes = self.config.ml.baye
        in_ch = self.channels
        seq_len = self.config.ml.embedding.max_len
        embedsz = self.config.ml.model.FlipyFlopy.embedsz
        head = self.config.ml.model.FlipyFlopy.head
        assert(len(head)==5)
        head = tuple(head)
        units = self.config.ml.model.FlipyFlopy.units
        units = embedsz if units==None else units
        blocks = self.config.ml.model.FlipyFlopy.blocks
        filtend = self.config.ml.model.FlipyFlopy.filtend
        drop = self.config.ml.model.FlipyFlopy.get('drop', 0.0)
        if drop != 0.0:
            self.drop = nn.Dropout(drop)
        std = self.config.ml.model.FlipyFlopy.std
        
        self.embed = nn.Parameter(Init((in_ch, embedsz)), requires_grad=True)
        self.pos = nn.Parameter(Init((embedsz, seq_len)), requires_grad=True)
        self.embed_norm = nn.BatchNorm1d(embedsz)
        self.main = nn.Sequential(*[TransBlock(*((config,)+(embedsz,)+head+(units,))) for _ in range(blocks)])
        if self.config.ml.bayesian_network.bayes:
            self.proj = LinearFlipout(embedsz, filtend)
        else:
            self.proj = nn.Parameter(Init((embedsz, filtend)), requires_grad=True)
        self.proj_norm = nn.BatchNorm1d(filtend)
        modules = []
        if drop != 0.0:
            modules.append(self.drop)
        modules.append(nn.Linear(filtend, outdim))
        modules.append(nn.Sigmoid())
        self.final = nn.Sequential(*modules)
        nn.init.xavier_normal_(self.final[-2].weight)
        nn.init.zeros_(self.final[-2].bias)

    def forward(self, inp):
        out = self.embed_norm(torch.einsum('abc,bd->adc', inp[0], self.embed) + self.pos)
        out = self.main(out)
        if self.config.ml.bayesian_network.bayes:
            out = torch.relu(self.proj_norm(self.proj(out.transpose(-1,-2))[0].transpose(-1,-2)))
        else:
            out = torch.relu(self.proj_norm(torch.einsum('abc,bd->adc', out, self.proj)))
        out = self.final(out.transpose(-1,-2))
        return ModelOutput(y_prime=out.mean(dim=1, keepdims=True))

