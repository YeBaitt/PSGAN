import numpy as np
import torch
import torch.nn as nn
from performer_pytorch import Performer


class PerfPredSqz(nn.Module):
    def __init__(self, Win, Wout, D, heads=5, dep=8, ff_mult=4):
        super().__init__()
        self.model_name = 'M_seq'
        self.token_emb = nn.Linear(D, D)
        self.pos_enc = FixedPositionalEmbedding(dim=D, max_seq_len=Win)
        self.layer_pos_enc = FixedPositionalEmbedding(dim=D//heads, max_seq_len=Win)
        
        # calculate a list of dimensions [Win, ... , Wout]
        Ws = np.round(Win * (Wout/Win) ** np.linspace(0, 1, dep+1)).astype(int)
        enc_perf = []
        enc_lin = []
        enc_bn = []
        for i in range(dep):
            enc_perf.append(Performer(dim=D, depth=1, heads=heads, causal=False, feature_redraw_interval=1, dim_head=None))
            enc_lin.append(nn.Linear(Ws[i], Ws[i+1]))
            enc_bn.append(nn.BatchNorm1d(Ws[i+1]))
        self.enc_perf = nn.ModuleList(enc_perf)
        self.enc_lin = nn.ModuleList(enc_lin)
        # self.enc_bn = nn.ModuleList(enc_bn)
        
        for m in self.enc_lin:
            nn.init.kaiming_normal_(m.weight)

        self.D = D
        
    def forward(self, x, **args):
        x = x.transpose(1, 2)
        for i in range(len(self.enc_perf)):
            if i == 0:
                x = self.token_emb(x) + self.pos_enc(x)[:,:,:x.shape[-1]]
                # x = x + self.pos_enc(x)[:,:,:x.shape[-1]]
                x = self.enc_perf[0](x, pos_enc=self.layer_pos_enc(x))
            else:
                x = self.enc_perf[i](x)
            x = x.transpose(-1, -2)
            x = self.enc_lin[i](x)
            x = x.transpose(-1, -2)
            # x = self.enc_bn[i](x)
            if i+1 < len(self.enc_perf):
                x = nn.functional.gelu(x)
        x = torch.tanh(x)
        x = x.transpose(1, 2)
        return x.reshape(x.shape[0],-1)
    
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)
