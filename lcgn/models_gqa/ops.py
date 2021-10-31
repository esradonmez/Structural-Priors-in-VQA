import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


activations = {
    'NON': lambda x: x,
    'TANH': torch.tanh,
    'SIGMOID': F.sigmoid,
    'RELU': F.relu,
    'ELU': F.elu,
}


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        fan_avg = (self.in_features + self.out_features) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


class Project(nn.Module):
    def __init__(self, in_d, out_d, m_d=None, num_hidden_layers=2, dropout = 0.2):
        super().__init__()
        if m_d is None:
            m_d = in_d

        self.layer_in = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Dropout(dropout),
            Linear(in_d, m_d),
            nn.ReLU()
        )
        self.layer_hidden = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(m_d),
            Linear(m_d, m_d),
            nn.ReLU()) for _ in range(num_hidden_layers-1)])
        
        self.layer_out = Linear(m_d, out_d)

        self.proj = Linear(in_d, out_d)
    
    def forward(self,x):
        residual_x = self.proj(x)
        x = self.layer_in(x)
        for layer in self.layer_hidden:
            x = layer(x)
        x = self.layer_out(x)
        return residual_x + x # linear residual vs non-linear

class ClassProject(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, num_hidden_layers=2, dropout=0.1):
        super().__init__()

        self.linear_in = Linear(n_in, n_hidden)
        self.seq = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout),
            Linear(n_hidden, n_hidden),
            nn.ELU()) for _ in range(num_hidden_layers)])
        
        self.linear_out = Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.linear_in(x)
        for layer in self.seq[:-1]:
            x = layer(x) + x
        x = self.linear_out(x)
        return x

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, *args,
                 **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size, *args, **kwargs)

        # compatible with xavier_initializer in TensorFlow
        if isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            kernel_h, kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size, kernel_size
        fan_in = kernel_h * kernel_w * in_channels
        fan_out = kernel_h * kernel_w * out_channels
        fan_avg = (fan_in + fan_out) / 2.
        bound = np.sqrt(3. / fan_avg)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

class Rnn(nn.Module):

    def __init__(self,input_size, hidden_size):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers = 1, batch_first=True, bidirectional=True)
        #initialize here differently if needed 
    
    def forward(self, x):
        _, (encoded, c0) = self.rnn(x)
        encoded = encoded.reshape(x.size(0), -1)
        return encoded 


class ExponentialMovingAverage():
    def __init__(self, param_dict, decay):
        assert decay >= 0. and decay < 1.
        self.decay = decay
        self.params_ema = {
            name: torch.zeros_like(p.data) for name, p in param_dict.items()}

    def step(self, param_dict):
        for name, p in param_dict.items():
            self.params_ema[name].mul_(self.decay).add_(1-self.decay, p.data)

    def state_dict(self):
        return self.params_ema

    def load_state_dict(self, state_dict):
        assert self.params_ema.keys() == state_dict.keys()
        for k, p_ema in self.params_ema.items():
            assert state_dict[k].dtype == p_ema.dtype
            assert state_dict[k].size() == p_ema.size()
            p_ema[...] = state_dict[k]

    def set_params_from_ema(self, param_dict):
        for k, p in param_dict.items():
            p.data[...] = self.params_ema[k]


def apply_mask1d(attention, image_locs):
    batch_size, num_loc = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask = torch.ge(tmp1, tmp2)
    attention = attention.masked_fill(mask, -1e30)
    return attention


def apply_mask2d(attention, image_locs):
    batch_size, num_loc, _ = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask1d = torch.ge(tmp1, tmp2)
    mask2d = mask1d[:, None, :] | mask1d[:, :, None]
    attention = attention.masked_fill(mask2d, -1e30)
    return attention

def apply_mask3d(attention, image_locs):
    '''
    we have to apply mask now considering the concept space 

    '''
    batch_size, cc_num, num_loc, _ = attention.size()
    tmp1 = attention.new_zeros(num_loc)
    tmp1[:num_loc] = torch.arange(
        0, num_loc, dtype=attention.dtype).unsqueeze(0)

    tmp1 = tmp1.expand(batch_size, num_loc)
    tmp2 = image_locs.type(tmp1.type())
    tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
    mask1d = torch.ge(tmp1, tmp2)
    mask2d = mask1d[:, None, :] | mask1d[:, :, None]
    attention = attention.masked_fill(mask2d, -1e30)
    return attention



def generate_scaled_var_drop_mask(shape, keep_prob):
    assert keep_prob > 0. and keep_prob <= 1.
    mask = torch.rand(shape, device='cuda').le(keep_prob)
    mask = mask.float() / keep_prob
    return mask
