''' Define the sublayers used in the transformer model
Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    '''TODO: compute embeddings from raw pixel set data.
    '''

    def __init__(self, n_channels, n_pixels, d_model):
        super(EmbeddingLayer, self).__init__()

        self.d_model = d_model
        self.linear = nn.Linear(n_channels * n_pixels, d_model)

    def forward(self, x):
        batch_size, len_seq, n_channels, n_pixels = x.shape
        x = x.view(batch_size, len_seq, n_channels * n_pixels)
        x = F.relu(self.linear(x))
        x = x.view(batch_size, len_seq, self.d_model)
        return x
    

class NDVI(nn.Module):
    '''TODO: compute NDVI time series from raw pixel set data.
    NDVI = (NIR - RED) / (NIR + RED)
    '''

    def __init__(self):
        super(NDVI, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        raise NotImplementedError
    

class BI(nn.Module):
    '''TODO: compute BI time series from raw pixel set data.
    BI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
    '''
    def __init__(self):
        super(BI, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        raise NotImplementedError
    

class SpectralIndicesLayer(nn.Module):
    '''TODO: compute features based on NDVI and BI time series from raw pixel set data.
    '''

    def __init__(self, d_model, blue=1, red=2, near_infrared=6, swir1=8, eps=1e-3):
        super(SpectralIndicesLayer, self).__init__()
        self.ndvi = NDVI(red, near_infrared, eps)
        self.bi = BI(blue, red, near_infrared, swir1, eps)
        self.mlp = nn.Linear(2 * d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch_size x len_seq x n_channels x n_pixels data
        """
        raise NotImplementedError
    

class PositionalEncoding(nn.Module):
    ''' Positional Encoding Layer.
    Source: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
    TODO: Update the positional encoding as described in "Satellite Image Time Series 
    Classification with Pixel-Set Encoders and Temporal Self-Attention, Garnot et al."
    '''
    def __init__(self, d_hid, n_position=365):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(1000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, doys):
        """TODO: update forward function to return the positional embedding only.
        """
        # doys.shape = batch_size x max_len_seq
        batch_size = doys.shape[0]
        pos_table = self.pos_table  # (1 x n_position x d_hid)
        pos_table = pos_table.repeat(batch_size, 1, 1)  # (batch_size x n_position x d_hid)
        doys = doys.unsqueeze(-1).repeat(1, 1, pos_table.shape[-1])
        positional_embedding = torch.gather(pos_table, index=doys, dim=1)
        return positional_embedding

class Temporal_Aggregator(nn.Module):
    ''' TODO: aggregate embeddings that are not masked.
    '''
    def __init__(self, mode='mean'):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, data, mask):
        if self.mode == 'mean':
            raise NotImplementedError
        elif self.mode == 'identity':
            out = data
        else:
            raise NotImplementedError
        return out