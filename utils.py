import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from typing import Union, Tuple


def dates2doys(dates: list[str]):
    '''TODO: convert a list of dates (list of str) 
    to a list of days of year.
    '''
    raise NotImplementedError


def pad_tensor(x: torch.Tensor, l: int, pad_value=0.):
    ''' Adds padding to a tensor.
    '''
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def fill_ts(ts: torch.Tensor, doys: torch.Tensor, full_doys: torch.Tensor):
    ''' Fill the gaps in a time series with NaN values.
    Args:
        ts: time series with missing data
        doys: days of year of the time series
        full_doys: complete list of days of year (including missing dates)
    '''
    full_length = len(full_doys)
    ts = pad_tensor(ts, full_length, pad_value=torch.nan)
    missing_doys = torch.tensor(list(
        set(full_doys.tolist()) - set(doys.tolist())
    ))
    missing_doys, _ = missing_doys.sort()
    doys = torch.cat((doys, missing_doys))
    doys, indices = doys.sort()
    indices = indices.view(-1, 1, 1).repeat(1, ts.shape[1], ts.shape[2])
    ts = torch.gather(ts, index=indices, dim=0)
    return ts


def get_params(model: torch.nn.Module):
    '''TODO: compute the number of trainable parameters of a model.
    '''
    raise NotImplementedError


def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    '''Credit: https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/
    '''
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops