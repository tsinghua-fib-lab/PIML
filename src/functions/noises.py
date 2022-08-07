# -*- coding: utf-8 -*-
"""
@author: zgz

"""
import torch


def random_walk_noise(velocity, mask_v, noise_std_last_step, dvs='cpu'):
    '''
    velocity: (t, N, 2)
    '''
    time_steps = velocity.shape[0]
    rw_noise = torch.randn(list(velocity.shape)) * (noise_std_last_step/time_steps**0.5)
    rw_noise = rw_noise.to(dvs)
    rw_noise *= mask_v.unsqueeze(-1).repeat(1, 1, 2)
    rw_noise = torch.cumsum(rw_noise, dim=0)
    rw_noise *= mask_v.unsqueeze(-1).repeat(1, 1, 2)
    return rw_noise
