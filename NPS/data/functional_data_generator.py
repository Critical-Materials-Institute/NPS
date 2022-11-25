import os

from data import common

import numpy as np
#import imageio

#import torch
import torch.utils.data as data

from .toy_spectrum_generator import generate

class toy_spectrum(): #data.Dataset):
    def __init__(self, args, name='Demo', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.scale = args.scale
        self.idx_scale = 0
        self.train = train
        self.benchmark = benchmark
        self.n_in = args.n_colors
        self._cache = None
        data_range = [list(map(int,r.split('-'))) for r in args.data_range.split('/')]
        if self.train:
            self.length=data_range[0][1]-data_range[0][0]
            f = os.path.join(args.dir_data, 'train.npy')
            if os.path.exists(f):
                self._cache = [(x[:1],x[1:]) for x in np.load(f)]
        else:
            self.length=data_range[1][1]-data_range[1][0]
            f = os.path.join(args.dir_data, 'test.npy')
            if os.path.exists(f):
                self._cache = [(x[:1],x[1:]) for x in np.load(f)]
            else:
                self._cache = [generate(self.n_in) for i in range(self.length)]

    def __getitem__(self, idx):
        if not self._cache:
            hr, lr = generate(self.n_in)
        else:
            hr, lr = self._cache[idx]
        return lr, hr, str(idx)

    def __len__(self):
        return self.length

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

