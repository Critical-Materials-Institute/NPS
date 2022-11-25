import os

#from data import common

import numpy as np
#import imageio

#import torch
import torch.utils.data as data

class npy(): #data.Dataset):
    def __init__(self, args, name='Demo', train=True):
        self.args = args
        self.name = name
        self.train = train
        self._cache = None
        data_range = [list(map(int,r.split('-'))) for r in args.data_range.split('/')]
        f = os.path.join(args.dir_data, 'train.npy' if train else 'test.npy')
        if os.path.exists(f):
            self._cache = np.load(f)
        self.length = 0

    def __getitem__(self):
        if not self._cache:
            raise "ERROR online generation of data NOT implemented yet"
        else:
            dat = self._cache[idx]
        return dat, str(idx)

    def __len__(self):
        return self.length


