__author__ = 'Fei Zhou'

# Data augmentation
import numpy as np
import torch
import torch.nn as nn
import re

from .pointgroup import PointGroup
from .noise_operator import noise_operator


class slice_operator:
    def __init__(self, args):
        """slice_op "SPEC_1,...,SPEC_N" where SPEC_i is length (if >0) or all (-1). E.g. "-1,-1,10" to take 10 whole x-y planes""" #"50_60" of length 50-60
        # super().__init__()
        specs = []
        self.periodic = args.periodic
        self.dim = args.dim
        self.dim_sp = 3 if args.channel_first else 2
        print('Training data will be sampled from slices of', args.slice_op)
        for x in filter(bool, args.slice_op.split(',')):
            specs.append(int(x))
            # start, length = x.split(':')
            # start = start.split('_')
            # start0 = int(start[0])
            # start1 = None if len(start) == 1 else int(start[1])
            # assert length.startswith('+')
            # length = length.split('_')
            # len0 = int(length[0][1:])
            # len1 = len0+1 if len(length) == 1 else int(length[1])
            # specs.append([start0, start1, len0, len1])
        if len(specs) == 1:
            specs *= args.dim
        self.specs = specs

    def __call__(self, x): # spatial indices are located at 2,...
        dim_sp = self.dim_sp
        r = np.random.randint
        if not self.specs:
            return x
        shape = x.shape[dim_sp:dim_sp+self.dim]
        if self.periodic:
            "TBD"
        else:
            start = [r(0, shape[i]-self.specs[i]+1) if self.specs[i]>0 else 0 for i in range(self.dim)]
            s = tuple([slice(0,None)]*dim_sp + 
              [slice(start[i], start[i]+ self.specs[i] if self.specs[i]>0 else None) for i in range(self.dim)])
            return x[s]


from e3nn import o3
def _rand_rotate_o3(g):
    rot = o3.rand_matrix()
    for attr in ('pos', 'velocities'):
        if hasattr(g, attr):
            g[attr] = g[attr] @ rot.T
    return g

class data_augment_operator:
    def __init__(self, args):
        # super().__init__()
        ops = list(filter(bool, args.data_aug.split(',')))
        print(f'During training, augmenting data with {ops}')
        # assert all([x in ('spg', 'slice', 'noise') for x in ops])
        self.pointgroup_op = PointGroup(args.pointgroup, args.dim, args.channel_first) if args.pointgroup != '1' else lambda x:x
        self.slice_op = slice_operator(args) if args.slice_op else lambda x:x
        self.noise_op = noise_operator(args.noise_op)
        op_dict = {'spg': self.pointgroup_op, 'slice': self.slice_op, 'noise': self.noise_op, 'rot_o3': _rand_rotate_o3}
        self.ops = [op_dict[x] for x in ops]

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x

if __name__ == '__main__':
    pass
    # x = torch.randn(4,8)
    # print(f'x = {x}')
    # for typ_str in ('add_uniform/0:1/1e-2', 'mul_uniform/1:2/1e-2', 'drop/1:6/0.3', 'add_normal/0:1/1e-3,mul_uniform/-2:9/1e-2'):
    #     print(f'  noise type {typ_str}')
    #     noise_op = noise_operator(typ_str)
    #     xp = noise_op(x.clone())
    #     print(xp, xp-x)

