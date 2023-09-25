__author__ = 'Fei Zhou'

import os
import numpy as np
import torch
from torch_geometric.data import Data
# import pickle
from NPS.data.longclip import longclip
# from NPS_common.io_utils import co, read_traj, read_topol
# from NPS_common.utils import unique_list_str


def register_args(parser):
    pass
    # parser.add_argument('--data_suffix', type=str, default='pdb', help='data file type: pdb|lammpstraj|xtc')
    # parser.add_argument('--extra_keys', type=str, default='', help='comma separated list, e.g. "v", "f", "angle_idx", "dihedral_idx"')

def post_process_args(args):
    pass
    # args.extra_keys = list(filter(bool, args.extra_keys.split(',')))

class array2dict_clip(longclip):
    def dataset_postprocess(self, args, data, to_torch=True, **kwx):
        super().dataset_postprocess(args, data, **kwx)
        the_shape = tuple(self.flat[0].shape)
        self.flat = [Data.from_dict({"x":x, "cell_shape":the_shape}) for x in self.flat]

    def __getitem__(self, i):
        j = self.start_pos[i]
        return self.flat[j:j+self.tot_len_frame:self.nskip]


