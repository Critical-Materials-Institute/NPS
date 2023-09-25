__author__ = 'Fei Zhou'

import os, glob
import numpy as np
from NPS_common.utils import load_array_auto
from .longclip import longclip

def register_args(parser):
    parser.add_argument('--vartime', default='1,1000', help='min, max variable time step')
    parser.add_argument('--vartime_scale', type=float, default=1000.0, help='time variable = timestep/scale')
    parser.add_argument('--vartime_test', default='1000,1000', help='min, max variable time step')

def post_process_args(args):
    args.vartime = list(map(int, args.vartime.split(',')))
    args.vartime_test = list(map(int, args.vartime_test.split(',')))

class vartime_clip(longclip):
    def __init__(self, *x, **kwx):
        super().__init__(*x, **kwx)
        self.vartime = self.args.vartime if self.is_train_set else self.args.vartime_test
        self.vartime_scale = self.args.vartime_scale

    def __getitem__(self, i):
        j = self.start_pos[i]
        vartime = np.random.randint(self.vartime[0], self.vartime[-1]+1)
        return np.concatenate((np.array(self.flat[j:j+vartime*self.tot_len:vartime]), np.full((self.tot_len,1), vartime/self.vartime_scale, np.float32)), axis=-1)
