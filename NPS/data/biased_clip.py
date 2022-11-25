__author__ = 'Fei Zhou'

import os
import numpy as np
from .longclip import longclip

def register_args(parser):
    pass

def post_process_args(args):
    pass

class biased_clip(longclip):
    def __init__(self, args, datf, *x, **kwx):
        self.is_train_set = (datf.endswith('train')) or (datf.endswith('train/')) or os.path.basename(datf).startswith('train')
        super().__init__(args, datf, *x, **kwx)

    def dataset_postprocess(self, args, data, **kwx):
        super().dataset_postprocess(args, data, **kwx)
        if self.is_train_set:
            dat_setting = eval(self.args.data_setting)
            bias_divisions = dat_setting['bias_divisions']
            assert len(bias_divisions) == self.flat.shape[-1]
            bias_divisions = [[-np.inf]+x+[np.inf] for x in bias_divisions]
            # start_flag = np.zeros(len(self.flat)).astype(bool)
            # start_flag[self.start_pos] = True
            flatdat = self.flat[self.start_pos]
            flatdat = dat_setting.get('bias_func', lambda x: x)(flatdat)
            div_flags = [[((flatdat[...,i]>=divs[j]) & (flatdat[...,i]<divs[j+1])) for j in range(len(divs)-1)] for i,divs in enumerate(bias_divisions)]
            # assert np.all([np.any(y) for x in div_flags for y in x])
            import itertools
            self.bin_indices = [np.where(np.all(flgs, 0))[0] for flgs in itertools.product(*div_flags)]
            print(f'  Points per bin: {list(map(len, self.bin_indices))}')
            self.bin_indices = list(filter(len, self.bin_indices))

    def biased_sample(self, i):
        frac = i/len(self.start_pos)
        ibin = int(frac*len(self.bin_indices))
        frac = frac*len(self.bin_indices) - ibin
        return self.bin_indices[ibin][int(frac*len(self.bin_indices[ibin]))]

    def __getitem__(self, i):
        j = self.start_pos[i if not self.is_train_set else self.biased_sample(i)]
        return np.array(self.flat[j:j+self.tot_len_frame:self.nskip])

