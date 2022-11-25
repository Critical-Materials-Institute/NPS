__author__ = 'Fei Zhou'

import numpy as np
import torchvision
from .longclip import longclip

class single_image(longclip):
    def load(self, fn):
        import ast
        setting = ast.literal_eval(self.configs.model_setting)
        offset= setting.get('splitoffset', 100)
        data = np.load(fn)[:,6:-6,6:-6,:]
        if data.ndim==self.dim+1:
            data=data[...,None]
        if self.train and False:
            s_in = data.shape[1:-1]
            shape= np.array(self.args.frame_shape)
            #assert self.dim==2, 'only implemented for 2d'
            from itertools import chain
            istarts = chain.from_iterable(*[list(range(0,s_in[i_d]-shape[i_d],offset))+[s_in[i_d]-shape[i_d]] for i_d in range(self.dim)])
            data = np.stack([data[:,i[0]:i[0]+shape[0], i[1]:i[1]+shape[1]] for i in i_starts])
        data = np.stack([data,data],axis=1)
        if self.train:
            self.data_aug = torchvision.transforms.RandomAffine(90, translate=None, scale=(0.9,1.1), resample=0, fillcolor=1)
        else:
            self.data_aug = lambda x: x
        return data

    def __getitem__(self, i):
        return self.data_aug(np.array(self.flat[i:i+self.tot_len]))
