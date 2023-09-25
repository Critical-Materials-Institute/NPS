__author__ = 'Fei Zhou'

import os, glob
import numpy as np
from NPS_common.utils import load_array_auto

def register_args(parser):
    pass

def post_process_args(args):
    pass
    # if not os.path.exists(args.data_train):
    #     args.data_train += '.npy'
    # if not os.path.exists(args.data_valid):
    #     args.data_valid += '.npy'
    # if not os.path.exists(args.data_predict):
    #     args.data_predict += '.npy'

class longclip:
    def __init__(self, args, datf, tot_len, clip_step, nskip=1, split='train', **kwx):
        self.args = args
        self.dim = args.dim
        self.nskip = nskip
        self.is_train_set = (split == 'train')
        self.tot_len = tot_len
        self.tot_len_frame = (self.tot_len-1)*self.nskip + 1
        data= self.load_data(datf)
        data = self.dataset_preprocess(args, data, **kwx)
        self.nclip = len(data)
        # start_pos = [np.arange(0,self.clip_len-self.tot_len_frame+1,clip_step)+i*self.clip_len for i in range(self.nclip)]
        # self.start_pos = np.array(start_pos).ravel()
        cumsum = np.cumsum([0]+ [len(x) for x in data])
        start_pos = [np.arange(0, len(data[i])-self.tot_len_frame+1, clip_step) + cumsum[i] for i in range(self.nclip)]
        self.start_pos = np.concatenate(start_pos)
        self.dataset_postprocess(args, data, **kwx)

    def dataset_preprocess(self, args, data, **kwx):
        if args.data_slice:
            print('data slicing with', args.data_slice)
            f = eval(f'lambda x: x[{args.data_slice}]')
            data = [f(d) for d in data]
        if args.data_filter:
            print('data filtering with', args.data_filter)
            f = eval(f'lambda x: {args.data_filter}')
            data = [d[f(d)] for d in data]
        if args.data_preprocess:
            data = [self.preprocess(args.data_preprocess, d) for d in data]
        if data[0].ndim == self.dim+2:
            # no channel. Let's add a channel
            data = data[:,:,None] if not isinstance(data, list) else [d[:,:,None] for d in data]
        # channel_first = True
        # if channel_first:
        #     new_ord = list(range(data[0].ndim))
        #     nch = new_ord.pop(-1)
        #     new_ord.insert(len(new_ord)-self.dim, nch)
        #     data = [d.transpose(new_ord) for d in data]
        if args.space_CG:
            raise NotImplementedError()
            data_cg = []
            frame_shp0 = np.array(data.shape[3:])
            frame_shp = np.array(args.frame_shape)
            ncg = frame_shp0//frame_shp
            assert np.all(ncg*frame_shp == frame_shp0)
            shape_cg = list(data.shape[:3]) + list(np.stack([frame_shp, ncg],1).ravel())
            print('reshaping spatial shape to', frame_shp)
            axis_mean = tuple(range(4,len(shape_cg),2))
            shifts = np.array(np.meshgrid(*[np.arange(i) for i in ncg], indexing='ij')).reshape(self.dim,-1).T
            data= np.concatenate([np.mean(np.roll(data,shift,axis=tuple(range(3,3+self.dim))).reshape(shape_cg),axis=axis_mean) for shift in shifts])
        else:
            if len(args.frame_shape):
                assert np.array_equal(data[0].shape[2:-1], args.frame_shape), ValueError(f'mismatch {data[0].shape[2:-1]} {args.frame_shape}')
        if args.time_CG > 1:
            raise NotImplementedError()
            data_cg = []
            shp0 = np.array(data.shape[1:2])
            ncg = args.time_CG
            shp = shp0//ncg
            n0 = ncg*shp[0]
            shape_cg = list(data.shape[:1]) + [shp[0], ncg] + list(data.shape[2:])
            print('reshaping clip length to', shp[0])
            axis_mean = (2,)
            shifts = range(shp0[0]-n0+1) #np.array(np.meshgrid(*[np.arange(i) for i in ncg], indexing='ij')).reshape(self.dim,-1).T
            data= np.concatenate([np.mean(data[:,shift:shift+n0].reshape(shape_cg),axis=axis_mean) for shift in shifts])
        return [clip for d in data for clip in d]

    def dataset_postprocess(self, args, data, **kwx):
        self.flat = np.concatenate(data)
        if args.mean_std_in:
            self.flat = (self.flat - np.array(args.mean_std_in)[::2]) / np.array(args.mean_std_in)[1::2]
        if args.channel_first:
            self.flat = self.flat.transpose(0, 1+self.dim, *tuple(range(1,1+self.dim)))
        # self.flat = data.reshape((-1,)+data.shape[2:])
        # self.flat = np.stack([y for x in data for y in x])
        self.statistics = {}

    def _load_data(self, f):
        f += ('', '.npy', '.npz')[[os.path.exists(f+suffix) for suffix in ('', '.npy', '.npz')].index(True)]
        print(f' Loading npy array {f}')
        return load_array_auto(f)

    def load_data(self, f):
        suffix = 'npy'
        files = sorted(glob.glob(f+f'/*.{suffix}')) if os.path.isdir(f) else [f]
        if os.path.isdir(f) and os.path.exists(f'{f}.npy'):
            print(f'*** WARNING: *** found both directory {f} and file {f}.npy. Ignore the latter. ARE YOU SURE???')
        return [self._load_data(file) for file in files]

    def preprocess(self, pp_opt_str, a):
        print('data preprocessing with', pp_opt_str)
        if 'fft' not in pp_opt_str:
            try:
                from importlib import import_module
                p, m = pp_opt_str.rsplit('.', 1)
                mod = import_module(p)
                f = getattr(mod, m)
            except:
                f = eval(f'lambda x: {pp_opt_str}')
            return f(a)
        import ast
        pp_opt = ast.literal_eval(pp_opt_str)
        import sys
        sys.path.insert(0, '.')
        from NPS_common import smooth
        if pp_opt.get("name", "fft") == "fft":
            return smooth.smooth_array_fft_np(a, keep_frac=(pp_opt['tkeep'],)+((pp_opt['skeep'],)*(self.dim)), nbatch=1, array_only=True)

    def shuffle(self):
        np.random.shuffle(self.start_pos)

    def sample(self, nsample):
        nparts = len(self.start_pos)//nsample
        assert nparts>0, ValueError(f'ERROR nsample {nsample} too large, valid start points {len(self.start_pos)}')
        start = np.split(self.start_pos[:nparts*nsample], nparts)
        return np.array([[self.flat[i:i+self.tot_len_frame:self.nskip] for i in st] for st in start])

    def __getitem__(self, i):
        j = self.start_pos[i]
        # if self.i_in_out:
        #     return i, self.flat[j:j+self.n_in], self.flat[j+self.n_in:j+self.tot_len]
        return np.array(self.flat[j:j+self.tot_len_frame:self.nskip])

    def __len__(self):
        return len(self.start_pos)
