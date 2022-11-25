
import numpy as np
from models.common import laplacian_roll as Laplacian

class toy_data:
    def __init__(self, args, datf, train=True, name='typ'):
        self.dim = args.dim
        self.shape=args.frame_shape
        self.r = np.stack(np.meshgrid(*((np.arange(0, nnx) for nnx in self.shape))),axis=-1)
        GT_func = args.s1
        if GT_func:
            for s in GT_func.split('; '): exec(s, globals())
        self.len=4096
        self.visualize('GT.txt')

    def shuffle(self):
        pass

    def sample(self, nsample):
        return np.split(self.gen_train_dat(self.r, self.dim, self.shape, 4, self.len), self.len//nsample)

    @staticmethod
    def G0_func(c0): return 0.88*(0+0.5*c0-c0**2)

    @staticmethod
    def S_ij_func(a,b): return (abs(a)+abs(b)+abs(a*b))*0.9 + 0.25

    @staticmethod
    def gen_train_dat(r, dim=3, nx=[32,32], nmode=4, batch=4):
        c0 = np.random.uniform(-0.05, 1.05, [batch]+ nx).astype('float32')
        G = toy_data.G0_func(c0)
        H_ij = np.stack([toy_data.S_ij_func(c0, np.roll(c0,1,axis=i+1)) for i in range(dim)], -1)
        noise = np.random.randn(*(H_ij.shape)) * np.sqrt(H_ij)
        H= np.sum([noise[...,i] - np.roll(noise[...,i],-1,axis=i+1) for i in range(dim)], axis=0).astype('float32')
        return np.stack([c0, c0+Laplacian(G,1,dim,'np')+H], 1)[...,None]


    def visualize(self, fname):
        cgrid=np.arange(0, 1, 0.01, dtype='float32')
        gt_pf = toy_data.G0_func(cgrid)
        gt_noise = toy_data.S_ij_func(cgrid, cgrid)
        np.savetxt(fname, np.stack([cgrid, gt_pf, gt_noise]))
