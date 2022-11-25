import numpy as np


def smooth_array_fft_np(a, keep_frac=(0.1,0.2,0.2,0.2), nbatch=0, array_only=True):
# a = a[:20]
    s = a.shape
    dim = len(keep_frac)
    axes = list(range(nbatch,nbatch+dim))
    a_f = np.fft.fftn(a, axes=axes)
    for i in range(dim):
        nkeep = int(s[nbatch+i]*keep_frac[i])
        a_f[(slice(None),)*(nbatch+i) + (slice(nkeep+1,-nkeep if nkeep>0 else s[nbatch+i]),)] = 0
        # print(f'debug i {i} nkeep {nkeep} slice {(slice(None),)*nbatch + (slice(nkeep+1,-nkeep),)}')
    ap = np.fft.ifftn(a_f, axes=axes).real
    # print('FT shape', a_f.shape, "out shape", ap.shape)
    print('FFT smoothing RMSE', np.sqrt(np.mean((a-ap)**2)), 'MAE', np.mean(np.abs(a-ap)))
    if array_only:
        return ap
    else:
        freq = np.stack(np.meshgrid(*[np.fft.fftfreq(i) for i in s[nbatch:nbatch+dim]], sparse=False, indexing='ij'), -1)
        # print(f'freq shape {freq.shape}')
        return ap, a_f, freq


def derivative_fft_np(a_f, freq, dx_index, axes=(1,2,3,4)):
    return np.fft.ifftn(2j*np.pi*a_f* freq[None,...,dx_index,None], axes=axes).real


def gradient_fft_np(a_f, freq, dx_indices, axes=(1,2,3,4)):
    return np.fft.ifftn(2j*np.pi*a_f* freq[None,...,dx_indices], axes=axes).real


def laplacian_fft_np(a_f, freq, dx_indices, axes=(1,2,3,4)):
    return np.fft.ifftn(-(2*np.pi)**2 *a_f* np.sum(freq[None,...,dx_indices]**2,axis=-1,keepdims=True), axes=axes).real


if __name__ == '__main__':
    # a = np.load('/g/g90/zhou6/data/LJ20new/valid.npy')
    # smooth_array_fft_np(a, nbatch=1)
    if False:
        print('testing derivative')
        import matplotlib.pyplot as plt
        t, x = np.mgrid[-10:11, -21:22]
        # a = np.stack([t**2*x, np.exp(-t/5)*x**2, np.exp(-t/5)*x**3, np.exp(-t/15)*np.sin(x**2/10), np.cos(-np.sqrt(t)/0.5)*np.sin(x**2/40)],0)
        a = np.stack([np.sin(t*0.9), np.cos(x*0.7), np.exp(-(t**2*0.05 + x**2*0.013 + t*x*0.018)), np.exp((t*1.5-x*0.9)/15), np.cos(-t*0.7-x*0.2)],0)
        dim=2
        a = (a/np.linalg.norm(a,axis=(1,2),keepdims=True))[..., None]
        a+= np.random.randn(*a.shape)*2e-3
        # dadt_analytic = np.stack([2*(t-10)*x, (-1/5)*np.exp(-t/5)*x**2, (-1/5)*np.exp(-t/5)*x**3, (-1/15)*np.exp(-t/15)*np.sin(x**2/10), np.sin(np.sqrt(t)/0.5)*(-1/np.sqrt(t)/0.25)*np.sin(x**2/40)],0)[..., None]
        ap, a_f, freq = smooth_array_fft_np(a, keep_frac=(0.2,0.2), nbatch=1, array_only=False)
        # dadt_diff = a[:,1:]-a[:,:-1]
        # dadt_diff = np.concatenate([dadt_diff[:,:1], (dadt_diff[:,1:]+dadt_diff[:,:-1])/2, dadt_diff[:,-1:]],1)
        dadt_diff = (np.roll(a,-1,1)-np.roll(a,1,1))/2
        lapl_diff = np.roll(a,-1,2)+np.roll(a,1,2)-2*a
        # dadt_fft = np.fft.ifftn(1j*((2*np.pi)**(dim/2))*a_f* freq[None,...,0,None], axes=(1,2)).real
        dadt_fft = derivative_fft_np(a_f, freq, 0, axes=(1,2))
        lapl_fft = laplacian_fft_np(a_f, freq, slice(1,2), axes=(1,2))
        print(f'da/dt diff {np.sqrt(np.mean((dadt_diff-dadt_fft)**2))}')
        fig, axs = plt.subplots(8,len(a))
        for i in range(len(a)):
            axs[0,i].matshow(a[i,...,0])
            axs[0,i].set_axis_off()
            axs[1,i].matshow(ap[i,...,0])
            axs[1,i].set_axis_off()
            axs[2,i].matshow(dadt_diff[i,...,0])
            axs[2,i].set_axis_off()
            axs[3,i].matshow(dadt_fft[i,...,0])
            axs[3,i].set_axis_off()
            axs[4,i].matshow(np.abs(a_f[i,...,0])**0.15)
            axs[4,i].set_axis_off()
            axs[5,i].scatter(dadt_fft[i].ravel(), dadt_diff[i].ravel())
            # axs[5,i].set_axis_off()
            axs[6,i].matshow(lapl_diff[i,...,0])
            axs[6,i].set_axis_off()
            axs[7,i].matshow(lapl_fft[i,...,0])
            axs[7,i].set_axis_off()
        plt.show()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('arr', type=str, default='', help='input array')
    parser.add_argument('-o', type=str, default='', help='output array')
    parser.add_argument('--tkeep', type=float, default=0.2, help='how many temporal Fourier components to keep')
    parser.add_argument('--skeep', type=float, default=0.2, help='how many spatial Fourier components to keep')
    args = parser.parse_args()
    a = np.load(args.arr)
    dim = a.ndim-2
    ap = smooth_array_fft_np(a, keep_frac=(args.tkeep,)+((args.skeep,)*(dim-1)), nbatch=1, array_only=True)
    if args.o:
        np.save(args.o, ap)
