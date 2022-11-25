#!/bin/env python
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from NPS_common.utils import load_array

parser = argparse.ArgumentParser()
parser.add_argument("--ichannel", "-i", type=str, default='-1', help="which channels, -1 for all, -999 to flatten channels")
parser.add_argument("data", help="data file(s) (.npy or .npz)", nargs='+')
parser.add_argument("--bins", type=int, default=100, help="no. bins")
parser.add_argument("--kde", type=str, default=None, help="If set, use Gaussian kernel-desity estimate, e.g. scott, silverman or a number")
parser.add_argument("--FE", action='store_true', help="If set, plot -kB T log(P)")
parser.add_argument("--linear", action='store_true', help="Switching to linear scale plot (default is log)")
parser.add_argument("--d2", "--2d", action='store_true', help="hist2d")
options = parser.parse_args()
options.ichannel = {'-1':slice(0, None, 1), '-999':None}.get(options.ichannel, 'TBD') #np.array(list(map(int, options.ichannel.split(','))))
if options.kde:
    from scipy.stats import gaussian_kde
    if not options.kde in ('scott', 'silverman'):
        options.kde = float(options.kde)

dat_all = [load_array(x)[..., options.ichannel] for x in options.data]
nchannel = dat_all[0].shape[-1]
nplot= len(options.data)
ncol = nchannel+int(options.d2)
fig, axs = plt.subplots(nrows=nplot, ncols=ncol, figsize=(ncol*8, nplot*5), squeeze=False)

for i in range(nplot):
    for j in range(nchannel):
        ax = axs[i, j]
        dat = dat_all[i][...,j].ravel()
        nNaN = np.count_nonzero(np.isnan(dat))
        nInf = np.count_nonzero(np.isinf(dat))
        if nNaN+nInf:
            print(f'WARNING data {i} {j} has {nNaN} NaN {nInf} Inf')
            dat = dat[~np.logical_or(np.isnan(dat), np.isinf(dat))]
        if options.kde:
            kde = gaussian_kde(dat, options.kde)
            X = np.mgrid[dat.min():dat.max():options.bins*1j]
            if options.FE:
                ax.plot(X, -np.log(kde(X)))
                ax.set_ylabel('-kTlog(P)')
            else:
                ax.plot(X, kde(X))
        else:
            ax.hist(dat, bins=options.bins)
        if not options.linear: ax.set_yscale('log')
        ax.text(0.05, 0.95, f'mean= {np.mean(dat):.3g} std= {np.std(dat):.3g}', transform=ax.transAxes)
    if options.d2:
        ax = axs[i, nchannel]
        dat = dat_all[i][...,:2].reshape(-1, 2)
        ax.hist2d(*dat.T, bins=(options.bins,)*2, norm=mpl.colors.LogNorm())#, cmap=mpl.cm.gray)
        # if not options.linear: ax.set_yscale('log')

plt.show()


# # obtaining the pdf (my_pdf is a function!)
# my_pdf = gaussian_kde(samp)

# # plotting the result
# x = np.linspace(samp.min(), samp.max(),100)
# plot(x,my_pdf(x),'r') # distribution function
# hist(samp, bins=50, density=1,alpha=.3) # histogram
# show()
