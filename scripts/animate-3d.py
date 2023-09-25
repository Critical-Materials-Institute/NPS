#!/bin/env python
#from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mpl_toolkits.mplot3d import Axes3D
# from skimage import measure 
# import argparse
from NPS_common.utils import load_array, str2slice
from NPS_common.animateND import parse_cmd, process_parser, setup_plots, run_animation

# parser = argparse.ArgumentParser()
parser = parse_cmd()
parser.add_argument("--type", default='slice2d', help="slice2d | slice | iso | all | hist | hist_map")
parser.add_argument("--iso_val", type=float, default=0.5, help="isosurface value")
parser.add_argument("--index2d", default='z=0', help="which 2d slice, e.g. z=0 slice. Comma separates multiple values, e.g. animate-3d.py a.npy a.npy --index2d 'z=0,x=50'")
parser.add_argument("--size", type=int, default=32, help="array size of flat .bin")
parser.add_argument("--nbins", type=int, default=64, help="no. bins for histogram (type=hist)")
options = parser.parse_args()
options.DIM = 3
options, data = process_parser(options)
(allmin, allmax) = options.range[:2]
nframe = len(data[0])
nplot = len(options.data)
nrow = options.nrow
ncol = int(np.ceil(nplot/nrow))
if options.type in ('slice', 'iso'):
    for i in range(len(data)):
        if len(data[i].shape) == 5:
            assert data[i].shape[-1] == 1
            data[i] = data[i][..., 0]

options.index2d = list(filter(bool, options.index2d.split(',')))
if len(options.index2d) == 1: options.index2d *= nplot
options.index2d= [[x.split('=')[0], int(x.split('=')[1])] for x in options.index2d]
# options.ichannel = list(map(str2slice, filter(bool, options.ichannel.split(','))))
# if len(options.ichannel) == 1: options.ichannel *= nplot
fig = plt.figure(figsize=plt.figaspect(nrow/ncol))
if options.type in ('slice2d', 'hist', 'hist_map'):
    axs=[fig.add_subplot(nrow, ncol, i+1) for i in range(nplot)]
else:
    from mpl_toolkits.mplot3d import Axes3D
    axs=[fig.add_subplot(nrow, ncol, i+1, projection='3d') for i in range(nplot)]
    for ax in axs:
        ax.view_init(azim=30)
# assert options.type == 'slice2d' or all([isinstance(x, int) for x in options.ichannel]), ValueError(f'Color output with slicing only available in the slice2d mode')
# fig, axs = setup_plots(options)

# data=[]
# dat_minmax = []
# for i in range(nplot):
#     data.append(load_array(options.data[i]).astype('float32'))
#     if options.channel_index == -999:
#         data[i] = data[i][...,None]
#     elif options.channel_index == -1:
#         pass
#     elif options.channel_index >= 0:
#         new_ax = list(range(0,options.channel_index))+list(range(options.channel_index+1,data[i].ndim))+[options.channel_index]
#         data[i] = np.transpose(data[i], new_ax)
#     else:
#         raise f"Unknown channel_index {options.channel_index}"
#     data[i] = data[i].reshape((-1,)+data[i].shape[-DIM-1:])
#     data[i] = data[i][::options.tskip, ..., options.ichannel[i]]
#     if np.any(np.isnan(data[i])):
#         print("WARNING NAN encountered")
#         np.nan_to_num(data[i], False)
#     print(options.data[i], 'value range', np.min(data[i]), np.max(data[i]))
#     if options.type not in ('slice2d', 'hist', 'hist_map'):
#         axs[i].set_xlim((0, data[i].shape[1]))
#         axs[i].set_ylim((0, data[i].shape[2]))
#         axs[i].set_zlim((0, data[i].shape[3]))
#         axs[i].view_init(25, 65)
#     dat_minmax.append([np.min(data[i]), np.max(data[i])])
# #data = np.array(data)
# dat_minmax=np.array(dat_minmax)
# if options.range:
#     allmin = float(options.range.split(',')[0])
#     allmax = float(options.range.split(',')[1])
# else:
#     allmin=np.min(dat_minmax[:,0])
#     allmax=np.max(dat_minmax[:,1])
# print('Overall min max', allmin, allmax)
# nframe = len(data[0])
# for i in range(nplot):
#     if data[i].shape[-1] == 3:
#         print(f'3 color channels in data {i}, normalizing for color display')
#         data[i] = (data[i]-allmin)/(allmax-allmin)

def plot_3D_slice(array, ax):
    pic = []
    min_val = array.min()
    max_val = array.max()
    n_x, n_y, n_z = array.shape
    colormap = plt.cm.YlOrRd
    nx0=ny0=nz0=0

    x_cut = array[nx0,:,:]
    Y, Z = np.mgrid[0:n_y, 0:n_z]
    X = nx0 * np.ones((n_y, n_z))
    pic.append(ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((x_cut-min_val)/(max_val-min_val)), shade=False))
    #ax.set_title("x slice")

    y_cut = array[:,ny0,:]
    X, Z = np.mgrid[0:n_x, 0:n_z]
    Y = ny0 * np.ones((n_x, n_z))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    pic.append(ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((y_cut-min_val)/(max_val-min_val)), shade=False))
    #ax.set_title("y slice")

    z_cut = array[:,:,nz0]
    X, Y = np.mgrid[0:n_x, 0:n_y]
    Z = nz0 * np.ones((n_x, n_y))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    pic.append(ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colormap((z_cut-min_val)/(max_val-min_val)), shade=False))
    #ax.set_title("z slice")
    return pic
    #plt.show()


def plot_3D_iso(arr, ax):
    try:
        from skimage import measure 
        verts, faces, _, _ = measure.marching_cubes_lewiner(arr, options.iso_val)
        return [ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], color=(0,0,0,0.5), lw=1)]
    except:
        return [ax.scatter([0], [0])]


def frame_slice(frm, xyz, idx):
    if xyz == 'x':
        return frm[idx,:,:]
    elif xyz == 'y':
        return frm[:,idx,:]
    elif xyz == 'z':
        return frm[:,:,idx]


if options.type == 'slice':
    plotfunc=lambda t: [plot_3D_slice(data[i][t], axs[i]) for i in range(nplot)]
elif options.type == 'iso':
    plotfunc=lambda t: [plot_3D_iso(data[i][t], axs[i]) for i in range(nplot)]
elif options.type == 'hist' or options.type == 'hist_map':
    bins=np.linspace(allmin-0.001, allmax+0.0001, options.nbins)
    _hist=lambda t: [np.histogram(data[i][t].ravel(), bins=bins)[0] for i in range(nplot)]
    plots_last = _hist(-1)
    plots_first = _hist(0)
    ymax=[max(plots_first[i].max(), plots_last[i].max()) for i in range(nplot)]
    if options.type == 'hist':
        plotfunc=lambda t: [(axs[i].hist(data[i][t].ravel(), range=(allmin, allmax), bins=30), 
          axs[i].set_xlim(allmin, allmax), axs[i].set_ylim(0, ymax[i])) for i in range(nplot)]
    elif options.type == 'hist_map':
        options.delay=999
        counts_dat = np.log(np.transpose(np.array([_hist(t) for t in range(nframe)]), (1,2,0)))
        nframe_ = nframe
        plotfunc=lambda t: [axs[i].imshow(counts_dat[i], origin='lower', aspect='auto', 
          interpolation=None, extent=(1,nframe_, allmin, allmax), cmap='gnuplot') for i in range(nplot)]#,
        nframe=1
        options.delay=99999
elif options.type == 'all':
    plotfunc=lambda t: [plot_3D_iso(data[i][t], axs[i]) + plot_3D_slice(data[i][t], axs[i]) for i in range(nplot)]
elif options.type == 'slice2d':
    plotfunc=lambda t: [axs[i].imshow(frame_slice(data[i][t],*options.index2d[i]), cmap=plt.get_cmap('hot'), vmin=allmin, vmax=allmax) for i in range(nplot)]
else:
    raise ValueError("unknown plotting type %s"%options.type)

plots = plotfunc(0)
#plt.show()
# animation function. This is called sequentially
def animate(t):
    if options.type == 'slice2d':
        for i in range(nplot):
            plots[i].set_data(frame_slice(data[i][t],*options.index2d[i]))
    elif options.type == 'hist' or options.type == 'hist_map':
        for i in range(nplot):
            axs[i].cla()
        plotfunc(t)
        #for i in range(nplot):
        #    axs[i].cla()
        #    plots[i] = newfigs[i]
    else:
        newfigs = plotfunc(t)
        for i in range(nplot):
            for j in range(len(plots[i])):
                plots[i][j].remove()
                plots[i][j] = newfigs[i][j]
    print('    step t', t)
    return plots

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=nframe, interval=options.delay, blit=(options.type=='slice2d'))

#run_animation(anim, fig)
if options.o:
    anim.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()
