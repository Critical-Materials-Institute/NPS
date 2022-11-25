#!/bin/env python
#from IPython.display import HTML
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from NPS_common.utils import load_array

def parse_cmd():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--DIM", "-d", type=int, default=0, help="Dimension of array. DO NOT SET")
    parser.add_argument("--tskip", type=int, default=1, help="time skip")
    parser.add_argument("--tbegin", type=int, default=0, help="time begin")
    parser.add_argument("--tend", type=int, default=None, help="time end")
    parser.add_argument("-o", default='', help="save as gif")
    parser.add_argument("data", help="data file(s) (.npy or .npz)", nargs='+')
    parser.add_argument("--range", default='', help="Default '' to auto detect range; set to e.g. '0,1' to override; ',' to set to None")
    parser.add_argument("--ichannel", "-i", "-c", default='0', help="which channel to show, default 0 (single channel), 0:2 or 1:4 RGB multi-channel")
    parser.add_argument("--rv", action='store_true', help="invert RGB color")
    parser.add_argument("--cmap", default='hot', help="cmap of imshow")
    parser.add_argument("--channel_index", type=int, default=-1, help="channel position, -1 for channel last (default) (-999 means no channel)")
    parser.add_argument("--slice", type=str, default=':', help="e.g. ':', '...,-1'")
    parser.add_argument("--delay", type=int, default=25, help="Delay between frames")
    parser.add_argument("--tick", type=int, default=1, help="ticks on/off")
    parser.add_argument("--axis", type=int, default=1, help="axis/frame on/off")
    parser.add_argument("--nrow", type=int, default=1, help="N rows")
    parser.add_argument("--stamp", type=int, default=1, help="time stamp on/off")
    parser.add_argument("--spacing", type=str, default='', help="between subplots. e.g. '0,0' to remove spacing")
    return parser

def process_parser(options):
    if options.o: matplotlib.use('Agg')
    options.ichannel = list(map(int, options.ichannel.split(':')))
    if len(options.ichannel) == 1:
        options.ichannel = slice(options.ichannel[0], options.ichannel[0]+1)
        options.use_RGB = False
    else:
        options.ichannel = slice(*options.ichannel)#eval(f'slice({options.ichannel})')
        options.use_RGB = True
    if not options.tick:
        plt.rcParams.update({"xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False})
    if not options.axis:
        plt.rcParams.update({"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,})
    options.slice = eval(f'lambda x: x[{options.slice}]')

    # data
    nplot = len(options.data)
    data = []
    dat_minmax = []
    for i in range(nplot):
        data.append(load_array(options.data[i]).astype('float32'))
        data[i] = options.slice(data[i])
        if options.channel_index == -999:
            data[i] = data[i][...,None]
        elif options.channel_index >= 0:
            new_ax = list(range(0,options.channel_index))+list(range(options.channel_index+1,data[i].ndim))+[options.channel_index]
            data[i] = np.transpose(data[i], new_ax)
        data[i]=data[i][...,options.ichannel]
        if options.DIM > 0:
            data[i]=data[i][...,:3]
            if data[i].shape[-1] == 2:
                data[i] = np.concatenate((data[i], np.zeros_like(data[i])[...,:1]), -1)
            if (data[i].shape[-1] == 3) and options.rv:
                data[i] = 1 - data[i]
            data[i]=data[i].reshape((-1,)+data[i].shape[-(options.DIM+1):])[options.tbegin:options.tend:options.tskip]
            # dat_minmax.append([np.amin(data[i],(0,1,2)), np.amax(data[i],(0,1,2))])
        dat_minmax.append([np.amin(data[i]), np.amax(data[i])])
        print(options.data[i], 'value range', dat_minmax[i])
    dat_minmax=np.array(dat_minmax)
    if options.range:
        range_vals = options.range.split(',')
        allmin, allmax = [float(x) if x else None for x in range_vals][:2]
    else:
        allmin=np.amin(dat_minmax[:,0])
        allmax=np.amax(dat_minmax[:,1])
    options.range = [allmin, allmax, options.range]
    return options, data

def run_animation(anim, fig):
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

