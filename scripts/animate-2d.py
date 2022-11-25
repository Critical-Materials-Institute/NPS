#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from NPS_common.animateND import parse_cmd, process_parser, run_animation

parser = parse_cmd()
parser.add_argument("--interp", type=str, default='antialiased', help="Interpolation method: antialiased, nearest, etc")
options = parser.parse_args()
options.DIM = 2
options, data = process_parser(options)
nplot = len(options.data)
nrow = options.nrow
ncol = int(np.ceil(nplot/nrow))
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*4), squeeze=False,\
    **({"gridspec_kw": {'wspace':float(options.spacing.split(',')[0]), 'hspace':float(options.spacing.split(',')[1])}} if options.spacing else {}))
axs = axs.ravel()

ims=[]
for i in range(nplot):
    ax = axs[i]
    # note using global vmin, vmax
    ims.append(ax.imshow(data[i][0,:,:], cmap=plt.get_cmap(options.cmap), vmin=options.range[0], vmax=options.range[1], interpolation=options.interp))
#        fig.colorbar(ims[i], ax=ax)
    ax.set_xlim((0, data[i].shape[-2]))
    ax.set_ylim((0, data[i].shape[-3]))
    if options.stamp: ax.set_title('0')
    if not options.axis: ax.set_axis_off()

# animation function. This is called sequentially
def animate(t):
    for i in range(nplot):
        ims[i].set_data(data[i][t])
        print('step t', t)
        ax = axs[i]
        if options.stamp: ax.set_title(str(t))
    return ims

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, frames=len(data[0]), interval=options.delay, blit=True)

run_animation(anim, fig)

if options.o:
    anim.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()
