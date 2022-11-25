#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from NPS_common.animateND import parse_cmd, process_parser

parser = parse_cmd()
for action in parser._actions:
    if action.dest == 'ichannel':
        action.default = '0:2'
options = parser.parse_args()
options.DIM = 0
options, data = process_parser(options)
if len(options.range) >=3 and options.range[2]:
    options.range.append(list(map(float, options.range[2].split(','))))
nplot = len(options.data)
fig, axs = plt.subplots(3, nplot, figsize=(nplot*4, 4), squeeze=False)
for i in range(nplot):
    for dat in data[i]:
        for j in range(dat.shape[1]):
            axs[j, i].plot(np.arange(len(dat[:,0])), dat[:, j])
        if dat.shape[1]>=2: axs[2, i].plot(dat[:,0], dat[:, 1])
        if len(options.range) >=3 and options.range[2]:
            axs[0, i].set_ylim(options.range[-1][:2])
            axs[1, i].set_ylim(options.range[-1][2:])
            axs[2, i].set_xlim(options.range[-1][:2])
            axs[2, i].set_ylim(options.range[-1][2:])


if options.o:
    fig.save(options.o, writer='imagemagick', fps=6)
else:
    plt.show()
