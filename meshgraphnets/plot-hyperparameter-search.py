import numpy as np; import matplotlib.pyplot as plt                                                                                                                                      

dsets=['random', '\\', '/', 'X', '[]']
runs=['n=0.005', 'default', '\\', '/', 'X', '[]', 'n=0.04', 'n=0', '40k']
title='default random, noise=0.02, 20k steps'                                                                                                                                                                                      
d=np.loadtxt('tmp.txt').reshape((len(dsets), len(runs)))

fig, ax = plt.subplots()
im = ax.imshow(d.T)
ax.set_title(title)
ax.set_xticks(np.arange(len(dsets))); ax.set_xticklabels(dsets)
ax.set_yticks(np.arange(len(runs))); ax.set_yticklabels(runs)
for i in range(len(dsets)): 
    for j in range(len(runs)): 
        text = ax.text(i, j, '%.3f'%(d[i, j]),ha="center", va="center",color="w")
plt.show()
