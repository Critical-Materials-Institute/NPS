import numpy as np;
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt;
Nsim=1200
Nseq=1
ndiscard=0; nwindow=1; smooth=lambda acum: (acum[ndiscard+nwindow:]-acum[ndiscard:-nwindow])/nwindow;
# cpu_noamr=np.load('nvert_time_device_-1-dat2D256-amrnoAMR.npy')[...,:Nsim]
gpu_amr=np.load('nvert_time_device_3-dat2D512-amrAMR2-nrollout100-npred2000.npy')[...,:Nsim]
gpu_amr=np.mean(gpu_amr,axis=0,keepdims=True)
gpu_noamr=np.load('nvert_time_device_2-dat2D512-amrnoAMR-nrollout100-npred20.npy')[...,:Nsim]
gpu_noamr=np.mean(gpu_noamr,axis=0,keepdims=True)
print(f'debug gpu_amr {gpu_amr.shape} gpu_noamr {gpu_noamr.shape}')

fig, axs = plt.subplots(Nseq, 3, figsize=(10*1.5,4.5*Nseq))
if Nseq==1: axs=[axs]
for i, axs_i in enumerate(axs):
  axL1, axM1, axR = axs_i

  ndiscard=0; nwindow=5;
  axL1.set_xlabel('step')
  axL1.set_ylabel('Num. vertices', color='r')
  axL1.tick_params(axis='y', labelcolor='r')
  axL2 = axL1.twinx()
  axL2.set_ylim(0.13, 0.324)
  axL2.set_ylabel('time (s)', color='b')
  axL2.tick_params(axis='y', labelcolor='b')
  # ax1.plot(cpu_noamr[i,0],'r:',linewidth=3.2)
  # ax2.plot(smooth(cpu_noamr[i,1]).ravel()/10, 'b:', label='cpu no AMR (time/10)')
  axL1.plot([0, 400],[np.mean(gpu_noamr[i,0])]*2,'r--',linewidth=3.2, label='Num. vert. (no AMR)', dashes=(3, 1.2))
  axL2.plot([0, 400],[np.mean(smooth(gpu_noamr[i,1]))]*2, 'b--',label='time (no AMR)', dashes=(5, 2))
  axL1.plot(gpu_amr[i,0], color='r',linewidth=3.2, label='Num. vert. (AMR)')
  axL1.set_ylim(np.min(gpu_amr[i,0])*0.92-1.2e4,np.max(gpu_amr[i,0])*1.27-2.4e4)
  time_amr = smooth(gpu_amr[i,1])
  axL2.plot(time_amr, color='b', label='time (AMR)')
  axL1.legend(loc=1)
  axL2.legend(loc=5)

  ndiscard=0; nwindow=1;
  axM1.set_xlabel('step')
  axM1.set_ylabel('Num. vertices', color='r')
  axM1.tick_params(axis='y', labelcolor='r')
  axM2 = axM1.twinx()
  axM2.set_ylabel('time (s)', color='b')
  axM2.set_ylim(0.13, 0.324)
  axM2.tick_params(axis='y', labelcolor='b')
  # ax1.plot(cpu_noamr[i,0],'r:',linewidth=3.2)
  # ax2.plot(smooth(cpu_noamr[i,1]).ravel()/10, 'b:', label='cpu no AMR (time/10)')
  axM1.scatter(np.arange(len(gpu_amr[i,0])),gpu_amr[i,0],s=3, color='r',label='No. vert. (AMR)', marker='x')
  axM1.set_ylim(np.min(gpu_amr[i,0])*0.92-1.2e4,np.max(gpu_amr[i,0])*1.27-2.4e4)
  axM1.annotate('', xy=(-50, np.max(gpu_amr[i,0])*1.013), xytext=(80, np.max(gpu_amr[i,0])*1.013), arrowprops=dict(arrowstyle="->",color='r'))
  time_amr = smooth(gpu_amr[i,1])
  axM2.scatter(np.arange(len(time_amr)),time_amr, s=1.8,color='b', label='time (AMR)')
  axM2.annotate('', xy=(len(time_amr)*1.04,np.min(time_amr)*1.24), xytext=(len(time_amr)*0.93,np.min(time_amr)*1.24), arrowprops=dict(arrowstyle="->",color='b'))
  axM2.annotate('', xy=(len(time_amr)*1.04,np.min(time_amr)*0.98), xytext=(len(time_amr)*0.93,np.min(time_amr)*0.98), arrowprops=dict(arrowstyle="->",color='b'))
  axLInset = fig.add_axes([0.51,0.66,0.12,0.28],label=f'{i+100}')
  axLInset.scatter(np.arange(len(time_amr[:15])),time_amr[:15], s=32,color='b')
  axLInset.set_xticks([0,5,10])
  axLInset.grid(axis = 'x', which='both')
  npart_amr = gpu_amr[i,0][ndiscard+nwindow//2:-nwindow//2]

  axR.scatter(npart_amr, time_amr, s=1.8, color='k')
  axR.set_xlabel('Num. vertices')
  axR.set_ylabel('time (s)')
  axR.set_ylim(0.13, 0.324)
  fig.tight_layout()
  # inset
  axRInset = fig.add_axes([0.774,0.66,0.11,0.28],label=f'{i}')
  axRInset.set_xlim(0,  np.max(npart_amr)*1.01)
  axRInset.set_ylim(0,  np.max(time_amr)*1.01)
  axRInset.scatter(npart_amr, time_amr, color='k',s=18.8)
  x=np.array([0,  np.max(npart_amr)*1.01])
#   m, b= np.polyfit(npart_amr[::5], time_amr[::5], 1)
  m, b= np.polyfit(npart_amr, time_amr, 1)
  print(f'debug m= {m} b= {b}')
  axRInset.plot(x,m*x+b,'y--')
#   print(f'debug AMR steps m2= {m2} m= {m} b= {b}')
#   axRInset.plot(x,m2*x**2+m*x+b,'y-')
#   m, b = np.polyfit(np.reshape(npart_amr[:5*(len(npart_amr)//5)],(-1,5))[:,1:].ravel(), np.reshape(time_amr[:5*(len(npart_amr)//5)],(-1,5))[:,1:].ravel(), 1)
#   print(f'debug normal steps m= {m} b= {b}')
#   axRInset.plot(x,m*x+b,'y--')
  # if i==0: plt.legend();#ax2.legend(); ax1.legend()
plt.show()

import glob
fig, ax = plt.subplots(figsize=(6,4.5*1))
sizes = (256, 512, 1024, 2048)
gpu_timing = [np.load(glob.glob( f'nvert_time_device_2-dat2D{d}-amrnoAMR-nrollout*-npred20.npy')[0]) for d in sizes[:3]]
cpu_timing = [np.load(glob.glob(f'nvert_time_device_-1-dat2D{d}-amrnoAMR-nrollout*-npred20.npy')[0]) for d in sizes[:4]]
# print('debug', gpu_timing.shape, cpu_timing.shape, cpu_timing)
ax.plot(sizes[:4], [np.mean(x[:,1,2:]-x[:,1,1:-1]) for x in cpu_timing], label='CPU', marker='+')
ax.plot(sizes[:3], [np.mean(x[:,1,2:]-x[:,1,1:-1]) for x in gpu_timing], label='GPU', marker='o')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('linear size')
ax.set_ylabel('time (s)')
ax.set_xticks(sizes)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
ax.legend()
plt.show()



# fig1, ax1 = plt.subplots()
# ax1.plot(sizes[:2], np.mean(gpu_timing[:,:,1,1:]-gpu_timing[:,:,1,0:-1],axis=(1,2)), label='GPU', marker='o')
# # ax1.plot(sizes[:1], np.mean(cpu_timing[:,:,1,1:]-cpu_timing[:,:,1,0:-1],axis=(1,2)), label='CPU', marker='x')
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xticks(sizes)
# ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# ax1.get_xaxis().set_tick_params(which='minor', size=0)
# ax1.get_xaxis().set_tick_params(which='minor', width=0) 

# plt.show()
