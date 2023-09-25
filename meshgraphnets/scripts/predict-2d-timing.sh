#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# prep data
for m in 4 8 16 32; do
  N_L=$((64*m))
  DAT=$HOME/data/grain/2D${N_L}
  calculator_array.py  "np.tile( $HOME/data/grain/test.npy [:,:3],(1,1,$m,$m,1)).astype(np.float32)" -o $DAT/test.npy
  python meshgraphnets/npy2tfrecord.py $DAT/test.npy -o $DAT/test --periodic
done

# conda activate powerai
PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/grain/square_justNN_noduplicate

device=0
counter=0
nrollout=100
for DIR in \
  meshgraphnets/experiment/grain-nfeat48_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr1e-3 ; do
#  meshgraphnets/experiment/grain-nfeat32_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr3e-3 \
#  meshgraphnets/experiment/grain-nfeat64_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr8e-4 ; do
for device in 3 ; do
for dat in 2D512 ; do
N_L=${dat#2D}
DAT=$HOME/data/grain/$dat
for npred in 400; do #200 400 600 1000 1400 2000
for N1 in 1; do
if [[ $N1 == 1 ]]; then amr=noAMR; else amr=AMR2; fi
for timing in 1 ; do #""
counter=$((counter+1))
tag=device_${device}-dat${dat}-amr${amr}-nrollout${nrollout}-npred${npred}
PKL=tmp.pkl
TEST_TIMING_ONLY=$timing CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 nohup $PYTHON `grep mode=train $DIR/command.txt|tail -n1|sed "s/.*model.py/-m meshgraphnets.run_model/;s:dataset_dir=\S*:dataset_dir=$DAT:;s/--num_rollouts=[0-9]*/--num_rollouts=$nrollout --num_predict_steps=$npred --rollout_path=$PKL/;s/--mode=[A-z]*/--mode=predict/;s/--amr_N1=[0-9]*/--amr_N1=$N1/;s/amr_threshold=\S*/amr_threshold=2e-3/;s/amr_N=\S*/amr_N=$N_L/"` 
if [[ $timing == 1 ]]; then
  python -c "import pickle; import numpy as np; np.save('nvert_time_${tag}.npy',np.stack([np.stack((np.array(x['mesh_pos'],dtype=float), (np.array(x['pred_velocity']).ravel()-x['pred_velocity'][0]))) for x in pickle.load(open('$PKL', 'rb'))]))"
fi
#&> device$d.txt
#device=$(((device+1)%4))
done
done
done
done
# if [[ $device == 3 ]]; then wait; fi
done
done

result=$(python <<EOF
import numpy as np;
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt;
Nsim=1000
Nseq=8
ndiscard=1; nwindow=10; smooth=lambda acum: (acum[ndiscard+nwindow:]-acum[ndiscard:-nwindow])/nwindow;
cpu_noamr=np.load('nvert_time_device_-1-dat2D256-amrnoAMR.npy')[...,:Nsim]
gpu_amr=np.load('nvert_time_device_3-dat2D512-amrAMR2-nrollout100-npred2000.npy')[...,:Nsim]
gpu_noamr=np.load('nvert_time_device_3-dat2D256-amrnoAMR.npy')[...,:Nsim]

fig, axs = plt.subplots(Nseq, 2, figsize=(10,4.5*Nseq))
for i, axs_i in enumerate(axs):
  ax1, axR = axs_i
  ax1.set_xlabel('step')
  ax1.set_ylabel('Num. vertices', color='r')
  ax1.tick_params(axis='y', labelcolor='r')
  ax2 = ax1.twinx()
  ax2.set_ylabel('time (s)', color='b')
  ax2.tick_params(axis='y', labelcolor='b')
  # ax1.plot(cpu_noamr[i,0],'r:',linewidth=3.2)
  # ax2.plot(smooth(cpu_noamr[i,1]).ravel()/10, 'b:', label='cpu no AMR (time/10)')
  # ax1.plot(gpu_noamr[i,0],'r--',linewidth=3.2, dashes=(5, 4))
  # ax2.plot(smooth(gpu_noamr[i,1]), 'b--',label='gpu no AMR', dashes=(5, 5))
  ax1.plot(gpu_amr[i,0], 'r-',linewidth=3.2)
  time_amr = smooth(gpu_amr[i,1])
  ax2.plot(time_amr, 'b-', label='gpu AMR');
  npart_amr = gpu_amr[i,0][ndiscard+nwindow//2:-nwindow//2]
  axR.scatter(npart_amr, time_amr, s=2.1, color='b')
  axR.set_xlabel('Num. vertices')
  axR.set_ylabel('time (s)')
  fig.tight_layout()
  # inset
  axInset = fig.add_axes([0.65,0.64,0.16,0.31],label=f'{i}')
  axInset.set_xlim(0,  256**2 *1.05)
  axInset.set_ylim(0,  0.075)
  axInset.scatter(npart_amr, time_amr, color='b')
  m, b = np.polyfit(npart_amr, time_amr, 1)
  print(f'debug m= {m} b= {b}')
  x=np.array([0,  256**2 *1.05])
  axInset.plot(x,m*x+b,'k-')
  if i==0: ax2.legend()
plt.show();
EOF
)

result=$(python <<EOF
import numpy as np;
# import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt;
Nsim=1000
Nseq=1
ndiscard=0; nwindow=1; smooth=lambda acum: (acum[ndiscard+nwindow:]-acum[ndiscard:-nwindow])/nwindow;
# cpu_noamr=np.load('nvert_time_device_-1-dat2D256-amrnoAMR.npy')[...,:Nsim]
gpu_amr=np.load('nvert_time_device_3-dat2D512-amrAMR2-nrollout100-npred2000.npy')[...,:Nsim]
gpu_amr=np.mean(gpu_amr,axis=0,keepdims=True)
# gpu_noamr=np.load('nvert_time_device_3-dat2D256-amrnoAMR.npy')[...,:Nsim]

fig, axs = plt.subplots(Nseq, 2, figsize=(10,4.5*Nseq))
if Nseq==1: axs=[axs]
for i, axs_i in enumerate(axs):
  ax1, axR = axs_i
  ax1.set_xlabel('step')
  ax1.set_ylabel('Num. vertices', color='r')
  ax1.tick_params(axis='y', labelcolor='r')
  ax2 = ax1.twinx()
  ax2.set_ylabel('time (s)', color='b')
  ax2.tick_params(axis='y', labelcolor='b')
  # ax1.plot(cpu_noamr[i,0],'r:',linewidth=3.2)
  # ax2.plot(smooth(cpu_noamr[i,1]).ravel()/10, 'b:', label='cpu no AMR (time/10)')
  # ax1.plot(gpu_noamr[i,0],'r--',linewidth=3.2, dashes=(5, 4))
  # ax2.plot(smooth(gpu_noamr[i,1]), 'b--',label='gpu no AMR', dashes=(5, 5))
  ax1.scatter(np.arange(len(gpu_amr[i,0])),gpu_amr[i,0],s=2.1, color='r',linewidth=3.2)
  ax1.set_ylim(np.min(gpu_amr[i,0])*0.98,np.max(gpu_amr[i,0])*1.15,)
  time_amr = smooth(gpu_amr[i,1])
  ax2.scatter(np.arange(len(time_amr)),time_amr, color='b', label='gpu AMR');
  npart_amr = gpu_amr[i,0][ndiscard+nwindow//2:-nwindow//2]
  axR.scatter(npart_amr, time_amr, s=2.1, color='b')
  axR.set_xlabel('Num. vertices')
  axR.set_ylabel('time (s)')
  fig.tight_layout()
  # inset
  axInset = fig.add_axes([0.65,0.64,0.16,0.31],label=f'{i}')
  axInset.set_xlim(0,  256**2 *1.05)
  axInset.set_ylim(0,  0.075)
  axInset.scatter(npart_amr, time_amr, color='b')
  m, b = np.polyfit(npart_amr, time_amr, 1)
  print(f'debug m= {m} b= {b}')
  x=np.array([0,  256**2 *1.05])
  axInset.plot(x,m*x+b,'k-')
  if i==0: ax2.legend()
plt.show();
EOF
)
