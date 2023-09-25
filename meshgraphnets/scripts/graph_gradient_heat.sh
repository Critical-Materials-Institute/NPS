#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# conda activate powerai
PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/heat-2d-toy

device=2
nlayer=2
DIR=meshgraphnets/experiment/heat-2d-toy
mkdir -p $DIR
CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --core=graph_gradient_model --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --nfeat_latent=1 --n_mpassing=$nlayer \
 --mode=predict \
 --rollout_split=test --num_rollouts=4 --rollout_path=$DIR/rollout.pkl --num_predict_steps=2 #&>>$DIR/log
            device=$((device+1))

#  --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=$threshold --amr_eval_freq=5 \
#  --num_training_steps=500000 --batch=8 --lr=$lr --lr_decay=70000 --cache --keep_ckpt=1 --valid_freq=8000 --rotate=$symm --mlp_activation=$act  --noise=0.02 \

$(python <<EOF
import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
import pickle
import scipy.signal

with open('meshgraphnets/experiment/heat-2d-toy/rollout.pkl', 'rb') as fp: a= pickle.load(fp); b=a[1]
fig,ax=plt.subplots(1,5,figsize=(15,3)); 
f0= b['pred_velocity'][0].reshape((64,64))
ax[0].matshow(f0);
ax[1].matshow(scipy.signal.convolve2d(f0,[[0,0,0],[0,-1,1],[0,0,0]],boundary='wrap',mode='same'));
ax[2].matshow(scipy.signal.convolve2d(f0,[[0,0,0],[1,-2,1],[0,0,0]],boundary='wrap',mode='same'));
ax[3].matshow(scipy.signal.convolve2d(f0,[[0,1,0],[1,-4,1],[0,1,0]],boundary='wrap',mode='same'));
ax[-1].matshow(b['pred_velocity'][1].reshape((64,64))); 
plt.show()
EOF
)
