#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# conda activate powerai
PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/DDD/lt32

device=0
for nlayer in 9; do 
for nlmlp in 2; do 
for buf in 1; do
    for N1 in 1; do
        for lr in 2e-3 4e-3; do # 1e-2 3e-3 1e-3; do
            for nfeat in 128; do #150 180 80 128
for symm in "" ; do #""
for act in mish swish; do #mish swish relu tanh
for threshold in 2e-3; do 
          # DIR=meshgraphnets/experiment/grain-nfeat${nfeat}_nlayer${nlayer}_symm${symm}_act${act}_threshold${threshold}_nlmlp${nlmlp}
            DIR=meshgraphnets/experiment/DDDlt32-nfeat${nfeat}_nlayer${nlayer}_symm${symm}_act${act}_nlmlp${nlmlp}_lr${lr}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR --dim=3 \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=9 --nfeat_out=9 --nfeat_latent=$nfeat --n_mpassing=$nlayer \
 --amr_N=32 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=$threshold --amr_eval_freq=5 \
 --mode=train \
 --num_training_steps=500000 --batch=1 --lr=$lr --lr_decay=200000 --cache --keep_ckpt=2 --valid_freq=4000 --rotate=$symm --mlp_activation=$act \
 --noise=0.02 --rollout_split=test --num_rollouts=10 &>>$DIR/log &
            device=$((device+1))
done
done
done
            done
            done
        done
    done
done
done
wait

# ipython -c "import pickle; import numpy as np;  np.save('pred.npy',np.linalg.norm(np.array([x['pred_velocity'] for x in pickle.load(open('$DIR/rollout.pkl', 'rb'))]),axis=-1,keepdims=True).reshape((10,-1,32,32,32,1)))"

