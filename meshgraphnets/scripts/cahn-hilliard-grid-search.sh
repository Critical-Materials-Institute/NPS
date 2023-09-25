#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/TF1/bin/python
DAT=$HOME/data/cahn-hilliard-SOC-third

device=0
for core in diffusion; do # diffusion core_model
for nlayer in 14 ; do
for nlmlp in 2; do
for buf in 1; do
for noise in 1e-3 1e-4 1e-6 1e-8; do
    for N1 in 1; do
        for lr in  3e-3 ; do # 1e-2 3e-3 1e-3; do
            for nfeat in 96; do #150 180 80 128
for symm in cubic ; do #""
for act in mish; do #mish swish relu tanh
for threshold in 2e-3; do 
            DIR=meshgraphnets/experiment/cahn-nfeat${nfeat}_nlayer${nlayer}_symm${symm}_act${act}_threshold${threshold}_nlmlp${nlmlp}_lr${lr}_N1${N1}_core${core}_noise${noise}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --core=$core \
 --model=NPS --unique_op=false --periodic=0 --nfeat_in=1 --nfeat_out=1 --nfeat_latent=$nfeat --n_mpassing=$nlayer \
 --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=$threshold --amr_eval_freq=5 \
 --mode=train \
 --num_training_steps=500000 --batch=8 --lr=$lr --lr_decay=70000 --cache --keep_ckpt=1 --valid_freq=8000 --rotate=$symm --mlp_activation=$act \
 --noise=$noise --rollout_split=test --num_rollouts=100 &>>$DIR/log &
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
done
done
wait

