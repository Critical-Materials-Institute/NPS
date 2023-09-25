#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi
#BSUB -G spnflcgc
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# conda activate powerai
PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/grain/square_justNN_noduplicate

device=0
if false; then
for buf in 0 1; do
    for N1 in 2 4; do
        for lr in 1e-4; do # 1e-2 3e-3 1e-3; do
            DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN-AMR-N1${N1}_buf${buf}_lr${lr}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 \
 --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=1e-3\
 --mode=train \
 --num_training_steps=400000 --batch=12 --lr=$lr --lr_decay=100000 --cache \
 --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 &>$DIR/log &
            device=$((device+1))
        done
    done
done
wait
fi

for buf in 1; do
    for N1 in 2; do
        for lr in 1e-4; do # 1e-2 3e-3 1e-3; do
            for act in swish relu tanh sigmoid; do # mish
            DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN-AMR-N1${N1}_buf${buf}_lr${lr}_act${act}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --mlp_activation=$act \
 --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=1e-3\
 --mode=train \
 --num_training_steps=500000 --batch=6 --lr=$lr --lr_decay=100000 --cache \
 --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 &>>$DIR/log &
            device=$((device+1))
            done
        done
    done
done
wait
