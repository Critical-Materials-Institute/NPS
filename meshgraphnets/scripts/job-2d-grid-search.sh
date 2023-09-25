#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi amsdnn
#BSUB -G amsdnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/TF1/bin/python
DAT=$HOME/data/grain/square_justNN_noduplicate

device=0
for nlayer in 11; do
for nlmlp in 2; do
for buf in 1; do
    for N1 in 4; do
        for lr in  1e-3 2e-3 3e-3 8e-4; do # 1e-2 3e-3 1e-3; do
            for nfeat in 48; do #150 180 80 128
for symm in cubic ; do #""
for act in mish; do #mish swish relu tanh
for threshold in 2e-3; do 
            DIR=meshgraphnets/experiment/grain-nfeat${nfeat}_nlayer${nlayer}_symm${symm}_act${act}_threshold${threshold}_nlmlp${nlmlp}_lr${lr}_N1${N1}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --nfeat_latent=$nfeat --n_mpassing=$nlayer \
 --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=$threshold --amr_eval_freq=5 \
 --mode=train \
 --num_training_steps=500000 --batch=8 --lr=$lr --lr_decay=70000 --cache --keep_ckpt=1 --valid_freq=8000 --rotate=$symm --mlp_activation=$act \
 --noise=0.02 --rollout_split=test --num_rollouts=100 &>>$DIR/log &
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

# # --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 &>>$DIR/log &


# buf=1; N1=2;
# device=0
# nfeat=128
# nlayer=11
# DIR=meshgraphnets/experiment/grain-nfeat128_nlayer11_symm${symm}
# mkdir -p $DIR
#             CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
#  --dataset_dir=$DAT --checkpoint_dir=$DIR \
#  --model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --nfeat_latent=$nfeat --n_mpassing=$nlayer  \
#  --amr_N=$SIZE --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=1e-3 --amr_eval_freq=5 \
#  --mode=train \
#  --num_training_steps=300000 --batch=6 --lr=4e-4 --lr_decay=80000 --cache --keep_ckpt=20 --rotate=$symm \
#  --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 &>>$DIR/log &
#             device=$((device+1))
# done
# done
# wait
# #