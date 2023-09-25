#!/usr/bin/env bash
# #BSUB -G msgnn ustruct cmi amsdnn
#BSUB -G msgnn
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

PYTHON=$HOME/lassen-space/anaconda2021/envs/TF1/bin/python
DAT=$HOME/amsdnn/2D_forest/dataset1

device=0
for nlayer in 10 ; do
for nlmlp in 2; do
# for buf in 1; do
for noise in 3e-2 1e-2 5e-3 5e-2; do
    # for N1 in 1; do
        for lr in 2e-3 ; do # 1e-2 3e-3 1e-3; do
            for nfeat in 96; do #150 180 80 128
# for symm in cubic ; do #""
for act in swish; do #mish swish relu tanh
# for threshold in 2e-3; do 
            DIR=meshgraphnets/experiment/forest2deasy-nfeat${nfeat}_nlayer${nlayer}_nlmlp${nlmlp}_lr${lr}_noise${noise}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=2 --nfeat_out=2 --nfeat_latent=$nfeat --n_mpassing=$nlayer --nlayer_mlp=$nlmlp \
 --mode=train \
 --num_training_steps=500000 --batch=2 --lr=$lr --lr_decay=70000 --cache --keep_ckpt=1 --valid_freq=8000 --mlp_activation=$act \
 --noise=$noise --rollout_split=valid --num_rollouts=3 --rollout_path=tmp.pkl &>>$DIR/log &
            device=$((device+1))
#  --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=$threshold --amr_eval_freq=5 \
#  --core=$core \
done
done
done
done
            done
            done

wait

