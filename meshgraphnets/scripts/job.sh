#!/usr/bin/env bash
# #BSUB -G spnflcgc
#BSUB -G ustruct
#BSUB -q pdebug
#BSUB -nnodes 1
#BSUB -W 2:00

conda activate powerai

DAT=learning_to_simulate/data/nano-Si
DIR=learning_to_simulate/experiments/nano-Si

gpu=0
for noise in 5e-4 2e-3 1e-2 5e-2; do # 5e-4 2e-3 1e-2 5e-2 3e-1; do
    for len in 6 ; do
        tag=${DIR}_noise${noise}_inlen${len}; mkdir -p $tag
        CUDA_VISIBLE_DEVICES=$gpu python -m learning_to_simulate.train \
            --data_path=$DAT --model_path=$tag --output_path=$tag --batch_size=32 --lr_decay=200000 --lr=1e-3  --noise_std=$noise --in_seq_len=$len --rotate --cache &> $tag/log &
        gpu=$((gpu+1))
    done
done
wait
