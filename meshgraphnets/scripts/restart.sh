#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# conda activate powerai
PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/grain/square_justNN_noduplicate

device=0

for DIR in meshgraphnets/experiment/grain-nfeat64_nlayer1?_symmcubic_act*_threshold2e-3; do 
CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON `tail -n1 $DIR/command.txt|sed 's/.*model.py/-m meshgraphnets.run_model/'` &>>$DIR/log &
            device=$((device+1))
done
wait
