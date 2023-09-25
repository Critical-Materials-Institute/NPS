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

for DIR in \
 meshgraphnets/experiment/grain-nfeat32_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr3e-3/ meshgraphnets/experiment/grain-nfeat32_nlayer10_symmcubic_actmish_threshold2e-3_nlmlp2_lr3e-3/ meshgraphnets/experiment/grain-nfeat32_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr1.5e-3/ meshgraphnets/experiment/grain-nfeat32_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr8e-4/ meshgraphnets/experiment/grain-nfeat32_nlayer10_symmcubic_actmish_threshold2e-3_nlmlp2_lr2e-3/ meshgraphnets/experiment/grain-nfeat32_nlayer9_symmcubic_actmish_threshold2e-3_nlmlp2_lr3e-3/ meshgraphnets/experiment/grain-nfeat32_nlayer9_symmcubic_actmish_threshold2e-3_nlmlp2_lr2e-3 \
 meshgraphnets/experiment/grain-nfeat48_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr1e-3/ meshgraphnets/experiment/grain-nfeat48_nlayer10_symmcubic_actmish_threshold2e-3_nlmlp2_lr1e-3/ meshgraphnets/experiment/grain-nfeat48_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr2e-3/ meshgraphnets/experiment/grain-nfeat48_nlayer10_symmcubic_actmish_threshold2e-3_nlmlp2_lr2e-3/\
 meshgraphnets/experiment/grain-nfeat64_nlayer10_symmcubic_actmish_threshold2e-3_nlmlp2_lr1e-3/ meshgraphnets/experiment/grain-nfeat64_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr8e-4/ meshgraphnets/experiment/grain-nfeat64_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr1.5e-3/ meshgraphnets/experiment/grain-nfeat64_nlayer11_symmcubic_actswish_threshold2e-3_nlmlp2_lr3e-3/ meshgraphnets/experiment/grain-nfeat64_nlayer10_symmcubic_actmish_threshold2e-3_nlmlp2_lr2e-3/ meshgraphnets/experiment/grain-nfeat64_nlayer11_symmcubic_actswish_threshold2e-3_nlmlp2_lr1.5e-3/; do 
# CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON `grep mode=train $DIR/command.txt|tail -n1|sed "s/.*model.py/-m meshgraphnets.run_model/;s:mode=train:mode=eval --rollout_path=$DIR/rollout.pkl:"` &>>$DIR/log &
CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON `grep mode=train $DIR/command.txt|tail -n1|sed "s/.*model.py/-m meshgraphnets.run_model/;s:mode=train:mode=eval:;s/rollout_path=\S*/rollout_path= /"` &>>$DIR/log &
device=$(((device+1)%4))
if [[ $device == 3 ]]; then wait; fi
done

