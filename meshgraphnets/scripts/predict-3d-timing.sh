#!/usr/bin/env bash
# #BSUB -G spnflcgc ustruct cmi
#BSUB -G cmi
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -W 12:00

# prep data
# for m in 1 2; do
#   N_L=$((64*m))
#   DAT=$HOME/data/grain/3D/3D${N_L}
#   calculator_array.py  "np.tile( $HOME/data/grain/3D/test.npy [:,:3],(1,1,$m,$m,$m,1)).astype(np.float32)" -o $DAT/test.npy
#   python meshgraphnets/npy2tfrecord.py $DAT/test.npy -o $DAT/test --periodic
# done

# conda activate powerai
PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python

device=0
counter=0
nrollout=3
dim=3
for DIR in \
  meshgraphnets/experiment/3d_grain ; do
#  meshgraphnets/experiment/grain-nfeat32_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr3e-3 \
#  meshgraphnets/experiment/grain-nfeat64_nlayer11_symmcubic_actmish_threshold2e-3_nlmlp2_lr8e-4 ; do
for device in -1 ; do
for dat in 3D256 ; do #3D64 3D128
N_L=${dat#*D}
DAT=$HOME/data/grain/3D/$dat
for npred in 400; do #200 400 600 1000 1400 2000
for N1 in 1 2; do
if [[ $N1 == 1 ]]; then amr=noAMR; else amr=AMR2; fi
for timing in 1 ; do #""
counter=$((counter+1))
tag=dim${dim}_device_${device}-dat${dat}-amr${amr}-nrollout${nrollout}-npred${npred}
PKL=tmp.pkl
TEST_TIMING_ONLY=$timing CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON `grep mode=train $DIR/command.txt|tail -n1|sed "s/.*model.py/-m meshgraphnets.run_model/;s:dataset_dir=\S*:dataset_dir=$DAT:;s:checkpoint_dir=\S*:checkpoint_dir=$DIR:;s/--num_rollouts=[0-9]*/--num_rollouts=$nrollout --num_predict_steps=$npred --rollout_path=$PKL/;s/--mode=[A-z]*/--mode=predict/;s/--amr_N1=[0-9]*/--amr_N1=$N1/;s/amr_threshold=\S*/amr_threshold=2e-3/;s/amr_N=\S*/amr_N=$N_L/"` 
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

