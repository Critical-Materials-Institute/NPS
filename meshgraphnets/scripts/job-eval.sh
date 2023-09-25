device=3

PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/grain/square_justNN_noduplicate
SIZE=64
# DAT=$HOME/data/grain/square_justNN_noduplicate/4x4
# SIZE=256

for buf in 0 1; do
    for N1 in 2; do
        for freq in 1 10 20 40 60 9999 ; do # 20  999; do
            for act in relu; do  #swish relu tanh sigmoid mish; do # mish
            DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN-AMR-N12_buf1_lr1e-4_actrelu
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --mlp_activation=$act \
 --amr_N=$SIZE --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=1e-3 --amr_eval_freq=$freq \
 --mode=eval \
 --num_training_steps=500000 --batch=6 --lr=1e-4 --lr_decay=100000 --cache \
 --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout-$freq.pkl --num_rollouts=10  &> tmp.txt
 t1=`grep 'Rollout trajectory 1' tmp.txt |sed 's/:/ /g'|awk '{print $2*3600*0 + $3*60 + $4}'`
 t2=`grep 'Rollout trajectory 9' tmp.txt |sed 's/:/ /g'|awk '{print $2*3600*0 + $3*60 + $4}'`
 echo buf $buf freq $freq `echo $t1 $t2 |awk '{print ($2-$1)/8}'` `grep total tmp.txt |awk '{print $3}'`
            # device=$((device+1))
            done
        done
    done
done

