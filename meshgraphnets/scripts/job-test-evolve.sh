PYTHON=~/lassen-space/conda-ibm/envs/powerai/bin/python
DAT=$HOME/data/grain/square_justNN_noduplicate

device=0
for nlayer in 11; do 
for buf in 1; do
    for N1 in 2; do
        for lr in 2e-4; do  
            for nfeat in 128 ; do 
for symm in cubic; do
for act in relu; do
for nevolve in {8..5}; do
            DIR=meshgraphnets/experiment/grain-nfeat${nfeat}_nlayer${nlayer}_nevolve${nevolve}
            mkdir -p $DIR
            CUDA_VISIBLE_DEVICES=$device TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model \
 --dataset_dir=$DAT --checkpoint_dir=$DIR \
 --model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --nfeat_latent=$nfeat --n_mpassing=$nlayer --n_evolve=$nevolve \
 --amr_N=64 --amr_N1=$N1 --amr_buffer=$buf --amr_threshold=1e-3 --amr_eval_freq=5 \
 --mode=train \
 --num_training_steps=500000 --batch=4 --lr=$lr --lr_decay=70000 --cache --keep_ckpt=20 --rotate=$symm  --noise=0.02 \
 --rollout_split=test --num_rollouts=100  &>> $DIR/log &
            device=$((device+1))
done
done
done
            done
            done
        done
    done
done
wait

if false; then
for evaluator in cfd_eval featureevolve_eval; do
CUDA_VISIBLE_DEVICES=3 TF_CPP_MIN_LOG_LEVEL=2 $PYTHON -m meshgraphnets.run_model 
--dataset_dir=$DAT --checkpoint_dir=meshgraphnets/experiment/grain-nfeat128_nlayer11_nevolve1 
--model=NPS --unique_op=false --periodic=1 --nfeat_in=1 --nfeat_out=1 --nfeat_latent=128 --n_mpassing=11 --n_evolve=1 
--amr_N=64 --amr_N1=1 --amr_buffer=1 --amr_threshold=1e-3 --amr_eval_freq=5 
--mode=eval 
--num_training_steps=500000 --batch=4 --lr=2e-4 --lr_decay=70000 --cache --keep_ckpt=20 --rotate=cubic --noise=0.02 
--rollout_split=test --num_rollouts=4 --evaluator=$evaluator
done
fi