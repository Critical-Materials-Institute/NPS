## tensorflow 1.x
# conda activate powerai


DIR=learning_to_simulate
python -m $DIR.train \
    --data_path=$DIR/data/GNN-hydrodynamics/WaterRamps/ \
    --model_path=$DIR/experiments/WaterRamps/models/ \
    --output_path=$DIR/experiments/WaterRamps/rollouts \
    --batch_size=14 --lr_decay=100000

############ Si nanoparticle
DAT=learning_to_simulate/data/nano-Si
DIR=learning_to_simulate/experiments/nano-Si
# data
(cd ../ising-gpu/MD/nano-Si/
for i in `seq 4`; do
    T=`shuf -i 800-1200 -n 1`;
    CUDA_VISIBLE_DEVICES=3 lmp_lassen_llnl -k on g 1 -sf kk  -i <(sed "s/t equal 2000.0/t equal $T/" in.Si)
    mv dump.xy dump.xy-$i
done
python learning_to_simulate/particle2tfrecord.py ../ising-gpu/MD/nano-Si/dump.xy-* -o $DAT/train
# train
CUDA_VISIBLE_DEVICES=3 nohup python -m learning_to_simulate.train --data_path=$DAT --model_path=$DIR --output_path=$DIR \
    --batch_size=64 --lr_decay=200000 --lr=2e-3  --in_seq_len=6 --noise_std=3e-3 --rotate --cache 
# eval
python -m learning_to_simulate.train --mode=eval \
    --data_path=$DAT --model_path=$DIR --output_path=$DIR 
# rollout
python -m learning_to_simulate.train --mode=eval_rollout \
    --data_path=$DAT --model_path=$DIR --output_path=$DIR 
# render
python my_scripts/render_3d.py  --rollout_path=$DIR/rollout_test_0.pkl


# cloth
python -m meshgraphnets.run_model --model=cloth --mode=train \
    --checkpoint_dir=meshgraphnets/experiment --dataset_dir=$HOME/data/airfoil/flag_minimal  --num_training_steps=100000
python -m meshgraphnets.run_model --model=cloth --mode=eval \
    --checkpoint_dir=meshgraphnets/experiment --dataset_dir=$HOME/data/airfoil/flag_minimal  \
    --rollout_path=meshgraphnets/experiment/rollout.pkl --num_rollouts=1
python -m meshgraphnets.plot_cloth --rollout_path=meshgraphnets/experiment/rollout.pkl

# cylinder_flow
DIR=meshgraphnets/experiment/cylinder_flow
DAT=$HOME/data/airfoil/cylinder_flow
python -m meshgraphnets.run_model --model=cfd --mode=train --checkpoint_dir=$DIR --dataset_dir=$DAT --num_training_steps=100000 --batch=16
python -m meshgraphnets.run_model --model=cfd --mode=eval  --checkpoint_dir=$DIR --dataset_dir=$DAT --rollout_path=$DIR/rollout.pkl --num_rollouts=1 
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl
# output format:
#   faces[ntime, nface, 3] int
#   mesh_pos[ntime, nnode, dim] float
#   pred_velocity,pred_velocity[ntime, nnode, nfeature] float

# wave 1 mode
DIR=meshgraphnets/experiment/wave1mode
DAT=$HOME/data/waves-1mode
for i in valid train test; do
    python meshgraphnets/npy2tfrecord.py $DAT/$i.npy -o $DAT/$i;
done
python -m meshgraphnets.run_model --model=NPS --mode=train --checkpoint_dir=$DIR --dataset_dir=$DAT --nfeat_in=1 --nfeat_out=1 --num_training_steps=100000 --batch=8  
python -m meshgraphnets.run_model --model=NPS --mode=eval  --checkpoint_dir=$DIR --dataset_dir=$DAT --nfeat_in=1 --nfeat_out=1 --rollout_path=$DIR/rollout.pkl --num_rollouts=1
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl



# grain growth
DIR=meshgraphnets/experiment/grain
DAT=$HOME/data/grain
# data generation
for i in valid train test; do
    python meshgraphnets/npy2tfrecord.py $DAT/$i.npy -o $DAT/$i --periodic;
done
# train
python -m meshgraphnets.run_model --model=NPS --mode=train --checkpoint_dir=$DIR --dataset_dir=$DAT --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=500000 --batch=8 --lr=4e-4 --lr_decay=100000
# rollout
python -m meshgraphnets.run_model --model=NPS --mode=eval  --checkpoint_dir=$DIR --dataset_dir=$DAT --periodic=1 --nfeat_in=1 --nfeat_out=1 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=1
# visualization
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl --skip=5 --mirrory
python -m meshgraphnets.plot_cfd --rollout_path=$DIR/rollout.pkl --skip=98 -o GNN-grain.gif --scale=0.5
################# tests, dilation tests, hyper parameters
base_cmd="python -m meshgraphnets.run_model --model=NPS --mode=train --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=200000 --batch=8 --lr=4e-4 --lr_decay=100000"
## noise vs no noise
device=0
for noise in 0 0.005 0.02 0.04; do
    tag=grain-noise_${noise}
    DIR=meshgraphnets/experiment/$tag
    mkdir -p $DIR
    echo $base_cmd --noise=$noise > $DIR/command-line.txt
    CUDA_VISIBLE_DEVICES=$device nohup $base_cmd --checkpoint_dir=$DIR  --dataset_dir=$DAT --noise=$noise &> $DIR/log &
    device=$((device+1))
done
## random triangle vs ordered triangle vs no-diagonal vs all diagonal
device=0
noise=0.02
for tri in 'square_1-1' 'square_11' 'square_X' 'square_justNN' 'square_justNN_noduplicate'; do
    tag=grain-noise_${noise}_tri_${tri}
    DIR=meshgraphnets/experiment/$tag
    DAT=$HOME/data/grain/$tri
    mkdir -p $DIR $DAT
    for i in valid train test; do
        python meshgraphnets/npy2tfrecord.py $DAT/../$i.npy -o $DAT/$i --periodic --typ=$tri;
    done
    echo $base_cmd --checkpoint_dir=$DIR  --dataset_dir=$DAT --noise=$noise > $DIR/command-line.txt
    CUDA_VISIBLE_DEVICES=$device nohup $base_cmd --checkpoint_dir=$DIR  --dataset_dir=$DAT --noise=$noise &> $DIR/log &
    device=$((device+1))
done
(cd meshgraphnets/experiment/; for i in grain*/log; do echo -n '"< grep Loss '$i'" u 8 w l t "'$i'",'; done |sed 's/^/set logscale y; set yrange [0.001:0.1]; set xlabel "step"\nset ylabel "loss"\nplot /;s/,$/\npause 99\n/' | gnuplot)
for tri in square_randomtriangle 'square_1-1' 'square_11' 'square_X' 'square_justNN'; do
    DAT=$HOME/data/grain/$tri
    for DIR in meshgraphnets/experiment/grain*/; do
        echo $tri $DIR
        CUDA_VISIBLE_DEVICES=1 python -m meshgraphnets.run_model --model=NPS --mode=eval --checkpoint_dir=$DIR --dataset_dir=$DAT --periodic=1 --nfeat_in=1 --nfeat_out=1 --rollout_split=test --rollout_path=$DIR/$tri.pkl --num_rollouts=20
    done
done > all-grain-test.txt
#################
## testing AMR
export DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN
CUDA_VISIBLE_DEVICES=1 python -m meshgraphnets.run_model --model=NPS \
  --mode=train --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=300000 --batch=8 --lr=4e-4 --lr_decay=100000 --checkpoint_dir=$DIR \
  --dataset_dir=$HOME/data/grain/square_justNN --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=1 \
  --amr_N=64 --amr_N1=2 --amr_buffer=0 --amr_threshold=1e-3

#################
## testing dataset with no duplicate edges, no AMR
export DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN
CUDA_VISIBLE_DEVICES=1 python -m meshgraphnets.run_model --model=NPS \
  --mode=eval --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=300000 --batch=8 --lr=4e-4 --lr_decay=100000 --checkpoint_dir=$DIR \
  --dataset_dir=$HOME/data/grain/square_justNN_noduplicate --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 \
  --amr_N=64 --amr_N1=1 --amr_buffer=0 --amr_threshold=1e-3
CUDA_VISIBLE_DEVICES=1 python -m meshgraphnets.run_model --model=NPS \
  --mode=eval --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=300000 --batch=8 --lr=4e-4 --lr_decay=100000 --checkpoint_dir=$DIR \
  --dataset_dir=$HOME/data/grain/square_justNN --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 \
  --amr_N=64 --amr_N1=1 --amr_buffer=0 --amr_threshold=1e-3

#################
## disable tf.unique op in building edges
export DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN
CUDA_VISIBLE_DEVICES=1 python -m meshgraphnets.run_model --model=NPS --unique_op=false \
 --mode=eval --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=300000 --batch=8 --lr=4e-4 --lr_decay=100000 --checkpoint_dir=$DIR \
 --dataset_dir=$HOME/data/grain/square_justNN_noduplicate --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 \
 --amr_N=64 --amr_N1=1 --amr_buffer=0 --amr_threshold=1e-3

#################
## AMR with unique_op off
export DIR=meshgraphnets/experiment/grain-noise_0.02_tri_square_justNN
CUDA_VISIBLE_DEVICES=1 python -m meshgraphnets.run_model --model=NPS --unique_op=false \
 --mode=eval --periodic=1 --nfeat_in=1 --nfeat_out=1 --num_training_steps=300000 --batch=8 --lr=4e-4 --lr_decay=100000 --checkpoint_dir=$DIR \
 --dataset_dir=$HOME/data/grain/square_justNN_noduplicate --noise=0.02 --rollout_split=test --rollout_path=$DIR/rollout.pkl --num_rollouts=10 \
 --amr_N=64 --amr_N1=2 --amr_buffer=0 --amr_threshold=1e-3

