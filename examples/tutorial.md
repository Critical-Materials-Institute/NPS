# Tutorial for NPS

# Dataset
Training data set can be 
* Microstructure evolution trajectory in the form of array(s) on a regular 2D/3D grid, of the shape \[N<sub>sequence</sub>, N<sub>timestep</sub>, N<sub>x</sub>, N<sub>y</sub>, \[N<sub>z</sub>\], N<sub>channel</sub>\]. Currently supported formats are .npy, .npz
* MD trajectory for crystal or molecule simulations. Currently supported formats include LAMMPS dump file (.lammpstrj), GROMACS (.xtc), PDB

## Dataset utilities
### Visualizing animated .npy trajectory with [scripts/animate-2d.py](../scripts/animate-2d.py) [scripts/animate-3d.py](../scripts/animate-3d.py)
```
animate-2d.py train.npy
```

# Example on grain growth
## Using baseline CNN
```bash
DIR=PATH_TO_JOB_DIRECTORY
DAT=PATH_TO_DATASET
python -m NPS.main --mode=train --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64 --nfeat_in=1 --periodic --pointgroup=4m --dim=2 \
 --model=NPS.model.convnext --kernel_size=3 \
 --batch=32 --lr=1e-4 --n_in=1 --n_out=1
```
where the training dataset is of the shape \[\*, \*, 64, 64, 1\] with one phase field channel and periodic boundary condition, a [convnext](https://arxiv.org/abs/2201.03545) network, 3x3 convolution kernals, learning rate of 1e-4.

## Using MeshGraphNet GNN
```bash
python -m NPS.main --mode=train --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64 --nfeat_in=1 --periodic --pointgroup=4m --dim=2 \
 --model=MeshGraphNets --trainer=MeshGraphNets --gnnmodel=NPS \
 --batch=32 --lr=1e-4 --n_in=1 --n_out=1
```
where a GNN rather than CNN was used

## Using a combined CNN+GNN architecture
```bash
python -m NPS.main --mode=train --dir=$DIR \
 --data=$DAT --dataset=longclip --frame_shape=64 --nfeat_in=1 --periodic --pointgroup=4m --dim=2 \
 --model=MeshGraphNets --trainer=MeshGraphNets --gnnmodel=NPS_autoencoder
 --autoencoder=rev2wae --nstrides_2wae=$ae_stride --nblocks_2wae=$ae_block --nlayer_mlp_encdec=$nencdec \
 --batch=32 --lr=1e-4 --n_in=1 --n_out=1
```
where a [reversible autoencoder](https://openreview.net/pdf?id=B1eY_pVYvB) was added for dimensionality reduction
