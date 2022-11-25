# Neural Phase Simulation (NPS)
NPS is a package of codes for simulating microstructure evolution and accelerated molecular dynamics with deep neural-networks based surrogate models. NPS is designed to offer quantitatively accurate and computationally efficient simulation capabilities by leveraging modern machine-learning techniques. The primary intended use cases of NPS are training neural network surrogate models, though performing simulations on a single node is also supported. 

# Features
## Capabilities
* 2D, 3D microstructure evolution, including grain growth, nucleation and growth, spinodal decomposition, dendrite growth
* Deterministic simulations similar to phase-field or partial differential equation
* Stochastic simulations similar to stochastic differential equation
* Accelerated full-atom or coarsened molecular dynamics
* 2D dislocation dynamics (surrogate mobility function)
* Periodic boundary conditions in 2D/3D
* Some symmetry (e.g. rotational equivariance) and conservation laws (e.g. mass conservation)
* New capabilities will be developed and added

## Training ground-truth or high-fidelity method
The NPS surrogate models can be trained from various ground truth simulation methods, which are supposed to be accurate but expensive, such as 
* Molecular dynamics
* Phase field methods
* Kinetic Monte Carlo
* Discrete dislocation dynamics

## Networks:
  * Convolutional neural networks (ResNet, U-net, ConvNext)
  * Convolutional LSTM (PredRNN, PredRNN++, E3dLSTM)
  * Graph neural networks [(MeshGraphNets)](https://arxiv.org/abs/2010.03409)
  * Rotationally equivariant GNN [(nequip)](https://github.com/mir-group/nequip)
  * Denoising diffusion probabilistic model
<!--  * VAE with RNN on latent bottleneck (TBD)
  * Attention based, transformer like (TBD) -->
<!--Loss functions:
  * L1, L2 loss of pixels/voxels-->
<!--  * GAN loss (TBD)
  * Perceptual loss (TBD) -->
<!--Special ops -->
<!--  * Point group symmetry through data augmentation -->
<!--  * Attention (TBD) -->

# Installation
NPS requires:
* Python >= 3.6
* PyTorch >= 1.9
* Torch Geometric, [E3NN](https://github.com/e3nn/e3nn)
* Numpy, Scipy, Matplotlib

# Getting started
The main entry is [NPS/main.py](NPS/main.py).
## Training
```
python -m NPS/main.py --mode=train ...
```

## Prediction/Simulation
```
python -m NPS/main.py --mode=predict ...
```

See the [tutorial](examples/tutorial.md)

# References
1. [Yang, Kaiqi, Yifan Cao, Youtian Zhang, Shaoxun Fan, Ming Tang, Daniel Aberg, Babak Sadigh, and Fei Zhou. “Self-Supervised Learning and Prediction of Microstructure Evolution with Convolutional Recurrent Neural Networks.” Patterns 2, 100243 (2021)](https://doi.org/10.1016/j.patter.2021.100243)
2. [Bertin, Nicolas, and Fei Zhou. 2022. “Accelerating Discrete Dislocation Dynamics Simulations with Graph Neural Networks.” arxiv:2208.03296](http://arxiv.org/abs/2208.03296)

# Authors
NPS is being developed by by [Fei Zhou](mailto:zhou6@llnl.gov)


# Getting Involved
Please contact [Fei Zhou](mailto:zhou6@llnl.gov) for questions.

# Contributing
The NPS package is intended to be a general and extensible framework to develop machine-learning surrogate models for materials simulation. Contributions are welcome. 
Just send us a pull request. When you send your request, make `develop` the destination branch on the repository.

Users who want the latest package versions, features, etc. can use `develop`.


# License
NPS is distributed under the terms of the MIT license.

All new contributions must be made under the MIT license.

SPDX-License-Identifier: MIT

LLNL-CODE-842508
