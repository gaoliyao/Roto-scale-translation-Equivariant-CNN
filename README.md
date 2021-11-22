# Deformation Robust Roto-Scale-Translation Equivariant CNNs

Experiment code for "Deformation Robust Roto-Scale-Translation Equivariant CNNs". This is a Convolutional Neural Network (CNN) model that is robust to rotation, scaling, and translation transformation under local deformation. 

## Reproduce experiments on Rot-scale MNIST and Rot-scale Fasion-MNIST
### Environment
1. torch 1.7.1
2. torchvision 0.8.2
3. numpy 1.20.1
4. scipy 1.5.2
5. matplotlib 3.3.2

### Steps of running experiments
1. Data generation. Generate rotated/scaled datasets to folder. 
2. Training. Specify hyperparameters and train with specified model. 
3. Check testing results. 

#### 1. Data generation
```
# MNIST
sh prepare_mnist_rot_scale.sh
# Fashion-MNIST
sh prepare_fmnist_rot_scale.sh
```
Please specify the rotation range and scaling range for data generation. We initially generate [0. 360] rotation and [0.3, 1.0] scaling. 
