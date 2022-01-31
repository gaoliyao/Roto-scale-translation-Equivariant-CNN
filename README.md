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
Please specify the rotation range and scaling range for data generation. We initially generate [0. 360] rotation and [0.3, 1.0] scaling. If resample/regeneration of training/testing dataset is required, please set i from 0 to the expected number of experiments you wish. 

#### 2. Training. Specify hyperparameters and train with specified model. 
We include the following models with the following names explained in this [link](https://github.com/gaoliyao/sesn/wiki/Model-and-their-names). 

To test for one or two models, just directly call the shell script. 
```
# For MNIST
sh experiments_mnist_small.sh
# For FMNIST
sh experiments_fmnist_small.sh
```

#### 3. Check testing results. 
The results are collected in file results.yml after running. An example running output is in the following. A line represents one experimental trial with model name, accuracy, used basis, and other hyper-parameters (like learning rate, batch size). 

```
- {model: 'mnist_cnn_56', acc: 0.89276, basis: 'D1', batch_size: 128, cuda: true, data_dir: './datasets/MNIST_scale/seed_0/scale_0.3_1.0', dataset: 'scale_mnist', decay: 0.0001, elapsed_time: 110, epochs: 45, extra_scaling: 0.5, lr: 0.01, lr_gamma: 0.1, lr_steps: [30], momentum: 0.9, nesterov: false, num_parameters: 494549, optim: 'adam', save_model_path: './saved_models/mnist/mnist_cnn_56_extra_scaling_0.5.pt', tag: 'sesn_experiments', time_per_epoch: 2}
```

## Models in paper
Due to double-blind policy, we cannot directly attach the links to specific models here. We mark the file names with line numbers instead in the following. We included some of the models within this repo. To run different models, please specify desired model name (line 40) and basis type (line 21) in the shell script in experiments_mnist_small.sh.

### 1. CNN (Baseline)

In file models/mnist_cnn.py Line 7 - 41. We applied function in models/mnist_cnn.py Line 48 - 49 to setup this CNN model. We note here our models are built upon this Convolutional Neural Network. There is no basis choice for CNN. 

In the shell script, please add **mnist_cnn_56** in model list to run. 

### 2. RDCF (Rotation-Equivariance Baseline)

In file models/mnist_res.py Line 11 - 54. We applied function in models/mnist_res.py Line 235 - 249 to setup this RDCF model. We use Fourier-Bessel with one scale basis (marked by D1 in models/impl/ses_basis.py) in this case. In the shell script, please add **mnist_res_scalar_56_rot_8** in model list to run. Further, please set the basis to be "D1". 

- **"rot_8"** indicates that there are 8 rotation channels uniformly distanced in range $(-pi, pi)$. 
- **scalar** means that there is no inter-rotation. **res** stands for "rotation-equivariant specified". 

### 3. SEVF (Scale-Equivariance Baseline)

In file models/mnist_sevf.py Line 50 - 88. We applied function in models/mnist_sevf.py Line 95 - 96 to setup this SEVF model. There is no choice of basis for this model. 

In the shell script, please add **mnist_sevf_scalar_56** in model list to run. **scalar** means that there is no inter-scale operations. 

### 4. SESN (Scale-Equivariance Baseline)

In file models/mnist_ses.py Line 11 - 54. We applied function in models/mnist_ses.py Line 226 - 241 to setup this SESN model. We use Hermite Gaussian with multi-scale basis (marked by C in models/impl/ses_basis.py) in this case. In the shell script, please add **mnist_ses_scalar_56_rot_1** in model list to run. Further, please set the basis to be "C". 

- **"rot_1"** indicates that there are no additional rotation channels. 
- **scalar** means that there is no inter-rotation. 
- **ses** stands for "scale-equivariant specified". 
- The number of scale channels is setup to be 4. 

### 5. SDCF (Scale-Equivariance Baseline)

In file models/mnist_ses.py Line 11 - 54. We applied function in models/mnist_ses.py Line 226 - 241 to setup this SDCF model. We use SL basis (marked by G in models/impl/ses_basis.py) in this case. In the shell script, please add **mnist_ses_scalar_56_rot_1** in model list to run. Further, please set the basis to be **"G"**. 

- **"rot_1"** indicates that there are no additional rotation channels. 
- **scalar** means that there is no inter-rotation. 
- **ses** stands for "scale-equivariant specified". 
- The number of scale channels is setup to be 4. 

### 6. RST-CNN FB/SL (Ours)

In file models/mnist_ses.py Line 11 - 54. We applied function in models/mnist_ses.py Line 296 - 311 to setup this RST-CNN model. We use FB/SL basis (marked by E/G in models/impl/ses_basis.py) in this case. In the shell script, please add **mnist_ses_scalar_56_rot_8** in model list to run. Further, please set the basis to be **"E"** or **"G"**. 

- **"rot_8"** indicates that there are 8 rotation channels uniformly distanced in range $(-pi, pi)$. 
- If we specify **mnist_ses_scalar_56_rot_4**, there will have 4 rotation channels uniformly distanced in range $(-pi, pi)$.
- **scalar** means that there is no inter-rotation. 
- **ses** stands for "scale-equivariant specified". 
- The number of scale channels is setup to be 4. 

### 7. RST-CNN Inter-rotation FB/SL (Ours)

In file models/mnist_ses.py Line 57 - 100. We applied function in models/mnist_ses.py Line 331 - 347 to setup this RST-CNN model. We use FB/SL basis (marked by E/G in models/impl/ses_basis.py) in this case. In the shell script, please add **mnist_ses_vector_56_rot_8_interrot_4** in model list to run. Further, please set the basis to be **"E"** or **"G"**. 

- **"rot_8"** indicates that there are 8 rotation channels uniformly distanced in range $(-pi, pi)$. 
- **vector** means that there will be inter-rotations. 
- **interrot_4** means that four rotation channels will be inter-rotated. 
- **ses** stands for "scale-equivariant specified". 
- The number of scale channels is setup to be 4. 
- If we specify **mnist_ses_vector_56_rot_8_interrot_8**, all rotation channels will be inter-rotated with each other. 


### Basis implementations
Basis A1: Hermite Gaussian one scale

Basis C: Hermite Gaussian rotation and multi-scale

Basis D1: Fourier Bessel one scale

Basis E: Fourier Bessel rotation and multi-scale

Basis G1: SL rotation and one-scale

Basis G: SL rotation and multi-scale
