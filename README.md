# DeepFrame
> A new deep learning framework. This repository only includes original C++ source code of deep learning
## This reporsitory is coowned and collaborated with Ning Xu (xuningandy@outlook.com).
## Apology for the possible delay due to pandemics, we are working to our full extent.
-----------
## Updates in next version (v1.1)
- **To be submitted to Journal of Machine learning Research.**
- **We migrate the package to CUDA and OpenCL framework for GPU parallel computation.**
- **We incorporate DeepFrame into ASpark framework**
- **We optimize the code for distributed computation (working in progress, may be delayed)**
- **We are trying to add Python portal (working in progress, highly likely to be delayed)**
- **We unify the source code for both windows and linux. (containers like docker are recommended for Win10)**
- **We add variational inference module and reversible jump MCMC module for training a sparse Bayesian Neural Network in ultrahigh dimensional spaces(beta).**
- **We add Gaussian Process regression module (beta).**
- **We add subsample ordering module for regularizaion and subsample ordered training for stagewise neural net (beta), see https://github.com/isaac2math/solar for detail.**
------------
## About this version (v1.0)
- **This reporsitory includes 2 Visual Studio solutions: *neurons_windows* and *neurons_linux*.**
- **C++ version should be at least C++ 14 here**
- **Both windows and linux solutions can be compiled via Visual Studio 2017 (the linux version needs a linux platform where g++ is installed).**
- **Locations of compiled static libraries of *neurons* and *dataset* module may be different on different linux machines.**
**You may have to update configurations of modules that may depend on *neurons* and *dataset* before you build the linux version solution.** 
- **The linux version is not complete because it is still under construction.**

## The *neurons* module
- **neurons** is a static library including some basic neuron concepts.
> Following algorithms and functions are included in this module
> - *Vector and Matrix Calculations*
> - *Activation Functions*
> - *Cost, Error or Loss Functions*
> - *Convolutional Functions*
> - *Pooling Functions*
> - *Forward Propagation*
> - *Back Propagation*
> - *Back Propagation Through Time*
> - *Batch Learning*
> - *Multi-threading*
> - *Basic Building Block of RNN (RNN_unit)*
> - *One Dimensional GMM EM Algorithm*
> - *Linear Regression Algorithm*

## The *dataset* module 
- **dataset** is a module with a general dataset interface for neural network training.
> Supports of following dataset are included in this module
> - **MNIST** for feedforward neural networks
>>> MNIST can be downloaded directly from opensource communities.
>>> Both MNIST and MNIST-fashion are supported.
> - **CIFAR_10** for feedforward neural networks
>>> CIFAR_10 can also be downloaded from opensource communities.
>>> 
> - **Media Review** for recurrent neural networks (RNN)
>>> I think you have to ask me for the dataset. Don't hesitate to contact me if you need it.

## The *dnn* module
- **dnn** is a multi-layer fully connected neural network program that can be trained.

## The *cnn* module
- **cnn** is a homemade multi-layer convolutional neural network program that can be trained.

## The *rnn* module
- **rnn** has recently implemented a simple RNN module. LSTM is still under construction.

## The *test* module
- **test** includes some basic test cases of neural calculations.

## The *misc* module
- **misc** may include any possible algorithms that depend on other major modules of this project.
