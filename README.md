# Deep_Learning_via_C_plus_plus
> This repository includes homemade C++ source code of deep learning
## This reporsitory includes 2 Visual Studio solutions: *neurons_windows* and *neurons_linux*

- **I am now trying my best to make the source code almost identical for both windows and linux.**
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
