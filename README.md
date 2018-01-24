# Deep_Learning_via_C_plus_plus
> This repository includes homemade C++ source code of deep learing
## You can download this repository as a Visual Studio 2017 solution
## This solution includes four projects:
- **neurons** is a static library including some basic neuron concepts.
> Following algorithms and functions are included in this module
> - *vector and matrix calculations*
> - *activation functions*
> - *cost or error functions*
> - *convolutional functions*
> - *pooling functions*
> - *forward and backward propagation algorithms*
> - *basic building block of RNN (RNN_unit)*
> - *One dimensional GMM EM algorithm*
- **dnn** is a multi-layer fully connected neural network program that can be trained.
- **cnn** is a homemade multi-layer convolutional neural network program that can be trained.
- **rnn** has recently implemented a simple RNN module. LSTM is still under construction.
- **test** includes some basic test cases of neural calculations.
- Dataset of this program is **MNIST**, you may have to update *Mnist.cpp* to add supports of other dataset types if you would like.
