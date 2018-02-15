#include "Simple_NN.h"
#include "Multi_Layer_NN.h"
#include "Mnist.h"
#include "Review.h"
#include "CIFAR_10.h"

#pragma optimize("", off)

int main(int argc, const char * argv[])
{
    std::string dataset_dir = "D:/develop/my_neurons/dataset/";
    lint batch_size = 64;
    lint threads = 8;
    lint epoch_size = 20;

    if (argc == 4)
    {
        batch_size = std::stoi(argv[1]);
        threads = std::stoi(argv[2]);
        epoch_size = std::stoi(argv[3]);
    }

    dataset::Mnist mnist{
        dataset_dir + "mnist/train-images-idx3-ubyte",
        dataset_dir + "mnist/train-labels-idx1-ubyte",
        dataset_dir + "mnist/t10k-images-idx3-ubyte",
        dataset_dir + "mnist/t10k-labels-idx1-ubyte" };

    dataset::CIFAR_10 cifar_10{ dataset_dir + "cifar-10/" };

    Multi_Layer_NN nn{ 0.001, batch_size, threads, 5000000, epoch_size, 7200, cifar_10 };

    // std::cout << nn;
    nn.train();

    std::cout << "=================== end of the program =================" << "\n";

    return 0;
}

#pragma optimize("", on)
