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
    lint threads = 4;
    lint epoch_size = 20;
    std::string dataset_type = "mnist";

    if (argc == 5)
    {
        batch_size = std::stoi(argv[1]);
        threads = std::stoi(argv[2]);
        epoch_size = std::stoi(argv[3]);
        dataset_type = argv[4];
    }

    std::shared_ptr<dataset::Dataset> data_set;

    if ("mnist" == dataset_type)
    {
        data_set = std::make_shared<dataset::Mnist>(
            dataset_dir + "mnist/train-images-idx3-ubyte",
            dataset_dir + "mnist/train-labels-idx1-ubyte",
            dataset_dir + "mnist/t10k-images-idx3-ubyte",
            dataset_dir + "mnist/t10k-labels-idx1-ubyte");
    }
    else if ("fashion-mnist" == dataset_type)
    {
        data_set = std::make_shared<dataset::Mnist>(
            dataset_dir + "fashion-mnist/train-images-idx3-ubyte",
            dataset_dir + "fashion-mnist/train-labels-idx1-ubyte",
            dataset_dir + "fashion-mnist/t10k-images-idx3-ubyte",
            dataset_dir + "fashion-mnist/t10k-labels-idx1-ubyte");
    }
    else if ("cifar-10" == dataset_type)
    {
        data_set = std::make_shared<dataset::CIFAR_10>(dataset_dir + "cifar-10/");
    }
    else
    {
        return 1;
    }

    Multi_Layer_NN nn{ 0.001, batch_size, threads, 10000000, epoch_size, 72000, *data_set };

    // std::cout << nn;
    nn.train();

    std::cout << "=================== end of the program =================" << "\n";

    return 0;
}

#pragma optimize("", on)
