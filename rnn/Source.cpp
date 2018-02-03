#include "Simple_RNN.h"
#include "Mnist.h"
#include "Review.h"

#pragma optimize("", off)

int main(int argc, const char * argv[])
{
    std::string dataset_dir = "D:/develop/my_neurons/mnist/";
    lint batch_size = 1;
    lint threads = 1;
    lint epoch_size = 2000;

    if (argc == 4)
    {
        batch_size = std::stoi(argv[1]);
        threads = std::stoi(argv[2]);
        epoch_size = std::stoi(argv[3]);
    }

    dataset::Mnist mnist{
        dataset_dir + "train-images-idx3-ubyte",
        dataset_dir + "train-labels-idx1-ubyte",
        dataset_dir + "t10k-images-idx3-ubyte",
        dataset_dir + "t10k-labels-idx1-ubyte" };

    dataset::Review review{ "D:/develop/my_neurons/rnn_data_set/glove.6B.50d.txt",
        "D:/develop/my_neurons/rnn_data_set/reviews", 0.2, 40 };

    Simple_RNN nn{ 0.001, batch_size, threads, 5000000, epoch_size, 7200, review };

    // std::cout << nn;
    nn.train();

    std::cout << "=================== end of the program =================" << "\n";

    return 0;
}

#pragma optimize("", on)