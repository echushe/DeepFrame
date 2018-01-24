#include "Simple_NN.h"
#include "Multi_Layer_NN.h"

#pragma optimize("", off)

int main(int argc, const char * argv[])
{
    std::string dataset_dir = "D:/develop/my_neurons/mnist/";
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
        dataset_dir + "train-images-idx3-ubyte",
        dataset_dir + "train-labels-idx1-ubyte",
        dataset_dir + "t10k-images-idx3-ubyte",
        dataset_dir + "t10k-labels-idx1-ubyte" };

    Multi_Layer_NN nn{ 0.001, batch_size, threads, 5000000, epoch_size, 7200, mnist };

    // std::cout << nn;
    nn.train();

    std::cout << "=================== end of the program =================" << "\n";


    auto mat = std::make_unique<neurons::Matrix>();
    std::unique_ptr<neurons::Matrix> mat2 = std::move(mat);

    return 0;
}

#pragma optimize("", on)
