#pragma once

#include "Mnist.h"
#include "CIFAR_10.h"
#include "PGM.h"
#include "Conv_NN.h"
#include "Simple_NN.h"
#include "Multi_Layer_NN.h"
#include "Conv_Pooling_NN.h"

#include <memory>

//#pragma optimize("", off)

std::string dataset_dir = "D:/develop/my_neurons/dataset/";
std::shared_ptr<dataset::Dataset> data_set;
std::string argv_dataset_type = "mnist";
lint argv_batch_size;
lint argv_threads;
lint argv_epoch_size;
std::string argv_mode;


int make_dataset(std::shared_ptr<dataset::Dataset> & data_set, std::string dataset_type)
{
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

    return 0;
}


void parse_args(std::vector<std::string> argv)
{
    argv_batch_size = std::stoi(argv[1]);
    argv_threads = std::stoi(argv[2]);
    argv_epoch_size = std::stoi(argv[3]);
    argv_dataset_type = argv[4];
    argv_mode = argv[5];
}


int run_dnn(int argc, std::vector<std::string> argv)
{
    argv_batch_size = 64;
    argv_threads = 4;
    argv_epoch_size = 100;
    std::string model_file_name = "dnn.dat";
    argv_mode = "test";

    if (argc == 6)
    {
        parse_args(argv);
    }

    int retval = make_dataset(data_set, argv_dataset_type);
    if (retval != 0)
    {
        return retval;
    }

    neurons::global::global_rand_engine.seed(static_cast<unsigned int>(neurons::now_in_seconds()));

    Multi_Layer_NN nn{ 0.001, 0.3, argv_threads, model_file_name, *data_set };
    // nn.print_layers(std::cout);
    // nn.save_layers_as_images();

    if ("train" == argv_mode)
    {
        nn.train_network(argv_batch_size, argv_epoch_size, 200, 20, 7200);
    }
    else
    {
        std::vector<neurons::TMatrix<>> test_inputs;
        std::vector<neurons::TMatrix<>> test_labels;

        data_set->get_test_set(test_inputs, test_labels, 5);
        auto prediction = nn.network_predict(argv_batch_size, test_inputs);

        for (size_t i = 0; i < prediction.size(); ++i)
        {
            test_inputs[i].reshape(test_inputs[i].shape().sub_shape(0, 1));
            std::cout << test_inputs[i] << '\n';
            std::cout << prediction[i] << '\n';
        }

        for (lint i = 0; i < 100; ++i)
            nn.test_network(argv_batch_size, argv_epoch_size);
    }



    std::cout << "=================== end of the program =================" << "\n";

    return 0;
}

//#pragma optimize("", on)


int run_cnn(int argc, std::vector<std::string> argv)
{
    argv_batch_size = 64;
    argv_threads = 4;
    argv_epoch_size = 10;
    std::string model_file_name = "cnn.dat";
    argv_mode = "train";

    if (argc == 6)
    {
        parse_args(argv);
    }

    int retval = make_dataset(data_set, argv_dataset_type);
    if (retval != 0)
    {
        return retval;
    }

    neurons::global::global_rand_engine.seed(static_cast<unsigned int>(neurons::now_in_seconds()));

    Conv_NN nn{ 0.001, 0.3, argv_threads, model_file_name, *data_set };
    // nn.print_layers(std::cout);
    // nn.save_layers_as_images();

    if ("train" == argv_mode)
    {
        nn.train_network(argv_batch_size, argv_epoch_size, 200, 20, 7200);
    }
    else
    {
        std::vector<neurons::TMatrix<>> tests;
        std::vector<neurons::TMatrix<>> labels;
        std::vector<neurons::TMatrix<>> test_inputs;
        std::vector<neurons::TMatrix<>> test_labels;

        data_set->get_test_set(tests, labels);

        for (lint i = 0; i < 10; ++i)
        {
            lint index = rand() % tests.size();
            test_inputs.push_back(tests[index]);
            test_labels.push_back(labels[index]);
        }

        auto prediction = nn.network_predict(argv_batch_size, test_inputs);

        double total = 0;
        double right = 0;
        for (size_t i = 0; i < prediction.size(); ++i)
        {

            test_inputs[i].reshape(test_inputs[i].shape().sub_shape(0, 1));
            std::cout << test_inputs[i] << '\n';
            std::cout << test_labels[i] << '\n';
            std::cout << prediction[i] << '\n';

            if (test_labels[i].argmax() == prediction[i].argmax())
            {
                right += 1;
            }
            total += 1;
        }

        std::cout << "accuracy: " << right / total << '\n';

        for (lint i = 0; i < 100; ++i)
            nn.test_network(argv_batch_size, argv_epoch_size);
    }


    std::cout << "=================== end of the program =================" << "\n";

    return 0;
}


int run_facial_network(
    bool hitopgm,
    bool test_only,
    lint seed,
    lint batch_size,
    lint n_threads,
    lint epoch_size,
    lint n_epochs,
    lint n_epochs_between_save,
    lint label_id,
    double learning_rate,
    double momentum,
    const std::string & network_type,
    const std::string & network_file_name,
    const std::string & network_from_file_name,
    const std::vector<std::string> & train_list,
    const std::vector<std::string> & test_1_list,
    const std::vector<std::string> & test_2_list)
{
    // Load dataset into memory
    dataset::PGM pgm_train_dataset{ train_list, test_1_list, label_id };
    dataset::PGM pgm_test_dataset{ std::vector<std::string>{}, test_2_list, label_id };

    std::unique_ptr<NN> network;

    if (seed < 0)
    {
        neurons::global::global_rand_engine.seed(static_cast<unsigned int>(neurons::now_in_seconds()));
    }
    else
    {
        neurons::global::global_rand_engine.seed(seed);
    }

    if ("cnn" == network_type)
    {
        network = std::make_unique<Conv_NN>(
            learning_rate, momentum, n_threads, "cnn_" + network_file_name, pgm_train_dataset);

        if (!network_from_file_name.empty())
        {
            network->load_until("cnn_" + network_from_file_name, network->n_layers() - 1);
        }
    }
    else if ("dnn" == network_type)
    {
        network = std::make_unique<Multi_Layer_NN>(
            learning_rate, momentum, n_threads, "dnn_" + network_file_name, pgm_train_dataset);

        if (!network_from_file_name.empty())
        {
            network->load_until("dnn_" + network_from_file_name, network->n_layers() - 1);
        }
    }
    else
    {
        std::cout << "No suitable network available, please choose cnn or dnn.\n";
        return 0;
    }

    if (hitopgm)
    {
        network->save_layers_as_images();
        return 1;
    }

    if (!test_only)
    {
        // Train the network
        network->train_network(batch_size, epoch_size, n_epochs, n_epochs_between_save, 36000);
    }

    std::vector<neurons::TMatrix<>> test_input;
    std::vector<neurons::TMatrix<>> test_labels;
    pgm_test_dataset.get_test_set(test_input, test_labels);

    lint n_tests = 0;
    double n_rights = 0;
    std::vector<neurons::TMatrix<>> test_pred = network->network_predict(batch_size, test_input);

    for (size_t i = 0; i < test_labels.size(); ++i)
    {
        //test_input[i].reshape(test_input[i].shape().sub_shape(0, 1));
        //std::cout << test_input[i] << '\n';

        std::cout << "Expected prediction: " << pgm_test_dataset.mat_to_name(test_labels[i]) << "\n";
        std::cout << test_labels[i] << "\n\n";

        std::cout << "Prediction: " << pgm_test_dataset.mat_to_name(test_pred[i]) << "\n";
        std::cout << test_pred[i] << "\n\n";

        std::cout << "--------------------------------------------------------\n";

        n_tests += 1;
        if (test_pred[i].argmax() == test_labels[i].argmax())
        {
            n_rights += 1;
        }
    }

    std::cout << "Test accuracy is: " << n_rights / n_tests << "\n";

    return 1;
}

